#!/usr/bin/env python3
"""Prepare SOC-LFADS data with BLOCK hold-out validation (non-overlapping).

Unlike the standard interleaved split (every 5th chop → valid), this script
splits each session's continuous timeseries into contiguous blocks:

    Training:   first 80% of timesteps  →  chopped
    Gap:        100 bins (1 second) discarded to guarantee zero overlap
    Validation: remaining ~20% of timesteps  →  chopped independently

PCR alignment is IDENTICAL to the original prepare_soc_data.py — only the
train/valid chopping strategy changes.

Usage:
    conda run -n stkit-nwb python scripts/prepare_soc_data_block.py \\
        --dataset_dir /snel/share/share/tmp/scratch/cbwash2/auyong_nwb/binsize_10ms_pcr_high_reg_ALL_cw \\
        --output_dir datasets/soc_gran_block \\
        --cat gran
"""

from __future__ import annotations

import argparse
import os
import re
import logging
import sys
from glob import glob

import _pickle as pickle
import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from snel_toolkit.datasets.base import DataWrangler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# ---------------------------------------------------------------------------
# PCR alignment functions (IDENTICAL to prepare_soc_data.py)
# ---------------------------------------------------------------------------

def aligned_cycle_averaging(dataset, dw, field, field_names=None):
    """Compute cycle-averaged traces aligned to trial onset."""
    if field_names is None:
        field_names = dataset.data[field].columns.values
    n_time = dw._t_df.align_time.unique().size
    cycle_avg = np.full((n_time, len(field_names)), np.nan)
    for i, fname in enumerate(field_names):
        cycle_aligned_df = dw.pivot_trial_df(dw._t_df, values=(field, fname))
        cycle_avg[:, i] = cycle_aligned_df.mean(axis=1, skipna=True)
    return cycle_avg


def concat_sessions(all_avg, all_means):
    global_avg = np.concatenate(all_avg, axis=1)
    global_means = np.concatenate(all_means, axis=1)
    return global_avg, global_means


def fit_global_pcs(global_avg, global_means, num_pcs, fit_ix):
    pca_obj = PCA(n_components=num_pcs)
    mean_cent = global_avg[fit_ix, :] - global_means
    global_pcs = pca_obj.fit_transform(mean_cent)
    logger.info(f"  PCA explained variance: {np.sum(pca_obj.explained_variance_ratio_):.4f}")
    return pca_obj, global_pcs


def fit_session_readins(all_avg, all_means, global_pcs, fit_ix, l2_scale=0):
    all_W = []
    all_b_out = []
    for sess_avg, sess_means in zip(all_avg, all_means):
        lr = Ridge(alpha=l2_scale, fit_intercept=False)
        lr.fit(sess_avg[fit_ix, :] - sess_means, global_pcs)
        W = lr.coef_.T
        all_W.append(W)
        all_b_out.append(np.squeeze(sess_means))
    return all_W, all_b_out


def chop_continuous(data: np.ndarray, window: int, overlap: int) -> np.ndarray:
    """Chop continuous (T, C) array into overlapping windows."""
    stride = window - overlap
    n_chops = (len(data) - window) // stride + 1
    chops = np.array([data[i * stride: i * stride + window] for i in range(n_chops)])
    return chops


def extract_session_id(filename: str) -> str:
    match = re.search(r"(\d{3})", os.path.basename(filename))
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract session ID from: {filename}")


# ---------------------------------------------------------------------------
# Block hold-out split (THE KEY DIFFERENCE)
# ---------------------------------------------------------------------------

def block_train_valid_split(data: np.ndarray, window: int, overlap: int,
                            train_frac: float = 0.80, gap_bins: int = 100):
    """Split continuous data into contiguous train/valid blocks, then chop each.

    Parameters
    ----------
    data : ndarray, shape (T, C)
    window : int — chop window size
    overlap : int — chop overlap
    train_frac : float — fraction of total timesteps for training
    gap_bins : int — dead zone between train and valid to guarantee zero overlap

    Returns
    -------
    train_chops, valid_chops : ndarray, each (n_chops, window, C)
    split_info : dict with split details for logging
    """
    T = len(data)
    train_end = int(T * train_frac)
    valid_start = train_end + gap_bins

    if valid_start + window > T:
        # Not enough data for even one valid chop — reduce gap
        gap_bins = max(0, T - train_end - window)
        valid_start = train_end + gap_bins
        logger.warning(f"  Reduced gap to {gap_bins} bins to fit at least one valid chop")

    train_data = data[:train_end]
    valid_data = data[valid_start:]

    train_chops = chop_continuous(train_data, window, overlap)
    valid_chops = chop_continuous(valid_data, window, overlap)

    split_info = {
        "total_bins": T,
        "train_bins": train_end,
        "gap_bins": gap_bins,
        "valid_bins": len(valid_data),
        "train_chops": len(train_chops),
        "valid_chops": len(valid_chops),
    }
    return train_chops, valid_chops, split_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare SOC-LFADS data with block hold-out validation."
    )
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cat", type=str, default="gran")
    parser.add_argument("--neural_field", type=str, default="lfads_rates")
    parser.add_argument("--emg_field", type=str, default="deEMG_mean")
    parser.add_argument("--num_neural_pcs", type=int, default=20)
    parser.add_argument("--num_emg_pcs", type=int, default=10)
    parser.add_argument("--l2_scale", type=float, default=1e-2)
    parser.add_argument("--align_range_start", type=int, default=-100)
    parser.add_argument("--align_range_end", type=int, default=600)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--overlap", type=int, default=20)
    parser.add_argument("--train_frac", type=float, default=0.80)
    parser.add_argument("--gap_bins", type=int, default=100,
                        help="Dead zone between train and valid blocks (bins)")
    args = parser.parse_args()

    align_range = (args.align_range_start, args.align_range_end)

    # --- Find and load dataset files ---
    pkl_pattern = os.path.join(args.dataset_dir, f"nlb_{args.cat}_*.pkl")
    pkl_files = sorted(glob(pkl_pattern))
    if not pkl_files:
        logger.error(f"No files matching {pkl_pattern}")
        return
    logger.info(f"Found {len(pkl_files)} dataset files")

    sessions = {}
    for pkl_path in pkl_files:
        sid = extract_session_id(pkl_path)
        logger.info(f"Loading session {sid}...")
        with open(pkl_path, "rb") as f:
            ds = pickle.load(f)
        sessions[sid] = {
            "ds": ds,
            "neural": ds.data[args.neural_field].values,
            "emg": ds.data[args.emg_field].values,
            "neural_names": ds.data[args.neural_field].columns.values,
            "emg_names": ds.data[args.emg_field].columns.values,
        }

    session_ids = sorted(sessions.keys())

    # ===================================================================
    # STEP 1: PCR alignment (IDENTICAL to original)
    # ===================================================================
    logger.info(f"\n=== Computing PCR alignment ===")
    all_neural_cycle_avg, all_neural_chan_means = [], []
    all_emg_cycle_avg, all_emg_chan_means = [], []

    for sid in session_ids:
        ds = sessions[sid]["ds"]
        ti = ds.trial_info
        excluded_trials = [0, 1, 2, 3, ti.trial_id.iloc[-2], ti.trial_id.iloc[-1]]
        excluded = ti.trial_id == -1
        for ex_t in excluded_trials:
            excluded[ti.trial_id == ex_t] = True

        dw = DataWrangler(ds)
        dw.make_trial_data(
            name="onset", align_field="start_time", align_range=align_range,
            ignored_trials=excluded, allow_overlap=True, set_t_df=True,
        )

        neural_ca = aligned_cycle_averaging(ds, dw, args.neural_field,
                                            sessions[sid]["neural_names"])
        all_neural_cycle_avg.append(neural_ca)
        all_neural_chan_means.append(np.nanmean(neural_ca, axis=0)[np.newaxis, :])

        emg_ca = aligned_cycle_averaging(ds, dw, args.emg_field,
                                         sessions[sid]["emg_names"])
        all_emg_cycle_avg.append(emg_ca)
        all_emg_chan_means.append(np.nanmean(emg_ca, axis=0)[np.newaxis, :])

        logger.info(f"  Session {sid}: neural_ca={neural_ca.shape}, emg_ca={emg_ca.shape}")

    gnav, gnm = concat_sessions(all_neural_cycle_avg, all_neural_chan_means)
    geav, gem = concat_sessions(all_emg_cycle_avg, all_emg_chan_means)

    logger.info(f"\nFitting global PCA for neural (dim={args.num_neural_pcs})...")
    nfix = ~np.any(np.isnan(gnav), axis=1)
    _, gnpcs = fit_global_pcs(gnav, gnm, args.num_neural_pcs, nfix)

    logger.info(f"Fitting global PCA for EMG (dim={args.num_emg_pcs})...")
    efix = ~np.any(np.isnan(geav), axis=1)
    _, gepcs = fit_global_pcs(geav, gem, args.num_emg_pcs, efix)

    logger.info(f"\nFitting session readins (Ridge, alpha={args.l2_scale})...")
    all_nW, all_nb = fit_session_readins(all_neural_cycle_avg, all_neural_chan_means,
                                          gnpcs, nfix, args.l2_scale)
    all_eW, all_eb = fit_session_readins(all_emg_cycle_avg, all_emg_chan_means,
                                          gepcs, efix, args.l2_scale)

    # ===================================================================
    # STEP 2: Block hold-out chopping (THE DIFFERENCE)
    # ===================================================================
    logger.info(f"\n=== Block hold-out chopping ===")
    logger.info(f"  window={args.window}, overlap={args.overlap}, "
                f"train_frac={args.train_frac}, gap_bins={args.gap_bins}")

    neural_dir = os.path.join(args.output_dir, "neural")
    emg_dir = os.path.join(args.output_dir, "emg")
    os.makedirs(neural_dir, exist_ok=True)
    os.makedirs(emg_dir, exist_ok=True)

    for i, sid in enumerate(session_ids):
        data = sessions[sid]

        # Split and chop neural
        n_train, n_valid, info = block_train_valid_split(
            data["neural"], args.window, args.overlap,
            args.train_frac, args.gap_bins,
        )
        # Split and chop EMG (same split points)
        e_train, e_valid, _ = block_train_valid_split(
            data["emg"], args.window, args.overlap,
            args.train_frac, args.gap_bins,
        )

        logger.info(f"  Session {sid}: {info['total_bins']} bins → "
                     f"train={info['train_bins']}({info['train_chops']} chops), "
                     f"gap={info['gap_bins']}, "
                     f"valid={info['valid_bins']}({info['valid_chops']} chops)")

        # --- Save Neural H5 ---
        npath = os.path.join(neural_dir, f"lfads_torch_readin{sid}_neural.h5")
        with h5py.File(npath, "w") as f:
            f.create_dataset("train_encod_data", data=n_train.astype(np.float32))
            f.create_dataset("train_recon_data", data=n_train.astype(np.float32))
            f.create_dataset("valid_encod_data", data=n_valid.astype(np.float32))
            f.create_dataset("valid_recon_data", data=n_valid.astype(np.float32))
            f.create_dataset("readin_weight", data=all_nW[i].astype(np.float64))
            f.create_dataset("readout_bias", data=all_nb[i].astype(np.float64))
        logger.info(f"    Saved: {npath}")

        # --- Save EMG H5 ---
        epath = os.path.join(emg_dir, f"lfads_torch_readin{sid}_emg.h5")
        with h5py.File(epath, "w") as f:
            f.create_dataset("train_encod_data", data=e_train.astype(np.float32))
            f.create_dataset("train_recon_data", data=e_train.astype(np.float32))
            f.create_dataset("valid_encod_data", data=e_valid.astype(np.float32))
            f.create_dataset("valid_recon_data", data=e_valid.astype(np.float32))
            f.create_dataset("readin_weight", data=all_eW[i].astype(np.float64))
            f.create_dataset("readout_bias", data=all_eb[i].astype(np.float64))
        logger.info(f"    Saved: {epath}")

    logger.info(f"\nDone! Output: {args.output_dir}")


if __name__ == "__main__":
    main()
