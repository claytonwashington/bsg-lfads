#!/usr/bin/env python3
"""Prepare data files for SOC-LFADS training.

Replicates the PCR alignment procedure from auyong_pcr_alignment_v4.py,
but uses lfads_rates (neural) and deEMG_mean (EMG) as the data fields.

Pipeline:
  1. Load NWBDataset pickles (one per session)
  2. Compute trial-aligned cycle averages (PSTHs) using DataWrangler
  3. Concatenate cycle averages across sessions (global space)
  4. Fit global PCA on mean-centered global averages
  5. Ridge regression per session -> readin_weight, readout_bias
  6. Chop continuous data into overlapping windows for training
  7. Save paired H5 files

Usage:
    conda run -n stkit-nwb python scripts/prepare_soc_data.py \\
        --dataset_dir /snel/share/share/tmp/scratch/cbwash2/auyong_nwb/binsize_10ms_pcr_high_reg_ALL_cw \\
        --output_dir datasets/soc_gran \\
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
# PCR alignment functions (following auyong_pcr_alignment_v4.py)
# ---------------------------------------------------------------------------

def aligned_cycle_averaging(dataset, dw, field, field_names=None):
    """Compute cycle-averaged traces aligned to trial onset.

    Returns
    -------
    cycle_avg : ndarray, shape (n_timepoints, n_channels)
    """
    if field_names is None:
        field_names = dataset.data[field].columns.values

    n_time = dw._t_df.align_time.unique().size
    cycle_avg = np.full((n_time, len(field_names)), np.nan)

    for i, fname in enumerate(field_names):
        cycle_aligned_df = dw.pivot_trial_df(dw._t_df, values=(field, fname))
        cycle_avg[:, i] = cycle_aligned_df.mean(axis=1, skipna=True)

    return cycle_avg


def concat_sessions(all_avg, all_means):
    """Concatenate all sessions horizontally to create global space."""
    global_avg = np.concatenate(all_avg, axis=1)
    global_means = np.concatenate(all_means, axis=1)
    return global_avg, global_means


def fit_global_pcs(global_avg, global_means, num_pcs, fit_ix):
    """Fit PCA on mean-centered global cycle averages."""
    pca_obj = PCA(n_components=num_pcs)
    mean_cent = global_avg[fit_ix, :] - global_means
    global_pcs = pca_obj.fit_transform(mean_cent)
    logger.info(f"  PCA explained variance: {np.sum(pca_obj.explained_variance_ratio_):.4f}")
    return pca_obj, global_pcs


def fit_session_readins(all_avg, all_means, global_pcs, fit_ix, l2_scale=0):
    """Fit Ridge regression from session averages to global PCs.

    Returns
    -------
    all_W : list of ndarray, each (n_channels, n_pcs) — readin_weight
    all_b_out : list of ndarray, each (n_channels,) — readout_bias (channel means)
    """
    all_W = []
    all_b_out = []
    for sess_avg, sess_means in zip(all_avg, all_means):
        lr = Ridge(alpha=l2_scale, fit_intercept=False)
        lr.fit(sess_avg[fit_ix, :] - sess_means, global_pcs)
        W = lr.coef_.T  # (n_channels, n_pcs)
        all_W.append(W)
        all_b_out.append(np.squeeze(sess_means))  # (n_channels,)
    return all_W, all_b_out


# ---------------------------------------------------------------------------
# Chopping functions
# ---------------------------------------------------------------------------

def chop_continuous(data: np.ndarray, window: int, overlap: int) -> np.ndarray:
    """Chop continuous (T, C) array into overlapping windows."""
    stride = window - overlap
    n_chops = (len(data) - window) // stride + 1
    chops = np.array([data[i * stride: i * stride + window] for i in range(n_chops)])
    return chops


def train_valid_split(n_chops: int, valid_ratio: float = 0.2):
    """Replicate lfads-torch 4-train/1-valid block split."""
    block_period = round(1 / valid_ratio)
    in_valid = np.arange(n_chops) % block_period == 0
    return np.where(~in_valid)[0], np.where(in_valid)[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_session_id(filename: str) -> str:
    """Extract session ID like '013' from 'nlb_gran_013.pkl'."""
    match = re.search(r"(\d{3})", os.path.basename(filename))
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract session ID from: {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare paired neural/EMG data for SOC-LFADS using proper PCR alignment."
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Directory containing NWBDataset .pkl files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for H5 files")
    parser.add_argument("--cat", type=str, default="gran",
                        help="Category name (e.g. 'gran')")
    parser.add_argument("--neural_field", type=str, default="lfads_rates",
                        help="NWBDataset field for neural data")
    parser.add_argument("--emg_field", type=str, default="deEMG_mean",
                        help="NWBDataset field for EMG data")
    # PCR parameters (matching auyong_pcr_alignment_v4.py)
    parser.add_argument("--num_neural_pcs", type=int, default=20)
    parser.add_argument("--num_emg_pcs", type=int, default=10)
    parser.add_argument("--l2_scale", type=float, default=1e-2)
    parser.add_argument("--align_range_start", type=int, default=-100,
                        help="Align range start in ms")
    parser.add_argument("--align_range_end", type=int, default=600,
                        help="Align range end in ms")
    # Chopping parameters
    parser.add_argument("--window", type=int, default=100, help="Chop window in bins")
    parser.add_argument("--overlap", type=int, default=20, help="Chop overlap in bins")
    parser.add_argument("--valid_ratio", type=float, default=0.2)
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

        neural_data = ds.data[args.neural_field].values
        emg_data = ds.data[args.emg_field].values
        logger.info(f"  {args.neural_field}: {neural_data.shape}")
        logger.info(f"  {args.emg_field}: {emg_data.shape}")

        sessions[sid] = {
            "ds": ds,
            "neural": neural_data,
            "emg": emg_data,
            "neural_names": ds.data[args.neural_field].columns.values,
            "emg_names": ds.data[args.emg_field].columns.values,
        }

    session_ids = sorted(sessions.keys())

    # ===================================================================
    # STEP 1: Compute PCR alignment (cycle-averaged, trial-aligned)
    # Following auyong_pcr_alignment_v4.py exactly
    # ===================================================================
    logger.info(f"\n=== Computing PCR alignment ===")
    logger.info(f"  align_range={align_range}, l2_scale={args.l2_scale}")

    all_neural_cycle_avg = []
    all_neural_chan_means = []
    all_emg_cycle_avg = []
    all_emg_chan_means = []

    for sid in session_ids:
        ds = sessions[sid]["ds"]
        ti = ds.trial_info

        # --- Exclude first 4 and last 2 trials (same as auyong script)
        excluded_trials = [0, 1, 2, 3, ti.trial_id.iloc[-2], ti.trial_id.iloc[-1]]
        excluded = ti.trial_id == -1
        for ex_t in excluded_trials:
            excluded[ti.trial_id == ex_t] = True
        ignore_trials = excluded

        # --- Align data using DataWrangler
        dw = DataWrangler(ds)
        dw.make_trial_data(
            name="onset",
            align_field="start_time",
            align_range=align_range,
            ignored_trials=ignore_trials,
            allow_overlap=True,
            set_t_df=True,
        )

        # --- Cycle average of neural data
        neural_cycle_avg = aligned_cycle_averaging(
            ds, dw, args.neural_field,
            field_names=sessions[sid]["neural_names"],
        )
        neural_chan_means = np.nanmean(neural_cycle_avg, axis=0)[np.newaxis, :]
        all_neural_cycle_avg.append(neural_cycle_avg)
        all_neural_chan_means.append(neural_chan_means)

        # --- Cycle average of EMG data
        emg_cycle_avg = aligned_cycle_averaging(
            ds, dw, args.emg_field,
            field_names=sessions[sid]["emg_names"],
        )
        emg_chan_means = np.nanmean(emg_cycle_avg, axis=0)[np.newaxis, :]
        all_emg_cycle_avg.append(emg_cycle_avg)
        all_emg_chan_means.append(emg_chan_means)

        logger.info(f"  Session {sid}: neural_cycle_avg={neural_cycle_avg.shape}, "
                     f"emg_cycle_avg={emg_cycle_avg.shape}")

    # --- Create global spaces
    global_neural_avg, global_neural_means = concat_sessions(
        all_neural_cycle_avg, all_neural_chan_means
    )
    global_emg_avg, global_emg_means = concat_sessions(
        all_emg_cycle_avg, all_emg_chan_means
    )

    # --- Fit global PCA (exclude NaN rows)
    logger.info(f"\nFitting global PCA for neural (dim={args.num_neural_pcs})...")
    neural_fit_ix = ~np.any(np.isnan(global_neural_avg), axis=1)
    _, global_neural_pcs = fit_global_pcs(
        global_neural_avg, global_neural_means, args.num_neural_pcs, neural_fit_ix
    )

    logger.info(f"Fitting global PCA for EMG (dim={args.num_emg_pcs})...")
    emg_fit_ix = ~np.any(np.isnan(global_emg_avg), axis=1)
    _, global_emg_pcs = fit_global_pcs(
        global_emg_avg, global_emg_means, args.num_emg_pcs, emg_fit_ix
    )

    # --- Fit session readins via Ridge regression
    logger.info(f"\nFitting session readins (Ridge, alpha={args.l2_scale})...")
    all_neural_W, all_neural_bias = fit_session_readins(
        all_neural_cycle_avg, all_neural_chan_means,
        global_neural_pcs, neural_fit_ix, args.l2_scale,
    )
    all_emg_W, all_emg_bias = fit_session_readins(
        all_emg_cycle_avg, all_emg_chan_means,
        global_emg_pcs, emg_fit_ix, args.l2_scale,
    )

    for i, sid in enumerate(session_ids):
        logger.info(f"  Session {sid}: neural_W={all_neural_W[i].shape}, "
                     f"emg_W={all_emg_W[i].shape}")

    # ===================================================================
    # STEP 2: Chop continuous data for training
    # ===================================================================
    logger.info(f"\n=== Chopping continuous data ===")
    logger.info(f"  window={args.window}, overlap={args.overlap}")

    for sid in session_ids:
        data = sessions[sid]
        neural_chops = chop_continuous(data["neural"], args.window, args.overlap)
        emg_chops = chop_continuous(data["emg"], args.window, args.overlap)
        train_inds, valid_inds = train_valid_split(len(neural_chops), args.valid_ratio)

        data["neural_chops"] = neural_chops
        data["emg_chops"] = emg_chops
        data["train_inds"] = train_inds
        data["valid_inds"] = valid_inds

        logger.info(f"  Session {sid}: {len(neural_chops)} chops "
                     f"({len(train_inds)} train, {len(valid_inds)} valid)")

    # ===================================================================
    # STEP 3: Save H5 files
    # ===================================================================
    neural_dir = os.path.join(args.output_dir, "neural")
    emg_dir = os.path.join(args.output_dir, "emg")
    os.makedirs(neural_dir, exist_ok=True)
    os.makedirs(emg_dir, exist_ok=True)

    for i, sid in enumerate(session_ids):
        data = sessions[sid]
        train_inds = data["train_inds"]
        valid_inds = data["valid_inds"]

        # --- Neural H5 ---
        neural_path = os.path.join(neural_dir, f"lfads_torch_readin{sid}_neural.h5")
        with h5py.File(neural_path, "w") as f:
            f.create_dataset("train_encod_data",
                             data=data["neural_chops"][train_inds].astype(np.float32))
            f.create_dataset("train_recon_data",
                             data=data["neural_chops"][train_inds].astype(np.float32))
            f.create_dataset("valid_encod_data",
                             data=data["neural_chops"][valid_inds].astype(np.float32))
            f.create_dataset("valid_recon_data",
                             data=data["neural_chops"][valid_inds].astype(np.float32))
            f.create_dataset("readin_weight",
                             data=all_neural_W[i].astype(np.float64))
            f.create_dataset("readout_bias",
                             data=all_neural_bias[i].astype(np.float64))
        logger.info(f"  Saved: {neural_path}")

        # --- EMG H5 ---
        emg_path = os.path.join(emg_dir, f"lfads_torch_readin{sid}_emg.h5")
        with h5py.File(emg_path, "w") as f:
            f.create_dataset("train_encod_data",
                             data=data["emg_chops"][train_inds].astype(np.float32))
            f.create_dataset("train_recon_data",
                             data=data["emg_chops"][train_inds].astype(np.float32))
            f.create_dataset("valid_encod_data",
                             data=data["emg_chops"][valid_inds].astype(np.float32))
            f.create_dataset("valid_recon_data",
                             data=data["emg_chops"][valid_inds].astype(np.float32))
            f.create_dataset("readin_weight",
                             data=all_emg_W[i].astype(np.float64))
            f.create_dataset("readout_bias",
                             data=all_emg_bias[i].astype(np.float64))
        logger.info(f"  Saved: {emg_path}")

    # --- Summary ---
    logger.info(f"\n=== Summary ===")
    for i, sid in enumerate(session_ids):
        data = sessions[sid]
        logger.info(
            f"  Session {sid}: "
            f"{len(data['train_inds'])} train / {len(data['valid_inds'])} valid, "
            f"neural_dim={data['neural'].shape[1]}, emg_dim={data['emg'].shape[1]}, "
            f"neural_readin={all_neural_W[i].shape}, emg_readin={all_emg_W[i].shape}"
        )
    logger.info(f"\nOutput: {args.output_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
