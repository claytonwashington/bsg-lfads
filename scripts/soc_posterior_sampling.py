#!/usr/bin/env python3
"""Step 1: Run SOC-LFADS inference on ALL chops and save raw outputs to H5.

Run with: conda run -n lfads-torch-cuda12 python scripts/soc_posterior_sampling.py

This saves the chopped model outputs that Step 2 (merge_soc_outputs.py) will
merge back into smooth continuous traces using snel_toolkit's merge_chops.
"""

import sys
from pathlib import Path

BSG_ROOT = str(Path(__file__).resolve().parent.parent)
if BSG_ROOT not in sys.path:
    sys.path.insert(0, BSG_ROOT)

import os
import re
import numpy as np
import torch
import h5py
from glob import glob

# ---------- CONFIG (from shared analysis_config) ----------
from analysis_config import (
    CKPT_PATH, SOC_H5 as OUT_H5, NEURAL_PATTERN, EMG_PATTERN,
    W_PATH, WINDOW, OVERLAP, get_model_config,
)


def chop_continuous(data, window, overlap):
    stride = window - overlap
    n_chops = (len(data) - window) // stride + 1
    return np.array([data[i * stride: i * stride + window] for i in range(n_chops)])


# ===========================================================
# 1. Load model
# ===========================================================
print("Loading model...")
from lfads_torch.soc_model import LFADS_SOC
from lfads_torch.modules.readin_readout import MultisessionReadin
from lfads_torch.modules.readout import MultisessionDualReadout
from lfads_torch.modules.priors import MultivariateNormal, AutoregressiveMultivariateNormal
from lfads_torch.datamodules import SessionBatch

readin = MultisessionReadin(datafile_pattern=NEURAL_PATTERN)
readout = MultisessionDualReadout(
    spike_file_pattern=NEURAL_PATTERN, emg_file_pattern=EMG_PATTERN, soc_N=200
)

cfg = get_model_config()
model = LFADS_SOC(
    encod_data_dim=20, encod_seq_len=100, recon_seq_len=100,
    ext_input_dim=0, ic_enc_seq_len=0,
    ic_enc_dim=cfg["ic_enc_dim"], ci_enc_dim=cfg["ci_enc_dim"], ci_lag=1,
    con_dim=cfg["con_dim"], co_dim=cfg["co_dim"], ic_dim=64,
    soc_N=200, soc_dt=0.5, soc_tau=50.0, soc_r0=20.0, soc_rmax=100.0,
    soc_W_path=W_PATH,
    readin=readin, readout=readout,
    variational=True,
    co_prior=AutoregressiveMultivariateNormal(tau=10.0, nvar=0.1, shape=(cfg["co_dim"],)),
    ic_prior=MultivariateNormal(mean=0, variance=0.1, shape=(64,)),
    ic_post_var_min=1e-4,
    cell_clip=5.0, dropout_rate=0.02,
    loss_scale=1e4, recon_reduce_mean=True,
    lr_scheduler=False, lr_init=4e-3, lr_stop=1e-5,
    lr_decay=0.95, lr_patience=6,
    lr_adam_beta1=0.9, lr_adam_beta2=0.999, lr_adam_epsilon=3.1623e-5,
    weight_decay=0.0,
    l2_start_epoch=0, l2_increase_epoch=50,
    l2_ic_enc_scale=0.0, l2_ci_enc_scale=0.0,
    kl_start_epoch=0, kl_increase_epoch=50,
    kl_ic_scale=0.0, kl_co_scale=0.0,
)

ckpt = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"  Loaded from {CKPT_PATH.name}, device={device}")

# ===========================================================
# 2. Load ALL chopped data from H5s and run inference
# ===========================================================
neural_files = sorted(glob(NEURAL_PATTERN))
emg_files = sorted(glob(EMG_PATTERN))

print(f"\nFound {len(neural_files)} sessions")

with h5py.File(OUT_H5, "w") as out_h5:
    for s_idx, (nf, ef) in enumerate(zip(neural_files, emg_files)):
        sid = re.search(r"readin(\d{3})", nf).group(1)
        print(f"\n  Session {sid} (index {s_idx}):")

        # Load train + valid, interleave back to original order
        with h5py.File(nf, "r") as h:
            train_neural = h["train_encod_data"][()]
            valid_neural = h["valid_encod_data"][()]
        with h5py.File(ef, "r") as h:
            train_emg = h["train_encod_data"][()]
            valid_emg = h["valid_encod_data"][()]

        n_train = train_neural.shape[0]
        n_valid = valid_neural.shape[0]
        n_total = n_train + n_valid

        # Reconstruct original chop order (every 5th is valid)
        block_period = 5
        valid_mask = np.arange(n_total) % block_period == 0
        train_inds = np.where(~valid_mask)[0][:n_train]
        valid_inds = np.where(valid_mask)[0][:n_valid]

        all_neural_chops = np.empty((n_total, WINDOW, train_neural.shape[2]), dtype=np.float32)
        all_emg_chops = np.empty((n_total, WINDOW, train_emg.shape[2]), dtype=np.float32)
        all_neural_chops[train_inds] = train_neural
        all_neural_chops[valid_inds] = valid_neural
        all_emg_chops[train_inds] = train_emg
        all_emg_chops[valid_inds] = valid_emg

        print(f"    {n_total} chops ({n_train} train + {n_valid} valid), "
              f"neural_dim={train_neural.shape[2]}, emg_dim={train_emg.shape[2]}")

        # Run inference in batches
        neural_pred_list, emg_pred_list = [], []
        rates_list, gen_inputs_list, co_means_list = [], [], []

        BATCH = 64
        for start in range(0, n_total, BATCH):
            end = min(start + BATCH, n_total)
            batch_neural = torch.tensor(all_neural_chops[start:end], dtype=torch.float32).to(device)
            ext_input = torch.zeros(batch_neural.shape[0], WINDOW, 0, device=device)

            sb = SessionBatch(
                encod_data=batch_neural,
                recon_data=batch_neural,
                ext_input=ext_input,
                truth=batch_neural,
                sv_mask=torch.ones_like(batch_neural),
            )

            with torch.no_grad():
                output = model.forward({s_idx: sb}, sample_posteriors=False)
                out = output[s_idx]

            neural_pred_list.append(out["neural_pred"].cpu().numpy())
            emg_pred_list.append(out["emg_pred"].cpu().numpy())
            rates_list.append(out["rates"].cpu().numpy())
            gen_inputs_list.append(out["gen_inputs"].cpu().numpy())
            co_means_list.append(out["co_means"].cpu().numpy())

        # Stack
        neural_pred = np.concatenate(neural_pred_list)
        emg_pred = np.concatenate(emg_pred_list)
        rates = np.concatenate(rates_list)
        gen_inputs = np.concatenate(gen_inputs_list)
        co_means = np.concatenate(co_means_list)

        print(f"    Output shapes: neural_pred={neural_pred.shape}, rates={rates.shape}")

        # Save to H5 (all chops in original order)
        grp = out_h5.create_group(f"session_{sid}")
        grp.create_dataset("neural_pred", data=neural_pred)
        grp.create_dataset("emg_pred", data=emg_pred)
        grp.create_dataset("rates", data=rates)
        grp.create_dataset("gen_inputs", data=gen_inputs)
        grp.create_dataset("co_means", data=co_means)
        grp.create_dataset("neural_target", data=all_neural_chops)
        grp.create_dataset("emg_target", data=all_emg_chops)
        grp.attrs["n_total"] = n_total
        grp.attrs["n_train"] = n_train
        grp.attrs["n_valid"] = n_valid
        grp.attrs["neural_dim"] = train_neural.shape[2]
        grp.attrs["emg_dim"] = train_emg.shape[2]
        grp.attrs["session_id"] = sid

    # Save global metadata
    out_h5.attrs["window"] = WINDOW
    out_h5.attrs["overlap"] = OVERLAP

print(f"\nDone! Saved to: {OUT_H5}")
