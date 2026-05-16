#!/usr/bin/env python3
"""Generate presentation-quality plots from the trained SOC-LFADS model.

Uses the snel_toolkit merge_chops algorithm to properly blend overlapping
chops back into smooth continuous traces — no more boundary jumps.

Pipeline:
  1. Load the best v1 checkpoint
  2. Re-chop original continuous data (all chops, not just train/valid)
  3. Run inference on every chop
  4. Merge overlapping chops with power-function blending (merge_chops)
  5. Plot smooth continuous traces

Usage:
    conda run -n lfads-torch-cuda12 python scripts/analyze_soc_merged.py
"""

import sys
from pathlib import Path

# Ensure bsg-lfads code takes priority
BSG_ROOT = str(Path(__file__).resolve().parent.parent)
if BSG_ROOT not in sys.path:
    sys.path.insert(0, BSG_ROOT)

import os
import numpy as np
import torch
import h5py
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
RUN_DIR = Path("/home/cbwash2/bsg-lfads/runs/bsg-lfads/soc_gran/260419_soc_mvp")
CKPT_PATH = RUN_DIR / "lightning_checkpoints" / "536-537.ckpt"
OUT_DIR = Path("/home/cbwash2/bsg-lfads/analysis_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEURAL_PATTERN = "/home/cbwash2/bsg-lfads/datasets/soc_gran/neural/lfads_torch_readin*_neural.h5"
EMG_PATTERN = "/home/cbwash2/bsg-lfads/datasets/soc_gran/emg/lfads_torch_readin*_emg.h5"
W_PATH = "/home/cbwash2/bsg-lfads/weights/W_soc_200.pt"
PKL_DIR = "/snel/share/share/tmp/scratch/cbwash2/auyong_nwb/binsize_10ms_pcr_high_reg_ALL_cw"

# Chopping params (must match prepare_soc_data.py)
WINDOW = 100
OVERLAP = 20
STRIDE = WINDOW - OVERLAP  # 80

DT_MS = 10
EMG_NAMES = [
    "BicL", "BicS", "Brach", "DeltA", "DeltM", "DeltP",
    "Infra", "LatD", "PecM", "SubSc", "SupSp", "TMaj", "TLaLo", "TrLo"
]

# Plot style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ===========================================================
# merge_chops from snel_toolkit/interfaces.py (standalone copy)
# ===========================================================
def merge_chops(data, overlap, orig_len=None, smooth_pwr=2):
    """Merge overlapping chops back into continuous data.

    Uses a power-function ramp to smoothly blend overlap regions.
    This is a standalone copy of snel_toolkit.interfaces.merge_chops.
    """
    merged = []
    full_weight_len = data.shape[1] - 2 * overlap
    if overlap > 0:
        x = np.linspace(1 / overlap, 1 - 1 / overlap, overlap)
        ramp = 1 - x ** smooth_pwr
    else:
        ramp = np.full(0, np.nan)
    ramp = np.expand_dims(ramp, axis=-1)
    split_ixs = np.cumsum([overlap, full_weight_len])
    for i in range(len(data)):
        first, middle, last = np.split(data[i], split_ixs)
        if i == 0:
            last = last * ramp
        elif i == len(data) - 1:
            first = first * (1 - ramp) + merged.pop(-1)
        else:
            first = first * (1 - ramp) + merged.pop(-1)
            last = last * ramp
        merged.extend([first, middle, last])
    if len(merged) < 1:
        n_samples, _, data_dim = data.shape
        merged = [np.empty((n_samples, data_dim))]
    merged = np.concatenate(merged)
    if orig_len is not None and len(merged) < orig_len:
        nans = np.full((orig_len - len(merged), merged.shape[1]), np.nan)
        merged = np.concatenate([merged, nans])
    return merged


def chop_continuous(data, window, overlap):
    """Chop continuous (T, C) array into overlapping windows."""
    stride = window - overlap
    n_chops = (len(data) - window) // stride + 1
    chops = np.array([data[i * stride: i * stride + window] for i in range(n_chops)])
    return chops


def r_squared(y_true, y_pred):
    """R² across all samples and time, per channel, then averaged."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    r2_per_channel = 1 - ss_res / (ss_tot + 1e-10)
    return r2_per_channel.mean(), r2_per_channel


# ===========================================================
# 1. Load model
# ===========================================================
print("Loading model...")
from lfads_torch.soc_model import LFADS_SOC
from lfads_torch.modules.readin_readout import MultisessionReadin
from lfads_torch.modules.readout import MultisessionDualReadout
from lfads_torch.modules.priors import MultivariateNormal, AutoregressiveMultivariateNormal

readin = MultisessionReadin(datafile_pattern=NEURAL_PATTERN)
readout = MultisessionDualReadout(
    spike_file_pattern=NEURAL_PATTERN, emg_file_pattern=EMG_PATTERN, soc_N=200
)
model = LFADS_SOC(
    encod_data_dim=20, encod_seq_len=100, recon_seq_len=100,
    ext_input_dim=0, ic_enc_seq_len=0,
    ic_enc_dim=128, ci_enc_dim=128, ci_lag=1,
    con_dim=64, co_dim=16, ic_dim=64,
    soc_N=200, soc_dt=0.5, soc_tau=50.0, soc_r0=20.0, soc_rmax=100.0,
    soc_W_path=W_PATH,
    readin=readin, readout=readout,
    variational=True,
    co_prior=AutoregressiveMultivariateNormal(tau=10.0, nvar=0.1, shape=(16,)),
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
# 2. Load original continuous data and re-chop ALL
# ===========================================================
print("\nLoading original continuous data from pkl files...")
import _pickle as pickle

pkl_files = sorted(glob(os.path.join(PKL_DIR, "nlb_gran_*.pkl")))
session_ids = []
continuous_data = {}  # {sid: {"neural": (T,C), "emg": (T,C)}}

for pkl_path in pkl_files:
    import re
    sid = re.search(r"(\d{3})", os.path.basename(pkl_path)).group(1)
    session_ids.append(sid)
    with open(pkl_path, "rb") as f:
        ds = pickle.load(f)
    continuous_data[sid] = {
        "neural": ds.data["lfads_rates"].values,
        "emg": ds.data["deEMG_mean"].values,
        "time_index": ds.data.index,
    }
    print(f"  Session {sid}: neural={continuous_data[sid]['neural'].shape}, "
          f"emg={continuous_data[sid]['emg'].shape}")


# ===========================================================
# 3. Chop ALL continuous data and run inference
# ===========================================================
print("\nChopping and running inference on all data...")

from lfads_torch.utils import send_batch_to_device
from lfads_torch.datamodules import SessionBatch

# Load readin weights for creating batches
neural_h5_files = sorted(glob(NEURAL_PATTERN))
emg_h5_files = sorted(glob(EMG_PATTERN))

merged_results = {}

for s_idx, sid in enumerate(session_ids):
    neural_cont = continuous_data[sid]["neural"]
    emg_cont = continuous_data[sid]["emg"]
    T_orig = len(neural_cont)

    # Chop ALL the continuous data (no train/valid split)
    neural_chops = chop_continuous(neural_cont, WINDOW, OVERLAP)
    emg_chops = chop_continuous(emg_cont, WINDOW, OVERLAP)
    n_chops = len(neural_chops)
    print(f"  Session {sid}: {n_chops} total chops from {T_orig} bins")

    # Run inference in batches
    neural_pred_chops = []
    emg_pred_chops = []
    rates_chops = []
    gen_inputs_chops = []
    co_means_chops = []

    BATCH = 64
    for start in range(0, n_chops, BATCH):
        end = min(start + BATCH, n_chops)
        batch_neural = torch.tensor(neural_chops[start:end], dtype=torch.float32).to(device)
        batch_emg = torch.tensor(emg_chops[start:end], dtype=torch.float32).to(device)

        # Create SessionBatch (ext_input is zeros)
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

        neural_pred_chops.append(out["neural_pred"].cpu().numpy())
        emg_pred_chops.append(out["emg_pred"].cpu().numpy())
        rates_chops.append(out["rates"].cpu().numpy())
        gen_inputs_chops.append(out["gen_inputs"].cpu().numpy())
        co_means_chops.append(out["co_means"].cpu().numpy())

    # Stack all chops
    neural_pred_all = np.concatenate(neural_pred_chops, axis=0)
    emg_pred_all = np.concatenate(emg_pred_chops, axis=0)
    rates_all = np.concatenate(rates_chops, axis=0)
    gen_inputs_all = np.concatenate(gen_inputs_chops, axis=0)
    co_means_all = np.concatenate(co_means_chops, axis=0)

    # Merge overlapping chops → continuous (THE KEY STEP)
    merged_neural_pred = merge_chops(neural_pred_all, OVERLAP, T_orig, smooth_pwr=2)
    merged_emg_pred = merge_chops(emg_pred_all, OVERLAP, T_orig, smooth_pwr=2)
    merged_rates = merge_chops(rates_all, OVERLAP, T_orig, smooth_pwr=2)
    merged_gen_inputs = merge_chops(gen_inputs_all, OVERLAP, T_orig, smooth_pwr=2)
    merged_co_means = merge_chops(co_means_all, OVERLAP, T_orig, smooth_pwr=2)

    merged_results[sid] = {
        "neural_pred": merged_neural_pred,
        "emg_pred": merged_emg_pred,
        "rates": merged_rates,
        "gen_inputs": merged_gen_inputs,
        "co_means": merged_co_means,
        "neural_target": neural_cont[:len(merged_neural_pred)],
        "emg_target": emg_cont[:len(merged_emg_pred)],
    }
    print(f"    Merged to continuous: {merged_neural_pred.shape[0]} bins")


# ===========================================================
# 4. Compute per-session R² on CONTINUOUS merged data
# ===========================================================
print("\n=== Per-Session R² (continuous merged data) ===")
for sid in session_ids:
    r = merged_results[sid]
    # Trim NaNs at end if any
    valid = ~np.isnan(r["neural_pred"][:, 0])
    r2n, _ = r_squared(r["neural_target"][valid], r["neural_pred"][valid])
    r2e, _ = r_squared(r["emg_target"][valid], r["emg_pred"][valid])
    print(f"  Session {sid}: R²_neural={r2n:.3f}, R²_emg={r2e:.3f}")


# ===========================================================
# PLOT 1: EMG traces — continuous, smooth
# ===========================================================
print("\nGenerating EMG trace plots...")
sid = session_ids[0]  # Session 013
r = merged_results[sid]

# Pick a 10-second window with clear movement
# Start at ~1000ms to skip the quiet beginning
t_start = 1000  # bins (= 10 seconds into recording)
t_end = t_start + 1500  # 15 seconds of data
time_ms = np.arange(t_start, t_end) * DT_MS

n_emg = r["emg_target"].shape[1]
n_cols = 2
n_rows = (min(n_emg, 8) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.0), sharex=True)
axes = axes.flatten()

for ch in range(min(n_emg, 8)):
    ax = axes[ch]
    ax.plot(time_ms, r["emg_target"][t_start:t_end, ch], "k-", lw=1.0, alpha=0.7, label="Target (deEMG)")
    ax.plot(time_ms, r["emg_pred"][t_start:t_end, ch], color="#E74C3C", lw=1.0, alpha=0.9, label="SOC Pred")
    name = EMG_NAMES[ch] if ch < len(EMG_NAMES) else f"EMG {ch}"
    ax.set_ylabel(name, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ch == 0:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

for i in range(min(n_emg, 8), len(axes)):
    axes[i].set_visible(False)
axes[-2].set_xlabel("Time (ms)")
axes[-1].set_xlabel("Time (ms)")
fig.suptitle(f"EMG Reconstruction — Session {sid} (continuous, merged)", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(OUT_DIR / "emg_traces_merged.png")
plt.close(fig)
print(f"  Saved: emg_traces_merged.png")


# ===========================================================
# PLOT 2: Neural rate traces — continuous, smooth
# ===========================================================
print("Generating neural rate trace plots...")
n_neurons_to_show = 6
neuron_inds = np.linspace(0, r["neural_target"].shape[1] - 1, n_neurons_to_show, dtype=int)

fig, axes = plt.subplots(n_neurons_to_show, 1, figsize=(12, n_neurons_to_show * 1.8), sharex=True)
for i, nidx in enumerate(neuron_inds):
    ax = axes[i]
    ax.plot(time_ms, r["neural_target"][t_start:t_end, nidx], "k-", lw=1.0, alpha=0.7, label="Target (LFADS rates)")
    ax.plot(time_ms, r["neural_pred"][t_start:t_end, nidx], color="#3498DB", lw=1.0, alpha=0.9, label="SOC Pred")
    ax.set_ylabel(f"Neuron {nidx}", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if i == 0:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

axes[-1].set_xlabel("Time (ms)")
fig.suptitle(f"Neural Rate Reconstruction — Session {sid} (continuous, merged)", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(OUT_DIR / "neural_traces_merged.png")
plt.close(fig)
print(f"  Saved: neural_traces_merged.png")


# ===========================================================
# PLOT 3: SOC Population Heatmap + LFADS rates (3 rows)
# ===========================================================
print("Generating SOC population heatmap (3 rows)...")

fig, axes = plt.subplots(3, 1, figsize=(14, 8), height_ratios=[1, 1, 1])

rates_slice = r["rates"][t_start:t_end]
neural_target_slice = r["neural_target"][t_start:t_end]

# Row 1: Excitatory SOC units
im1 = axes[0].imshow(rates_slice[:, :100].T, aspect="auto", cmap="inferno",
                      extent=[time_ms[0], time_ms[-1], 100, 0])
axes[0].set_ylabel("Excitatory (1–100)")
axes[0].set_title("SOC Population vs LFADS Target Rates (continuous, merged)")
plt.colorbar(im1, ax=axes[0], label="Rate (Hz)", shrink=0.8)

# Row 2: Inhibitory SOC units
im2 = axes[1].imshow(rates_slice[:, 100:].T, aspect="auto", cmap="inferno",
                      extent=[time_ms[0], time_ms[-1], 100, 0])
axes[1].set_ylabel("Inhibitory (101–200)")
plt.colorbar(im2, ax=axes[1], label="Rate (Hz)", shrink=0.8)

# Row 3: LFADS target rates (session-specific neurons)
im3 = axes[2].imshow(neural_target_slice.T, aspect="auto", cmap="inferno",
                      extent=[time_ms[0], time_ms[-1], neural_target_slice.shape[1], 0])
axes[2].set_ylabel(f"LFADS Rates ({neural_target_slice.shape[1]} neurons)")
axes[2].set_xlabel("Time (ms)")
plt.colorbar(im3, ax=axes[2], label="Rate", shrink=0.8)

plt.tight_layout()
fig.savefig(OUT_DIR / "soc_population_heatmap_merged.png")
plt.close(fig)
print(f"  Saved: soc_population_heatmap_merged.png")


# ===========================================================
# PLOT 4: R² bar chart (per-session)
# ===========================================================
print("Generating R² bar charts...")

r2_neural_list = []
r2_emg_list = []
for sid in session_ids:
    r = merged_results[sid]
    valid = ~np.isnan(r["neural_pred"][:, 0])
    r2n, _ = r_squared(r["neural_target"][valid], r["neural_pred"][valid])
    r2e, _ = r_squared(r["emg_target"][valid], r["emg_pred"][valid])
    r2_neural_list.append(r2n)
    r2_emg_list.append(r2e)

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(session_ids))
w = 0.35
bars1 = ax.bar(x - w/2, r2_neural_list, w, label="Neural R²", color="#3498DB", edgecolor="white")
bars2 = ax.bar(x + w/2, r2_emg_list, w, label="EMG R²", color="#E74C3C", edgecolor="white")
ax.set_xlabel("Session")
ax.set_ylabel("R² (continuous merged)")
ax.set_title("SOC-LFADS Reconstruction Quality — Per Session (Merged)")
ax.set_xticks(x)
ax.set_xticklabels([f"S{sid}" for sid in session_ids])
ax.legend()
ax.set_ylim(0, 1.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
fig.savefig(OUT_DIR / "r2_bar_chart_merged.png")
plt.close(fig)
print(f"  Saved: r2_bar_chart_merged.png")


# ===========================================================
# PLOT 5: Per-muscle EMG R²
# ===========================================================
print("Generating per-muscle EMG R² plot...")

all_emg_true = np.concatenate([merged_results[s]["emg_target"] for s in session_ids], axis=0)
all_emg_pred = np.concatenate([merged_results[s]["emg_pred"] for s in session_ids], axis=0)
valid_mask = ~np.isnan(all_emg_pred[:, 0])
_, r2_per_muscle = r_squared(all_emg_true[valid_mask], all_emg_pred[valid_mask])

fig, ax = plt.subplots(figsize=(10, 4))
colors = plt.cm.RdYlGn(r2_per_muscle / max(r2_per_muscle.max(), 1.0))
bars = ax.bar(range(len(r2_per_muscle)), r2_per_muscle, color=colors, edgecolor="white")
ax.set_xticks(range(len(EMG_NAMES)))
ax.set_xticklabels(EMG_NAMES, rotation=45, ha="right")
ax.set_ylabel("R² (continuous merged, all sessions)")
ax.set_title("Per-Muscle EMG Reconstruction Quality")
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar, val in zip(bars, r2_per_muscle):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
fig.savefig(OUT_DIR / "emg_r2_per_muscle_merged.png")
plt.close(fig)
print(f"  Saved: emg_r2_per_muscle_merged.png")


# ===========================================================
# PLOT 6: Controller outputs (all 16 dims) + EMG overlay
# ===========================================================
print("Generating controller + EMG plot...")

sid = session_ids[0]
r = merged_results[sid]

fig, axes = plt.subplots(3, 1, figsize=(14, 7), height_ratios=[2, 1, 1],
                          sharex=True, gridspec_kw={"hspace": 0.15})

# Top: all 16 CO dims
co = r["gen_inputs"][t_start:t_end]
for i in range(co.shape[1]):
    axes[0].plot(time_ms, co[:, i], linewidth=0.6, alpha=0.7)
axes[0].set_ylabel("Controller Output")
axes[0].set_title("Controller Outputs (16 dims) vs EMG — Session 013 (merged)")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Middle: BicL (flexor)
axes[1].plot(time_ms, r["emg_target"][t_start:t_end, 0], "k-", lw=1, alpha=0.7, label="Target")
axes[1].plot(time_ms, r["emg_pred"][t_start:t_end, 0], color="#E74C3C", lw=1, alpha=0.9, label="SOC Pred")
axes[1].set_ylabel("BicL")
axes[1].legend(fontsize=7, loc="upper right")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

# Bottom: DeltA (extensor)
axes[2].plot(time_ms, r["emg_target"][t_start:t_end, 3], "k-", lw=1, alpha=0.7, label="Target")
axes[2].plot(time_ms, r["emg_pred"][t_start:t_end, 3], color="#E74C3C", lw=1, alpha=0.9, label="SOC Pred")
axes[2].set_ylabel("DeltA")
axes[2].set_xlabel("Time (ms)")
axes[2].legend(fontsize=7, loc="upper right")
axes[2].spines["top"].set_visible(False)
axes[2].spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(OUT_DIR / "controller_emg_merged.png")
plt.close(fig)
print(f"  Saved: controller_emg_merged.png")


# ===========================================================
# Summary
# ===========================================================
print(f"\n{'='*60}")
print(f"  All merged plots saved to: {OUT_DIR}")
for f in sorted(OUT_DIR.glob("*_merged.png")):
    print(f"    {f.name}")
print(f"{'='*60}")
