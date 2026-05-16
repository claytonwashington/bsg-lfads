#!/usr/bin/env python3
"""Generate presentation-quality plots from the trained SOC-LFADS model.

Loads the best checkpoint from v1, runs inference on validation data,
and generates:
  1. EMG trace overlay (predicted vs actual) — the money plot
  2. Neural rate trace overlay for example neurons
  3. SOC population activity heatmap
  4. Per-session R² bar chart (neural + EMG)
  5. Controller output visualization (gain and I_e)

Usage:
    conda run -n lfads-torch-cuda12 python scripts/analyze_soc.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------- CONFIG ----------
RUN_DIR = Path("/home/cbwash2/bsg-lfads/runs/bsg-lfads/soc_gran/260419_soc_mvp")
CKPT_PATH = RUN_DIR / "lightning_checkpoints" / "536-537.ckpt"
OUT_DIR = Path("/home/cbwash2/bsg-lfads/analysis_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same model/data config as v1
DATA_NEURAL = "/home/cbwash2/bsg-lfads/datasets/soc_gran/neural/lfads_torch_readin*_neural.h5"
DATA_EMG = "/home/cbwash2/bsg-lfads/datasets/soc_gran/emg/lfads_torch_readin*_emg.h5"
W_PATH = "/home/cbwash2/bsg-lfads/weights/W_soc_200.pt"

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

DT_MS = 10  # bin size in ms
EMG_NAMES = [
    "BicL", "BicS", "Brach", "DeltA", "DeltM", "DeltP",
    "Infra", "LatD", "PecM", "SubSc", "SupSp", "TMaj", "TLaLo", "TrLo"
]


# ===================================================================
# 1. Load model + data
# ===================================================================
print("Loading model and data...")

from lfads_torch.soc_datamodules import SOCDataModule
from lfads_torch.soc_model import LFADS_SOC
from lfads_torch.modules.readin_readout import MultisessionReadin
from lfads_torch.modules.readout import MultisessionDualReadout
from lfads_torch.modules.priors import MultivariateNormal, AutoregressiveMultivariateNormal

# DataModule
dm = SOCDataModule(spike_file_pattern=DATA_NEURAL, emg_file_pattern=DATA_EMG, batch_size=256)
dm.setup()

# Model (same params as v1)
readin = MultisessionReadin(datafile_pattern=DATA_NEURAL)
readout = MultisessionDualReadout(
    spike_file_pattern=DATA_NEURAL, emg_file_pattern=DATA_EMG, soc_N=200
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

# Load checkpoint
ckpt = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"  Model loaded from {CKPT_PATH.name}, device={device}")


# ===================================================================
# 2. Run inference on validation data
# ===================================================================
print("Running inference on validation data...")

from lfads_torch.utils import send_batch_to_device

val_loader = dm.val_dataloader()
all_results = {}

with torch.no_grad():
    for batch in val_loader:
        batch = send_batch_to_device(batch, device)
        # Get neural-only batch for forward pass
        neural_batch = {s: b[0] for s, b in batch.items()}
        output = model.forward(neural_batch, sample_posteriors=False)

        for s in sorted(batch.keys()):
            sb = batch[s][0]  # SessionBatch
            emg_target = batch[s][1][0]  # EMG target tensor
            out = output[s]

            entry = {
                "neural_target": sb.recon_data.cpu().numpy(),
                "emg_target": emg_target.cpu().numpy(),
                "neural_pred": out["neural_pred"].cpu().numpy(),
                "emg_pred": out["emg_pred"].cpu().numpy(),
                "rates": out["rates"].cpu().numpy(),
                "gen_inputs": out["gen_inputs"].cpu().numpy(),
                "co_means": out["co_means"].cpu().numpy(),
            }
            if s not in all_results:
                all_results[s] = {k: [] for k in entry}
            for k, v in entry.items():
                all_results[s][k].append(v)

# Concatenate all batches
for s in all_results:
    for k in all_results[s]:
        all_results[s][k] = np.concatenate(all_results[s][k], axis=0)
    n = all_results[s]["neural_target"].shape[0]
    print(f"  Session {s}: {n} valid chops")


# ===================================================================
# Helper: compute R²
# ===================================================================
def r_squared(y_true, y_pred):
    """R² across all samples and time, per channel, then averaged."""
    # Flatten (samples, time) for each channel
    yt = y_true.reshape(-1, y_true.shape[-1])
    yp = y_pred.reshape(-1, y_pred.shape[-1])
    ss_res = np.sum((yt - yp) ** 2, axis=0)
    ss_tot = np.sum((yt - yt.mean(axis=0)) ** 2, axis=0)
    r2_per_channel = 1 - ss_res / (ss_tot + 1e-10)
    return r2_per_channel.mean(), r2_per_channel


# ===================================================================
# PLOT 1: EMG trace overlay (the money plot)
# ===================================================================
print("Generating EMG trace plots...")

# Pick session 0 (013) and show a few consecutive chops to form a longer trace
sess = sorted(all_results.keys())[0]
res = all_results[sess]
n_chops_to_show = 4
n_emg = res["emg_target"].shape[-1]
n_cols = 2
n_rows = (min(n_emg, 8) + n_cols - 1) // n_cols  # show up to 8 EMG channels

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.0), sharex=True)
axes = axes.flatten()

time_ms = np.arange(n_chops_to_show * 100) * DT_MS

# Concatenate chops temporally for visualization
emg_true_cat = res["emg_target"][:n_chops_to_show].reshape(-1, n_emg)
emg_pred_cat = res["emg_pred"][:n_chops_to_show].reshape(-1, n_emg)

for ch in range(min(n_emg, 8)):
    ax = axes[ch]
    ax.plot(time_ms, emg_true_cat[:, ch], "k-", linewidth=1.0, alpha=0.7, label="Target")
    ax.plot(time_ms, emg_pred_cat[:, ch], color="#E74C3C", linewidth=1.0, alpha=0.9, label="SOC Pred")
    name = EMG_NAMES[ch] if ch < len(EMG_NAMES) else f"EMG {ch}"
    ax.set_ylabel(name, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ch == 0:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

# Hide unused axes
for i in range(min(n_emg, 8), len(axes)):
    axes[i].set_visible(False)

axes[-2].set_xlabel("Time (ms)")
axes[-1].set_xlabel("Time (ms)")
fig.suptitle(f"EMG Reconstruction — Session {sess} (validation)", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(OUT_DIR / "emg_traces.png")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'emg_traces.png'}")


# ===================================================================
# PLOT 2: Neural rate trace overlay
# ===================================================================
print("Generating neural rate trace plots...")

n_neurons_to_show = 6
neural_true_cat = res["neural_target"][:n_chops_to_show].reshape(-1, res["neural_target"].shape[-1])
neural_pred_cat = res["neural_pred"][:n_chops_to_show].reshape(-1, res["neural_pred"].shape[-1])

fig, axes = plt.subplots(n_neurons_to_show, 1, figsize=(12, n_neurons_to_show * 1.8), sharex=True)
# Pick evenly spaced neurons
neuron_inds = np.linspace(0, neural_true_cat.shape[1] - 1, n_neurons_to_show, dtype=int)

for i, nidx in enumerate(neuron_inds):
    ax = axes[i]
    ax.plot(time_ms, neural_true_cat[:, nidx], "k-", linewidth=1.0, alpha=0.7, label="Target (LFADS rates)")
    ax.plot(time_ms, neural_pred_cat[:, nidx], color="#3498DB", linewidth=1.0, alpha=0.9, label="SOC Pred")
    ax.set_ylabel(f"Neuron {nidx}", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if i == 0:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

axes[-1].set_xlabel("Time (ms)")
fig.suptitle(f"Neural Rate Reconstruction — Session {sess} (validation)", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(OUT_DIR / "neural_traces.png")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'neural_traces.png'}")


# ===================================================================
# PLOT 3: SOC Population Activity Heatmap
# ===================================================================
print("Generating SOC population heatmap...")

rates_cat = res["rates"][:n_chops_to_show].reshape(-1, 200)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[1, 1])

# Excitatory units (first 100)
im = axes[0].imshow(rates_cat[:, :100].T, aspect="auto", cmap="inferno",
                     extent=[0, rates_cat.shape[0] * DT_MS, 100, 0])
axes[0].set_ylabel("Excitatory (1-100)")
axes[0].set_title("SOC Population Firing Rates (validation)")
plt.colorbar(im, ax=axes[0], label="Rate (Hz)", shrink=0.8)

# Inhibitory units (last 100)
im = axes[1].imshow(rates_cat[:, 100:].T, aspect="auto", cmap="inferno",
                     extent=[0, rates_cat.shape[0] * DT_MS, 100, 0])
axes[1].set_ylabel("Inhibitory (101-200)")
axes[1].set_xlabel("Time (ms)")
plt.colorbar(im, ax=axes[1], label="Rate (Hz)", shrink=0.8)

plt.tight_layout()
fig.savefig(OUT_DIR / "soc_population_heatmap.png")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'soc_population_heatmap.png'}")


# ===================================================================
# PLOT 4: R² Bar Chart (per-session)
# ===================================================================
print("Computing per-session R²...")

sessions = sorted(all_results.keys())
r2_neural = []
r2_emg = []
session_labels = []

for s in sessions:
    r = all_results[s]
    r2n_mean, r2n_per = r_squared(r["neural_target"], r["neural_pred"])
    r2e_mean, r2e_per = r_squared(r["emg_target"], r["emg_pred"])
    r2_neural.append(r2n_mean)
    r2_emg.append(r2e_mean)
    session_labels.append(f"S{s}")
    print(f"  Session {s}: R²_neural={r2n_mean:.3f}, R²_emg={r2e_mean:.3f}")

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(sessions))
w = 0.35
bars1 = ax.bar(x - w/2, r2_neural, w, label="Neural R²", color="#3498DB", edgecolor="white")
bars2 = ax.bar(x + w/2, r2_emg, w, label="EMG R²", color="#E74C3C", edgecolor="white")

ax.set_xlabel("Session")
ax.set_ylabel("R² (validation)")
ax.set_title("SOC-LFADS Reconstruction Quality — Per Session")
ax.set_xticks(x)
ax.set_xticklabels(session_labels)
ax.legend()
ax.set_ylim(0, 1.0)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR / "r2_bar_chart.png")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'r2_bar_chart.png'}")


# ===================================================================
# PLOT 5: Per-muscle EMG R² bar chart
# ===================================================================
print("Generating per-muscle EMG R² plot...")

# Use all sessions pooled for per-muscle R²
all_emg_true = np.concatenate([all_results[s]["emg_target"] for s in sessions], axis=0)
all_emg_pred = np.concatenate([all_results[s]["emg_pred"] for s in sessions], axis=0)
_, r2_per_muscle = r_squared(all_emg_true, all_emg_pred)

fig, ax = plt.subplots(figsize=(10, 4))
colors = plt.cm.RdYlGn(r2_per_muscle / max(r2_per_muscle.max(), 1.0))
bars = ax.bar(range(len(r2_per_muscle)), r2_per_muscle, color=colors, edgecolor="white")
ax.set_xticks(range(len(EMG_NAMES)))
ax.set_xticklabels(EMG_NAMES, rotation=45, ha="right")
ax.set_ylabel("R² (validation, all sessions pooled)")
ax.set_title("Per-Muscle EMG Reconstruction Quality")
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for bar, val in zip(bars, r2_per_muscle):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR / "emg_r2_per_muscle.png")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'emg_r2_per_muscle.png'}")


# ===================================================================
# PLOT 6: Controller output dynamics (gen_inputs)
# ===================================================================
print("Generating controller output plot...")

gen_inputs_cat = res["gen_inputs"][:n_chops_to_show].reshape(-1, res["gen_inputs"].shape[-1])
n_co = gen_inputs_cat.shape[-1]

fig, ax = plt.subplots(figsize=(12, 4))
for i in range(min(n_co, 8)):
    ax.plot(time_ms, gen_inputs_cat[:, i], linewidth=0.8, alpha=0.8, label=f"CO dim {i}")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Controller Output")
ax.set_title("Controller Output Dimensions (drives gain and I_e)")
ax.legend(fontsize=7, ncol=4, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(OUT_DIR / "controller_outputs.png")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'controller_outputs.png'}")


# ===================================================================
# Summary
# ===================================================================
print(f"\n{'='*60}")
print(f"  All plots saved to: {OUT_DIR}")
print(f"  Files:")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"    {f.name}")
print(f"{'='*60}")
