#!/usr/bin/env python3
"""Step 2: Merge chopped SOC outputs into smooth continuous traces + generate plots.

Run with: conda run -n stkit-nwb python scripts/merge_soc_outputs.py

Reads the H5 from soc_posterior_sampling.py, applies merge_chops to get smooth
continuous data, saves a merged analysis pkl, and generates presentation plots.
"""

import os
import re
import sys
import logging
from pathlib import Path
from glob import glob

import dill
import h5py
import numpy as np
import pandas as pd
import _pickle as pickle

from snel_toolkit.interfaces import merge_chops

# ---------- CONFIG (from shared analysis_config) ----------
from analysis_config import (
    SOC_H5, PKL_DIR, OUT_DIR, MERGED_PKL as OUT_PKL,
    DT_MS, WINDOW, OVERLAP, STRIDE, BIN_S, EMG_NAMES,
    r_squared, plt, RUN_TAG,
)




# ===========================================================
# 1. Load SOC model outputs (chopped) and original pkl data
# ===========================================================
print("Loading SOC outputs and original datasets...")

soc_h5 = h5py.File(SOC_H5, "r")
WINDOW = int(soc_h5.attrs["window"])
OVERLAP = int(soc_h5.attrs["overlap"])
print(f"  window={WINDOW}, overlap={OVERLAP}")

pkl_files = sorted(glob(os.path.join(PKL_DIR, "nlb_gran_*.pkl")))
session_ids = [re.search(r"(\d{3})", os.path.basename(f)).group(1) for f in pkl_files]

merged_results = {}

for sid, pkl_path in zip(session_ids, pkl_files):
    print(f"\n  Session {sid}:")

    # Load original continuous data
    with open(pkl_path, "rb") as f:
        ds = pickle.load(f)

    neural_cont = ds.data["lfads_rates"].values
    emg_cont = ds.data["deEMG_mean"].values
    spikes_smooth_cont = ds.data["spikes_smooth_30ms"].values
    T_orig = len(neural_cont)
    print(f"    Original continuous: {T_orig} bins")

    # Load chopped outputs
    grp = soc_h5[f"session_{sid}"]
    neural_pred_chops = grp["neural_pred"][()]  # (n_chops, 100, neural_dim)
    emg_pred_chops = grp["emg_pred"][()]
    rates_chops = grp["rates"][()]
    gen_inputs_chops = grp["gen_inputs"][()]
    co_means_chops = grp["co_means"][()]
    neural_target_chops = grp["neural_target"][()]
    emg_target_chops = grp["emg_target"][()]

    n_chops = neural_pred_chops.shape[0]
    print(f"    {n_chops} chops loaded")

    # Merge with smooth blending (smooth_pwr=2: slightly prefer chop ends)
    merged_neural_pred = merge_chops(neural_pred_chops, OVERLAP, T_orig, smooth_pwr=2)
    merged_emg_pred = merge_chops(emg_pred_chops, OVERLAP, T_orig, smooth_pwr=2)
    merged_rates = merge_chops(rates_chops, OVERLAP, T_orig, smooth_pwr=2)
    merged_gen_inputs = merge_chops(gen_inputs_chops, OVERLAP, T_orig, smooth_pwr=2)
    merged_co_means = merge_chops(co_means_chops, OVERLAP, T_orig, smooth_pwr=2)
    # Also merge the targets so they're on the same footing
    merged_neural_target = merge_chops(neural_target_chops, OVERLAP, T_orig, smooth_pwr=2)
    merged_emg_target = merge_chops(emg_target_chops, OVERLAP, T_orig, smooth_pwr=2)

    print(f"    Merged to continuous: {merged_neural_pred.shape[0]} bins")

    merged_results[sid] = {
        "neural_pred": merged_neural_pred,
        "emg_pred": merged_emg_pred,
        "rates": merged_rates,
        "gen_inputs": merged_gen_inputs,
        "co_means": merged_co_means,
        "neural_target": neural_cont[:len(merged_neural_pred)],
        "emg_target": emg_cont[:len(merged_emg_pred)],
        "spikes_smooth": spikes_smooth_cont[:len(merged_neural_pred)],
        "neural_target_merged": merged_neural_target,
        "emg_target_merged": merged_emg_target,
        "time_index": ds.data.index[:len(merged_neural_pred)],
    }

soc_h5.close()

# Save merged analysis object
print(f"\nSaving merged analysis object to {OUT_PKL}...")
with open(OUT_PKL, "wb") as f:
    dill.dump(merged_results, f, protocol=dill.HIGHEST_PROTOCOL)
print("  Done!")


# ===========================================================
# 2. Compute R² on continuous merged data (PURE VALIDATION ONLY)
# ===========================================================
print("\n=== Per-Session R² (pure validation bins only) ===")
r2_neural_list, r2_emg_list = [], []

for sid in session_ids:
    r = merged_results[sid]
    T_orig = len(r["neural_pred"])

    # Identify purely validation bins. A bin is pure validation if it was
    # NEVER part of a training crop.
    is_train_bin = np.zeros(T_orig, dtype=bool)
    n_chops = (T_orig - WINDOW) // (WINDOW - OVERLAP) + 1
    stride = WINDOW - OVERLAP

    for i in range(n_chops):
        # In soc_posterior_sampling.py, valid_mask = i % 5 == 0
        if i % 5 != 0:  # Train chop
            end_idx = min(i * stride + WINDOW, T_orig)
            is_train_bin[i * stride : end_idx] = True

    pure_valid_mask = ~is_train_bin
    # Also ignore NaNs (from edges)
    valid_test = pure_valid_mask & ~np.isnan(r["neural_pred"][:, 0])

    r2n, _ = r_squared(r["neural_target"][valid_test], r["neural_pred"][valid_test])
    r2e, _ = r_squared(r["emg_target"][valid_test], r["emg_pred"][valid_test])
    r2_neural_list.append(r2n)
    r2_emg_list.append(r2e)

    print(f"  Session {sid}: {valid_test.sum()}/{T_orig} pure valid bins "
          f"-> R²_neural={r2n:.3f}, R²_emg={r2e:.3f}")


# ===========================================================
# PLOT 1: EMG traces (continuous, merged, smooth)
# ===========================================================
print("\nGenerating plots...")
sid = session_ids[0]
r = merged_results[sid]

# Pick window with clear movement (skip quiet start)
t_start = 1000
t_end = t_start + 1500
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
# PLOT 2: Neural rate traces (continuous, merged)
# ===========================================================
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
# ===========================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1, 1, 1]})

rates_slice = r["rates"][t_start:t_end] + 20.0
neural_target_slice = r["neural_target"][t_start:t_end] / BIN_S  # convert to Hz
spikes_smooth_slice = r["spikes_smooth"][t_start:t_end] / BIN_S  # convert to Hz

im1 = axes[0].imshow(rates_slice[:, :100].T, aspect="auto", cmap="inferno",
                      extent=[time_ms[0], time_ms[-1], 100, 0])
axes[0].set_ylabel("Excitatory (1–100)")
axes[0].set_title("SOC Population vs LFADS Target Rates (continuous, merged)")
plt.colorbar(im1, ax=axes[0], label="Rate (Hz)", shrink=0.8)

im2 = axes[1].imshow(rates_slice[:, 100:].T, aspect="auto", cmap="inferno",
                      extent=[time_ms[0], time_ms[-1], 100, 0])
axes[1].set_ylabel("Inhibitory (101–200)")
plt.colorbar(im2, ax=axes[1], label="Rate (Hz)", shrink=0.8)

im3 = axes[2].imshow(neural_target_slice.T, aspect="auto", cmap="inferno",
                      extent=[time_ms[0], time_ms[-1], neural_target_slice.shape[1], 0])
axes[2].set_ylabel(f"LFADS Rates ({neural_target_slice.shape[1]})")
plt.colorbar(im3, ax=axes[2], label="Rate (Hz)", shrink=0.8)

im4 = axes[3].imshow(spikes_smooth_slice.T, aspect="auto", cmap="inferno",
                      extent=[time_ms[0], time_ms[-1], spikes_smooth_slice.shape[1], 0])
axes[3].set_ylabel(f"Smoothed Spikes ({spikes_smooth_slice.shape[1]})")
axes[3].set_xlabel("Time (ms)")
plt.colorbar(im4, ax=axes[3], label="Rate (Hz)", shrink=0.8)

plt.tight_layout()
fig.savefig(OUT_DIR / "soc_population_heatmap_merged.png")
plt.close(fig)
print(f"  Saved: soc_population_heatmap_merged.png")


# ===========================================================
# PLOT 4: R² bar chart
# ===========================================================
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
all_emg_true, all_emg_pred = [], []
for sid in session_ids:
    r = merged_results[sid]
    T_orig = len(r["neural_pred"])
    is_train_bin = np.zeros(T_orig, dtype=bool)
    n_chops = (T_orig - WINDOW) // (WINDOW - OVERLAP) + 1
    stride = WINDOW - OVERLAP
    for i in range(n_chops):
        if i % 5 != 0:
            is_train_bin[i * stride : min(i * stride + WINDOW, T_orig)] = True
    valid_test = ~is_train_bin & ~np.isnan(r["emg_pred"][:, 0])
    all_emg_true.append(r["emg_target"][valid_test])
    all_emg_pred.append(r["emg_pred"][valid_test])

all_emg_true = np.concatenate(all_emg_true)
all_emg_pred = np.concatenate(all_emg_pred)
_, r2_per_muscle = r_squared(all_emg_true, all_emg_pred)

fig, ax = plt.subplots(figsize=(10, 4))
colors = plt.cm.RdYlGn(r2_per_muscle / max(r2_per_muscle.max(), 1.0))
bars = ax.bar(range(len(r2_per_muscle)), r2_per_muscle, color=colors, edgecolor="white")
ax.set_xticks(range(len(EMG_NAMES)))
ax.set_xticklabels(EMG_NAMES, rotation=45, ha="right")
ax.set_ylabel("R² (pure valid bins, all sessions)")
ax.set_title("Per-Muscle EMG Reconstruction Quality (Validation Only)")
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
# PLOT 6: Controller + EMG panel
# ===========================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 7),
                          sharex=True, gridspec_kw={"height_ratios": [2, 1, 1], "hspace": 0.15})

co = r["gen_inputs"][t_start:t_end]
for i in range(co.shape[1]):
    axes[0].plot(time_ms, co[:, i], linewidth=0.6, alpha=0.7)
axes[0].set_ylabel("Controller Output")
axes[0].set_title(f"All {co.shape[1]} Controller Outputs vs EMG — Session {sid} (merged)")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[1].plot(time_ms, r["emg_target"][t_start:t_end, 0], "k-", lw=1, alpha=0.7, label="Target")
axes[1].plot(time_ms, r["emg_pred"][t_start:t_end, 0], color="#E74C3C", lw=1, alpha=0.9, label="SOC Pred")
axes[1].set_ylabel("BicL")
axes[1].legend(fontsize=7, loc="upper right")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

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
print(f"  Merged analysis saved to: {OUT_PKL}")
print(f"  Plots saved to: {OUT_DIR}")
for f in sorted(OUT_DIR.glob("*_merged.png")):
    print(f"    {f.name}")
print(f"{'='*60}")
