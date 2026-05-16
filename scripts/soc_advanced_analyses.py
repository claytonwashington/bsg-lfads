#!/usr/bin/env python3
"""Advanced SOC-LFADS analyses: PC comparison, gain covariance, cascade plot.

Run with: conda run -n stkit-nwb python scripts/soc_advanced_analyses.py

Loads:
  - soc_merged_analysis.pkl (merged continuous outputs)
  - gain_projection_weights.npz (controller→gain/I_e weights)

Generates:
  1. PC comparison: Procrustes-aligned PCs of controller vs EMG vs neural
  2. Gain covariance matrix: (200×200) sorted by E/I
  3. Peak-sorted cascade heatmap: neurons sorted by peak firing time
"""

import os
import sys
import numpy as np
import _pickle as pickle
import dill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

# ---------- CONFIG ----------
RUN_DIR = "/home/cbwash2/bsg-lfads/runs/bsg-lfads/soc_gran/260419_soc_mvp"
MERGED_PKL = os.path.join(RUN_DIR, "soc_merged_analysis.pkl")
GAIN_NPZ = os.path.join(RUN_DIR, "gain_projection_weights.npz")
OUT_DIR = "/home/cbwash2/bsg-lfads/analysis_plots"
os.makedirs(OUT_DIR, exist_ok=True)

DT_MS = 10
EMG_NAMES = [
    "BicL", "BicS", "Brach", "DeltA", "DeltM", "DeltP",
    "Infra", "LatD", "PecM", "SubSc", "SupSp", "TMaj", "TLaLo", "TrLo"
]

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


def softplus(x):
    """Numpy softplus: log(1 + exp(x)), numerically stable."""
    return np.where(x > 20, x, np.log1p(np.exp(x)))


# ===========================================================
# Load data
# ===========================================================
print("Loading merged analysis...")
with open(MERGED_PKL, "rb") as f:
    merged = dill.load(f)

print("Loading gain projection weights...")
gain_data = np.load(GAIN_NPZ)
g_weight = gain_data["g_weight"]  # (200, 16)
g_bias = gain_data["g_bias"]      # (200,)
ie_weight = gain_data["ie_weight"]
ie_bias = gain_data["ie_bias"]

session_ids = sorted(merged.keys())
sid = session_ids[0]  # Primary session for detailed plots
r = merged[sid]

print(f"  Session {sid}: {r['neural_pred'].shape[0]} bins")
print(f"  Available keys: {list(r.keys())}")


# ===========================================================
# Reconstruct gains from gen_inputs
# ===========================================================
print("\nReconstructing per-neuron gains from controller outputs...")

gains_all = {}
ie_all = {}

for sid_loop in session_ids:
    co = merged[sid_loop]["gen_inputs"]  # (T, 16)
    # Mask NaN rows (from merge_chops tail padding)
    valid_rows = ~np.isnan(co[:, 0])
    co_clean = np.where(np.isnan(co), 0.0, co)  # zero-fill for matmul, mask after
    # Apply linear projection: g = softplus(co @ g_weight.T + g_bias)
    g_raw = co_clean @ g_weight.T + g_bias  # (T, 200)
    g = softplus(g_raw)
    ie = co_clean @ ie_weight.T + ie_bias  # (T, 200)
    # Re-insert NaN for invalid rows
    g[~valid_rows] = np.nan
    ie[~valid_rows] = np.nan
    gains_all[sid_loop] = g
    ie_all[sid_loop] = ie
    print(f"  Session {sid_loop}: gains range [{np.nanmin(g):.2f}, {np.nanmax(g):.2f}], "
          f"I_e range [{np.nanmin(ie):.2f}, {np.nanmax(ie):.2f}], "
          f"valid bins: {valid_rows.sum()}/{len(valid_rows)}")


# ===========================================================
# PLOT A: PC comparison — Controller vs EMG vs Neural
# ===========================================================
print("\n--- PC Comparison (Procrustes-aligned) ---")

n_pcs = 3
t_start, t_end = 1000, 6500  # use a long window for good PCA

co_data = r["gen_inputs"][t_start:t_end]
emg_data = r["emg_target"][t_start:t_end]
neural_data = r["neural_target"][t_start:t_end]

# Remove NaNs (from edges)
valid = ~np.isnan(co_data[:, 0]) & ~np.isnan(emg_data[:, 0]) & ~np.isnan(neural_data[:, 0])
co_data = co_data[valid]
emg_data = emg_data[valid]
neural_data = neural_data[valid]

# PCA on each
pca_co = PCA(n_components=n_pcs).fit_transform(co_data)
pca_emg = PCA(n_components=n_pcs).fit_transform(emg_data)
pca_neural = PCA(n_components=n_pcs).fit_transform(neural_data)

# Procrustes align controller PCs to EMG and neural PCs
# procrustes returns (mtx1_standardized, mtx2_standardized, disparity)
_, pca_co_to_emg, disp_emg = procrustes(pca_emg, pca_co)
_, pca_co_to_neural, disp_neural = procrustes(pca_neural, pca_co)

# Also standardize the targets for consistent scale
pca_emg_std, _, _ = procrustes(pca_emg, pca_emg)  # self-standardize
pca_neural_std, _, _ = procrustes(pca_neural, pca_neural)

time_ms = np.arange(len(pca_co)) * DT_MS

fig, axes = plt.subplots(n_pcs, 2, figsize=(16, n_pcs * 2.5), sharex=True)

for pc in range(n_pcs):
    # Left: Controller vs EMG
    ax = axes[pc, 0]
    ax.plot(time_ms, pca_emg_std[:, pc], "k-", lw=1.0, alpha=0.7, label="EMG PCs")
    ax.plot(time_ms, pca_co_to_emg[:, pc], color="#E74C3C", lw=0.8, alpha=0.8, label="Controller PCs (aligned)")
    ax.set_ylabel(f"PC {pc+1}", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if pc == 0:
        ax.set_title(f"Controller → EMG  (disparity={disp_emg:.3f})", fontsize=12)
        ax.legend(fontsize=8, loc="upper right")

    # Right: Controller vs Neural
    ax = axes[pc, 1]
    ax.plot(time_ms, pca_neural_std[:, pc], "k-", lw=1.0, alpha=0.7, label="Neural PCs")
    ax.plot(time_ms, pca_co_to_neural[:, pc], color="#3498DB", lw=0.8, alpha=0.8, label="Controller PCs (aligned)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if pc == 0:
        ax.set_title(f"Controller → Neural  (disparity={disp_neural:.3f})", fontsize=12)
        ax.legend(fontsize=8, loc="upper right")

axes[-1, 0].set_xlabel("Time (ms)")
axes[-1, 1].set_xlabel("Time (ms)")
fig.suptitle("Procrustes-Aligned PC Comparison (Controller vs EMG/Neural)", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pc_comparison_procrustes.png"))
plt.close(fig)
print(f"  Saved: pc_comparison_procrustes.png  (disp_emg={disp_emg:.3f}, disp_neural={disp_neural:.3f})")


# ===========================================================
# PLOT B: Gain Covariance Matrix
# ===========================================================
print("\n--- Gain Covariance Matrix ---")

# Pool gains across all sessions
all_gains = np.concatenate([gains_all[s] for s in session_ids], axis=0)

# Remove any NaN rows
valid_mask = ~np.isnan(all_gains[:, 0])
all_gains = all_gains[valid_mask]
print(f"  Total gain samples: {all_gains.shape[0]}")

# Compute covariance
gain_cov = np.cov(all_gains.T)  # (200, 200)

# Sort: E units first (0-99), I units second (100-199)
# Already in this order by construction

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5),
                          gridspec_kw={"width_ratios": [1, 1, 0.05]})

# Full matrix
vmax = np.percentile(np.abs(gain_cov), 99)
im = axes[0].imshow(gain_cov, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
axes[0].axhline(y=99.5, color="white", linewidth=1.5, linestyle="--")
axes[0].axvline(x=99.5, color="white", linewidth=1.5, linestyle="--")
axes[0].set_xlabel("Neuron index")
axes[0].set_ylabel("Neuron index")
axes[0].set_title("Gain Covariance (200×200)\nE | I partition")

# Add quadrant labels
for (y, x, label) in [(50, 50, "E↔E"), (50, 150, "E↔I"), (150, 50, "I↔E"), (150, 150, "I↔I")]:
    axes[0].text(x, y, label, ha="center", va="center", fontsize=11,
                 color="white", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.4))

# Diagonal: gain variance per neuron
axes[1].bar(range(100), np.diag(gain_cov)[:100], color="#2ECC71", alpha=0.8, label="Excitatory")
axes[1].bar(range(100, 200), np.diag(gain_cov)[100:], color="#9B59B6", alpha=0.8, label="Inhibitory")
axes[1].set_xlabel("Neuron index")
axes[1].set_ylabel("Gain variance")
axes[1].set_title("Per-Neuron Gain Variance")
axes[1].legend(fontsize=9)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.colorbar(im, cax=axes[2], label="Covariance")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gain_covariance_matrix.png"))
plt.close(fig)

# Summary stats
ee_block = gain_cov[:100, :100]
ii_block = gain_cov[100:, 100:]
ei_block = gain_cov[:100, 100:]
print(f"  Saved: gain_covariance_matrix.png")
print(f"  Mean |cov| — E↔E: {np.abs(ee_block).mean():.4f}, "
      f"I↔I: {np.abs(ii_block).mean():.4f}, E↔I: {np.abs(ei_block).mean():.4f}")


# ===========================================================
# PLOT C: Peak-Sorted Cascade Heatmap
# ===========================================================
print("\n--- Peak-Sorted Cascade Heatmap ---")

# Pick a single locomotion cycle window — look for a ~500ms burst region
# Use session 013, around the first clear movement (~1200-1700 bins = 12-17s)
cascade_start = 1300
cascade_end = 1600  # 3 seconds
time_slice = np.arange(cascade_start, cascade_end) * DT_MS

rates = r["rates"][cascade_start:cascade_end]        # (300, 200)
neural_target = r["neural_target"][cascade_start:cascade_end]  # (300, n_neurons)

# Split E and I
e_rates = rates[:, :100]   # (T, 100) excitatory
i_rates = rates[:, 100:]   # (T, 100) inhibitory

# Normalize each signal to [0, 1] per neuron for comparable sorting
def normalize_cols(X):
    mn = X.min(axis=0, keepdims=True)
    mx = X.max(axis=0, keepdims=True)
    rng = mx - mn
    rng[rng < 1e-10] = 1.0
    return (X - mn) / rng

e_norm = normalize_cols(e_rates)
i_norm = normalize_cols(i_rates)
n_norm = normalize_cols(neural_target)

# Sort by peak firing time (argmax along time axis)
e_peak_order = np.argsort(np.argmax(e_norm, axis=0))
i_peak_order = np.argsort(np.argmax(i_norm, axis=0))
n_peak_order = np.argsort(np.argmax(n_norm, axis=0))

fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)

# Excitatory
im1 = axes[0].imshow(e_norm[:, e_peak_order].T, aspect="auto", cmap="magma",
                      extent=[time_slice[0], time_slice[-1], 100, 0])
axes[0].set_ylabel("E units (sorted by peak time)")
axes[0].set_xlabel("Time (ms)")
axes[0].set_title("Excitatory SOC Units")

# Inhibitory
im2 = axes[1].imshow(i_norm[:, i_peak_order].T, aspect="auto", cmap="magma",
                      extent=[time_slice[0], time_slice[-1], 100, 0])
axes[1].set_ylabel("I units (sorted by peak time)")
axes[1].set_xlabel("Time (ms)")
axes[1].set_title("Inhibitory SOC Units")

# LFADS target rates
im3 = axes[2].imshow(n_norm[:, n_peak_order].T, aspect="auto", cmap="magma",
                      extent=[time_slice[0], time_slice[-1], neural_target.shape[1], 0])
axes[2].set_ylabel("LFADS neurons (sorted by peak time)")
axes[2].set_xlabel("Time (ms)")
axes[2].set_title(f"LFADS Target Rates ({neural_target.shape[1]} neurons)")

fig.suptitle("Peak-Sorted Cascade — Do units cascade during locomotion?", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "peak_sorted_cascade.png"))
plt.close(fig)
print(f"  Saved: peak_sorted_cascade.png")


# ===========================================================
# PLOT D: Gain time series (sample neurons)
# ===========================================================
print("\n--- Gain Time Series ---")

g = gains_all[sid]
t_start_plot, t_end_plot = 1000, 2500
time_plot = np.arange(t_start_plot, t_end_plot) * DT_MS

fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True,
                          gridspec_kw={"hspace": 0.15})

# Top: mean gain for E and I populations
axes[0].plot(time_plot, g[t_start_plot:t_end_plot, :100].mean(axis=1),
             color="#2ECC71", lw=1.5, label="E mean gain")
axes[0].plot(time_plot, g[t_start_plot:t_end_plot, 100:].mean(axis=1),
             color="#9B59B6", lw=1.5, label="I mean gain")
axes[0].set_ylabel("Mean gain (g)")
axes[0].set_title("Population-Average Gain Modulation Over Time")
axes[0].legend(fontsize=9)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Middle: gain heatmap for E units
im1 = axes[1].imshow(g[t_start_plot:t_end_plot, :100].T, aspect="auto", cmap="viridis",
                      extent=[time_plot[0], time_plot[-1], 100, 0])
axes[1].set_ylabel("E units")
plt.colorbar(im1, ax=axes[1], label="g", shrink=0.8)

# Bottom: gain heatmap for I units
im2 = axes[2].imshow(g[t_start_plot:t_end_plot, 100:].T, aspect="auto", cmap="viridis",
                      extent=[time_plot[0], time_plot[-1], 100, 0])
axes[2].set_ylabel("I units")
axes[2].set_xlabel("Time (ms)")
plt.colorbar(im2, ax=axes[2], label="g", shrink=0.8)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gain_time_series.png"))
plt.close(fig)
print(f"  Saved: gain_time_series.png")


# ===========================================================
# Summary
# ===========================================================
print(f"\n{'='*60}")
print("  Advanced analysis plots saved:")
for f in ["pc_comparison_procrustes.png", "gain_covariance_matrix.png",
           "peak_sorted_cascade.png", "gain_time_series.png"]:
    print(f"    {f}")
print(f"{'='*60}")
