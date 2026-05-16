#!/usr/bin/env python3
"""Generate trace plots using ONLY validation chops with clear temporal gaps.

Run with: conda run -n stkit-nwb python scripts/plot_validation_only.py

Shows non-adjacent validation chops with break markers between them,
making it visually unambiguous that these are held-out data segments.
"""

import os
import numpy as np
import dill

from matplotlib.patches import FancyBboxPatch

# ---------- CONFIG (from shared analysis_config) ----------
from analysis_config import (
    MERGED_PKL, OUT_DIR, DT_MS, WINDOW, OVERLAP, STRIDE, BIN_S,
    EMG_NAMES, r_squared, plt,
)



# ===========================================================
# Load data
# ===========================================================
print("Loading merged analysis...")
with open(MERGED_PKL, "rb") as f:
    merged = dill.load(f)

session_ids = sorted(merged.keys())
sid = session_ids[0]  # Session 013 for detailed plots
r = merged[sid]
T_orig = len(r["neural_pred"])

# ===========================================================
# Identify validation chop time ranges
# ===========================================================
n_chops = (T_orig - WINDOW) // STRIDE + 1
val_chop_indices = [i for i in range(n_chops) if i % 5 == 0]

# Each val chop spans [i*STRIDE, i*STRIDE + WINDOW) in bin space
val_ranges = [(i * STRIDE, i * STRIDE + WINDOW) for i in val_chop_indices]
print(f"  Session {sid}: {len(val_ranges)} validation chops out of {n_chops} total")

# Pick 6 well-spaced validation chops across the session
pick_indices = np.linspace(2, len(val_ranges) - 3, 6, dtype=int)
selected = [val_ranges[i] for i in pick_indices]
print(f"  Selected chops at bins: {[(s, e) for s, e in selected]}")
print(f"  Corresponding times: {[(s*DT_MS/1000, e*DT_MS/1000) for s, e in selected]} seconds")


# ===========================================================
# PLOT 1: EMG validation traces with break markers
# ===========================================================
muscles_to_show = [0, 2, 3, 6, 10, 13]  # BicL, Brach, DeltA, Infra, SupSp, TrLo
n_muscles = len(muscles_to_show)

fig, axes = plt.subplots(n_muscles, 1, figsize=(16, n_muscles * 2.0), sharex=True)

# Concatenate the selected chops with a small gap
gap_bins = 15  # visual gap between chops
total_plot_bins = len(selected) * WINDOW + (len(selected) - 1) * gap_bins
x_concat = np.arange(total_plot_bins) * DT_MS

for row, m_idx in enumerate(muscles_to_show):
    ax = axes[row]
    offset = 0
    for ci, (s, e) in enumerate(selected):
        t_local = np.arange(offset, offset + WINDOW) * DT_MS
        target_chunk = r["emg_target"][s:e, m_idx]
        pred_chunk = r["emg_pred"][s:e, m_idx]

        ax.plot(t_local, target_chunk, "k-", lw=1.0, alpha=0.7,
                label="Target" if ci == 0 and row == 0 else None)
        ax.plot(t_local, pred_chunk, color="#E74C3C", lw=1.0, alpha=0.9,
                label="SOC Pred" if ci == 0 and row == 0 else None)

        # Time annotation at bottom of each chop
        real_t_s = s * DT_MS / 1000
        real_t_e = e * DT_MS / 1000
        if row == n_muscles - 1:
            mid = (offset + WINDOW / 2) * DT_MS
            ax.text(mid, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.1,
                    f"{real_t_s:.1f}–{real_t_e:.1f}s",
                    ha="center", va="top", fontsize=7, color="gray")

        # Draw break markers between chops
        if ci < len(selected) - 1:
            break_x = (offset + WINDOW + gap_bins / 2) * DT_MS
            ax.axvline(break_x, color="#CCCCCC", linestyle=":", linewidth=1.5)
            # Diagonal break marks
            ylo, yhi = ax.get_ylim() if ax.get_ylim() != (0, 1) else (0, 1)

        offset += WINDOW + gap_bins

    ax.set_ylabel(EMG_NAMES[m_idx], fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Add break markers after setting ylims
for row, m_idx in enumerate(muscles_to_show):
    ax = axes[row]
    offset = 0
    for ci, (s, e) in enumerate(selected):
        if ci < len(selected) - 1:
            break_center = (offset + WINDOW + gap_bins / 2) * DT_MS
            ylo, yhi = ax.get_ylim()
            ymid = (ylo + yhi) / 2
            dy = (yhi - ylo) * 0.06
            dx = gap_bins * DT_MS * 0.3
            ax.plot([break_center - dx, break_center + dx],
                    [ymid - dy, ymid + dy], color="#999999", lw=1.5, clip_on=False)
            ax.plot([break_center - dx - 15, break_center + dx - 15],
                    [ymid - dy, ymid + dy], color="#999999", lw=1.5, clip_on=False)
        offset += WINDOW + gap_bins

axes[0].legend(fontsize=9, loc="upper right", framealpha=0.8)

# Remove x ticks (they're meaningless since we concatenated non-adjacent chops)
axes[-1].set_xticks([])
axes[-1].set_xlabel("")

# Add a custom x-axis label
fig.text(0.5, -0.01,
         "6 non-adjacent validation chops (each 1.0s, ~4s apart) — Session 013",
         ha="center", fontsize=11, color="#555555")

fig.suptitle("EMG Reconstruction — Validation Chops Only", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "emg_traces_validation_only.png"))
plt.close(fig)
print("  Saved: emg_traces_validation_only.png")


# ===========================================================
# PLOT 2: Neural validation traces with break markers
# ===========================================================
n_neur = r["neural_target"].shape[1]
neurons_to_show = np.linspace(0, n_neur - 1, 6, dtype=int)

fig, axes = plt.subplots(len(neurons_to_show), 1, figsize=(16, len(neurons_to_show) * 2.0), sharex=True)

for row, n_idx in enumerate(neurons_to_show):
    ax = axes[row]
    offset = 0
    for ci, (s, e) in enumerate(selected):
        t_local = np.arange(offset, offset + WINDOW) * DT_MS
        target_chunk = r["neural_target"][s:e, n_idx]
        pred_chunk = r["neural_pred"][s:e, n_idx]

        ax.plot(t_local, target_chunk, "k-", lw=1.0, alpha=0.7,
                label="Target (LFADS rates)" if ci == 0 and row == 0 else None)
        ax.plot(t_local, pred_chunk, color="#3498DB", lw=1.0, alpha=0.9,
                label="SOC Pred" if ci == 0 and row == 0 else None)

        if ci < len(selected) - 1:
            break_x = (offset + WINDOW + gap_bins / 2) * DT_MS
            ax.axvline(break_x, color="#CCCCCC", linestyle=":", linewidth=1.5)

        offset += WINDOW + gap_bins

    ax.set_ylabel(f"Neuron {n_idx}", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Break marks
for row, n_idx in enumerate(neurons_to_show):
    ax = axes[row]
    offset = 0
    for ci in range(len(selected)):
        if ci < len(selected) - 1:
            break_center = (offset + WINDOW + gap_bins / 2) * DT_MS
            ylo, yhi = ax.get_ylim()
            ymid = (ylo + yhi) / 2
            dy = (yhi - ylo) * 0.06
            dx = gap_bins * DT_MS * 0.3
            ax.plot([break_center - dx, break_center + dx],
                    [ymid - dy, ymid + dy], color="#999999", lw=1.5, clip_on=False)
            ax.plot([break_center - dx - 15, break_center + dx - 15],
                    [ymid - dy, ymid + dy], color="#999999", lw=1.5, clip_on=False)
        offset += WINDOW + gap_bins

axes[0].legend(fontsize=9, loc="upper right", framealpha=0.8)
axes[-1].set_xticks([])
axes[-1].set_xlabel("")
fig.text(0.5, -0.01,
         "6 non-adjacent validation chops (each 1.0s, ~4s apart) — Session 013",
         ha="center", fontsize=11, color="#555555")

fig.suptitle("Neural Rate Reconstruction — Validation Chops Only", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "neural_traces_validation_only.png"))
plt.close(fig)
print("  Saved: neural_traces_validation_only.png")


# ===========================================================
# PLOT 3: SOC heatmap — validation chops
# ===========================================================

fig, axes = plt.subplots(4, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [1, 1, 1, 1]})

# Concatenate validation chop data
rates_concat = []
neural_target_concat = []
spikes_smooth_concat = []
break_positions = []
offset = 0
for ci, (s, e) in enumerate(selected):
    rates_concat.append(r["rates"][s:e])
    neural_target_concat.append(r["neural_target"][s:e])
    spikes_smooth_concat.append(r["spikes_smooth"][s:e])
    if ci < len(selected) - 1:
        # Insert gap
        rates_concat.append(np.full((gap_bins, 200), np.nan))
        neural_target_concat.append(np.full((gap_bins, r["neural_target"].shape[1]), np.nan))
        spikes_smooth_concat.append(np.full((gap_bins, r["spikes_smooth"].shape[1]), np.nan))
        break_positions.append(offset + WINDOW + gap_bins / 2)
    offset += WINDOW + gap_bins

rates_concat = np.concatenate(rates_concat) + 20.0
neural_target_concat = np.concatenate(neural_target_concat) / BIN_S  # convert to Hz
spikes_smooth_concat = np.concatenate(spikes_smooth_concat) / BIN_S  # convert to Hz

x_extent = [0, len(rates_concat) * DT_MS]

im1 = axes[0].imshow(rates_concat[:, :100].T, aspect="auto", cmap="inferno",
                      extent=[x_extent[0], x_extent[1], 100, 0])
axes[0].set_ylabel("Excitatory (1–100)")
axes[0].set_title("SOC Population Rates — Validation Chops Only")
plt.colorbar(im1, ax=axes[0], label="Rate (Hz)", shrink=0.8)

im2 = axes[1].imshow(rates_concat[:, 100:].T, aspect="auto", cmap="inferno",
                      extent=[x_extent[0], x_extent[1], 100, 0])
axes[1].set_ylabel("Inhibitory (101–200)")
plt.colorbar(im2, ax=axes[1], label="Rate (Hz)", shrink=0.8)

im3 = axes[2].imshow(neural_target_concat.T, aspect="auto", cmap="inferno",
                      extent=[x_extent[0], x_extent[1], neural_target_concat.shape[1], 0])
axes[2].set_ylabel(f"LFADS Rates ({neural_target_concat.shape[1]})")
plt.colorbar(im3, ax=axes[2], label="Rate (Hz)", shrink=0.8)

im4 = axes[3].imshow(spikes_smooth_concat.T, aspect="auto", cmap="inferno",
                      extent=[x_extent[0], x_extent[1], spikes_smooth_concat.shape[1], 0])
axes[3].set_ylabel(f"Smoothed Spikes ({spikes_smooth_concat.shape[1]})")
plt.colorbar(im4, ax=axes[3], label="Rate (Hz)", shrink=0.8)

# Mark breaks
for ax in axes:
    for bp in break_positions:
        ax.axvline(bp * DT_MS, color="white", linewidth=2, linestyle=":")

axes[-1].set_xticks([])
axes[-1].set_xlabel("")
fig.text(0.5, -0.01,
         "6 non-adjacent validation chops (each 1.0s) — Session 013",
         ha="center", fontsize=11, color="#555555")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "soc_heatmap_validation_only.png"))
plt.close(fig)
print("  Saved: soc_heatmap_validation_only.png")


# ===========================================================
# PLOT 4: Controller + EMG — validation chops
# ===========================================================
fig, axes = plt.subplots(3, 1, figsize=(16, 7),
                          sharex=True, gridspec_kw={"height_ratios": [2, 1, 1], "hspace": 0.15})

offset = 0
for ci, (s, e) in enumerate(selected):
    t_local = np.arange(offset, offset + WINDOW) * DT_MS
    co_chunk = r["gen_inputs"][s:e]
    for d in range(co_chunk.shape[1]):
        axes[0].plot(t_local, co_chunk[:, d], linewidth=0.5, alpha=0.6)

    axes[1].plot(t_local, r["emg_target"][s:e, 0], "k-", lw=1, alpha=0.7,
                 label="Target" if ci == 0 else None)
    axes[1].plot(t_local, r["emg_pred"][s:e, 0], color="#E74C3C", lw=1, alpha=0.9,
                 label="SOC Pred" if ci == 0 else None)

    axes[2].plot(t_local, r["emg_target"][s:e, 3], "k-", lw=1, alpha=0.7)
    axes[2].plot(t_local, r["emg_pred"][s:e, 3], color="#E74C3C", lw=1, alpha=0.9)

    if ci < len(selected) - 1:
        for ax in axes:
            break_x = (offset + WINDOW + gap_bins / 2) * DT_MS
            ax.axvline(break_x, color="#CCCCCC", linestyle=":", linewidth=1.5)

    offset += WINDOW + gap_bins

axes[0].set_ylabel("Controller Output")
axes[0].set_title("All 16 Controller Outputs + EMG — Validation Chops Only")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[1].set_ylabel("BicL")
axes[1].legend(fontsize=7, loc="upper right")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

axes[2].set_ylabel("DeltA")
axes[2].spines["top"].set_visible(False)
axes[2].spines["right"].set_visible(False)

axes[-1].set_xticks([])
axes[-1].set_xlabel("")
fig.text(0.5, -0.01,
         "6 non-adjacent validation chops — Session 013",
         ha="center", fontsize=11, color="#555555")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "controller_emg_validation_only.png"))
plt.close(fig)
print("  Saved: controller_emg_validation_only.png")


# ===========================================================
# Summary: R² computed only on validation bins
# ===========================================================
print("\n=== Per-Session R² (pure validation bins only) ===")
for sid_loop in session_ids:
    r_loop = merged[sid_loop]
    T = len(r_loop["neural_pred"])
    n_chops_loop = (T - WINDOW) // STRIDE + 1
    is_train = np.zeros(T, dtype=bool)
    for i in range(n_chops_loop):
        if i % 5 != 0:
            is_train[i * STRIDE : min(i * STRIDE + WINDOW, T)] = True
    valid_test = ~is_train & ~np.isnan(r_loop["neural_pred"][:, 0])
    r2n, _ = r_squared(r_loop["neural_target"][valid_test], r_loop["neural_pred"][valid_test])
    r2e, _ = r_squared(r_loop["emg_target"][valid_test], r_loop["emg_pred"][valid_test])
    print(f"  Session {sid_loop}: {valid_test.sum()}/{T} valid bins -> "
          f"R²_neural={r2n:.3f}, R²_emg={r2e:.3f}")

print("\nDone!")
