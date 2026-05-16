#!/usr/bin/env python3
"""Plot the learned neural readout weights for Session 013 (readout.0)."""

import os
import sys
import numpy as np
import torch

# ---------- CONFIG (from shared analysis_config) ----------
from analysis_config import CKPT_PATH, OUT_DIR, plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

print("Loading checkpoint...")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
sd = ckpt["state_dict"]

# Session 013 neural readout weights
W_readout = sd["readout.0.readout_neural.weight"].numpy()  # (71, 200)

print(f"Loaded readout weights: {W_readout.shape}")

# Create plot
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[3, 1], hspace=0.3, wspace=0.1)

# 1. Full Matrix Heatmap
ax_mat = fig.add_subplot(gs[0, 0])
vmax = np.max(np.abs(W_readout))
im = ax_mat.imshow(W_readout, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[0, 200, 71, 0])
ax_mat.axvline(100, color="k", linestyle="--", linewidth=2)
ax_mat.set_xlabel("SOC Units (0-99: E, 100-199: I)")
ax_mat.set_ylabel("LFADS Targets (71 units)")
ax_mat.set_title("Neural Readout Weights (Session 013)")
plt.colorbar(im, ax=ax_mat, label="Weight value")

# 2. Per-target E vs I sum of absolute weights
ax_bar_y = fig.add_subplot(gs[0, 1], sharey=ax_mat)
e_sum = np.sum(np.abs(W_readout[:, :100]), axis=1)
i_sum = np.sum(np.abs(W_readout[:, 100:]), axis=1)

y_pos = np.arange(71) + 0.5
ax_bar_y.barh(y_pos, e_sum, height=0.8, color="#2ECC71", alpha=0.8, label="Sum |w| (E)")
ax_bar_y.barh(y_pos, -i_sum, height=0.8, color="#9B59B6", alpha=0.8, label="Sum |w| (I)")
ax_bar_y.axvline(0, color="k", linewidth=1)
ax_bar_y.set_xlabel("Sum Absolute Weight")
ax_bar_y.set_title("Target Dependence")
ax_bar_y.legend(loc="upper right", fontsize=8)
ax_bar_y.invert_yaxis()

# 3. Per-SOC unit mean absolute weight (column-wise)
ax_bar_x = fig.add_subplot(gs[1, 0], sharex=ax_mat)
col_mean_abs = np.mean(np.abs(W_readout), axis=0)

ax_bar_x.bar(np.arange(100), col_mean_abs[:100], width=1.0, color="#2ECC71", alpha=0.8, label="Excitatory")
ax_bar_x.bar(np.arange(100, 200), col_mean_abs[100:], width=1.0, color="#9B59B6", alpha=0.8, label="Inhibitory")
ax_bar_x.axvline(100, color="k", linestyle="--", linewidth=2)
ax_bar_x.set_ylabel("Mean |w|")
ax_bar_x.set_xlabel("SOC Unit Index")
ax_bar_x.set_title("How much is each SOC unit used?")
ax_bar_x.legend(loc="upper right", fontsize=8)

# Stats
ax_stats = fig.add_subplot(gs[1, 1])
ax_stats.axis("off")
mean_e = np.mean(np.abs(W_readout[:, :100]))
mean_i = np.mean(np.abs(W_readout[:, 100:]))
text = (
    f"Mean |w| Excitatory: {mean_e:.4f}\n\n"
    f"Mean |w| Inhibitory: {mean_i:.4f}\n\n"
    f"Ratio (I/E): {mean_i/mean_e:.2f}x\n\n"
    f"Max |w| Excitatory: {np.max(np.abs(W_readout[:, :100])):.4f}\n\n"
    f"Max |w| Inhibitory: {np.max(np.abs(W_readout[:, 100:])):.4f}"
)
ax_stats.text(0.1, 0.5, text, fontsize=12, va="center", ha="left",
              bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

fig.suptitle("Readout Weights Analysis: Excitatory vs Inhibitory Usage", fontsize=16, y=1.02)
out_path = os.path.join(OUT_DIR, "soc_readout_weights.png")
plt.tight_layout()
fig.savefig(out_path)
print(f"Saved: {out_path}")
