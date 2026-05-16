"""Shared configuration for SOC-LFADS analysis scripts.

All plotting and analysis scripts import from here so that switching
between model runs only requires changing RUN_TAG.
"""

import os
from pathlib import Path

# ============================================================
# >>>  CHANGE THIS TO SWITCH RUNS  <<<
# ============================================================
RUN_TAG = "260419_soc_mvp"
# RUN_TAG = "260419_soc_v2_small_ctrl"
# ============================================================

# --- Derived paths ---
RUNS_ROOT = Path("/home/cbwash2/bsg-lfads/runs/bsg-lfads/soc_gran")
RUN_DIR = RUNS_ROOT / RUN_TAG
OUT_DIR = Path("/home/cbwash2/bsg-lfads/analysis_plots") / RUN_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Model outputs
SOC_H5 = RUN_DIR / "soc_output_all_chops.h5"
MERGED_PKL = RUN_DIR / "soc_merged_analysis.pkl"

# Checkpoint (auto-detect best or last)
CKPT_DIR = RUN_DIR / "lightning_checkpoints"
def _find_best_ckpt():
    """Find the best (non-last) checkpoint, fallback to last.ckpt."""
    ckpts = sorted(CKPT_DIR.glob("*.ckpt"))
    for c in ckpts:
        if c.name != "last.ckpt":
            return c
    return CKPT_DIR / "last.ckpt"

CKPT_PATH = _find_best_ckpt()

# Data
PKL_DIR = "/snel/share/share/tmp/scratch/cbwash2/auyong_nwb/binsize_10ms_pcr_high_reg_ALL_cw"
NEURAL_PATTERN = "/home/cbwash2/bsg-lfads/datasets/soc_gran/neural/lfads_torch_readin*_neural.h5"
EMG_PATTERN = "/home/cbwash2/bsg-lfads/datasets/soc_gran/emg/lfads_torch_readin*_emg.h5"
W_PATH = "/home/cbwash2/bsg-lfads/weights/W_soc_200.pt"

# Timing
DT_MS = 10
WINDOW = 100
OVERLAP = 20
STRIDE = WINDOW - OVERLAP
BIN_S = DT_MS / 1000  # 0.01

# Labels
EMG_NAMES = [
    "BicL", "BicS", "Brach", "DeltA", "DeltM", "DeltP",
    "Infra", "LatD", "PecM", "SubSc", "SupSp", "TMaj", "TLaLo", "TrLo"
]

# Per-run model architecture (needed for posterior sampling)
RUN_CONFIGS = {
    "260419_soc_mvp": {
        "ic_enc_dim": 128, "ci_enc_dim": 128,
        "con_dim": 64, "co_dim": 16,
    },
    "260419_soc_v2_small_ctrl": {
        "ic_enc_dim": 128, "ci_enc_dim": 64,
        "con_dim": 32, "co_dim": 8,
    },
}

def get_model_config():
    """Return the architecture config dict for the active RUN_TAG."""
    return RUN_CONFIGS[RUN_TAG]

# Plot style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

def r_squared(y_true, y_pred):
    """R² per channel, then mean."""
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    return r2.mean(), r2

# Summary
print(f"[analysis_config] RUN_TAG = {RUN_TAG}")
print(f"  RUN_DIR  = {RUN_DIR}")
print(f"  OUT_DIR  = {OUT_DIR}")
print(f"  CKPT     = {CKPT_PATH.name}")
print(f"  SOC_H5   = {'EXISTS' if SOC_H5.exists() else 'MISSING'}")
print(f"  MERGED   = {'EXISTS' if MERGED_PKL.exists() else 'MISSING'}")
