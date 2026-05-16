#!/usr/bin/env python3
"""Run a single SOC-LFADS training run (no PBT).

Usage:
    conda run -n lfads-torch-cuda12 python scripts/run_soc.py

This is a standalone training run that bypasses Ray Tune entirely.
It directly instantiates the model/datamodule from the Hydra configs
and trains using PyTorch Lightning.

For PBT hyperparameter search, adapt run_pbt.py with SOC-specific params.
"""

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ---- Ensure bsg-lfads code takes priority over other lfads-torch installs ----
BSG_LFADS_ROOT = str(Path(__file__).resolve().parent.parent)
if BSG_LFADS_ROOT not in sys.path:
    sys.path.insert(0, BSG_LFADS_ROOT)

import torch

# ---- Limit CPU usage on shared system (48 cores, leave ~8 for others) ----
N_CPUS = 40
torch.set_num_threads(N_CPUS)
os.environ["OMP_NUM_THREADS"] = str(N_CPUS)
os.environ["MKL_NUM_THREADS"] = str(N_CPUS)

# ---- GPU selection (single GPU for non-PBT run) ----
GPU_ID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

from lfads_torch.run_model import run_model

# ---------- OPTIONS ----------
PROJECT_STR = "bsg-lfads"
DATASET_STR = "soc_gran"
RUN_TAG = datetime.now().strftime("%y%m%d") + "_soc_mvp"
RUN_DIR = Path("runs") / PROJECT_STR / DATASET_STR / RUN_TAG
OVERWRITE = True
# ------------------------------

# Overwrite the directory if necessary
if RUN_DIR.exists() and OVERWRITE:
    shutil.rmtree(RUN_DIR)
RUN_DIR.mkdir(parents=True)

# Copy this script into the run directory for reproducibility
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)

# Switch to the RUN_DIR and train the model
os.chdir(RUN_DIR)

print(f"{'='*60}")
print(f"  SOC-LFADS Training Run")
print(f"  Config: soc_gran.yaml")
print(f"  Run dir: {RUN_DIR.resolve()}")
print(f"  GPU: {GPU_ID}")
print(f"  CPU threads: {N_CPUS}")
print(f"{'='*60}")

run_model(
    overrides={
        "datamodule": DATASET_STR,
        "model": DATASET_STR,
    },
    config_path="../configs/soc_gran.yaml",
    do_posterior_sample=False,  # Skip posterior sampling for MVP
)
