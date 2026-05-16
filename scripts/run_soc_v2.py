#!/usr/bin/env python3
"""SOC-LFADS v2: Reduced controller capacity to force SOC dynamics to work harder.

Changes from v1:
  - ci_enc_dim: 128 → 64
  - con_dim:    64  → 32
  - co_dim:     16  → 8
  - batch_size: 256 → 1024

Rationale: If the controller is too expressive, it can override the frozen SOC
dynamics entirely, reducing the model to "controller-with-a-fancy-activation".
Shrinking the controller bottleneck forces the SOC recurrent dynamics (W, gain,
I_e) to carry more of the representational load.
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

# ---- GPU selection ----
GPU_ID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

from lfads_torch.run_model import run_model

# ---------- OPTIONS ----------
PROJECT_STR = "bsg-lfads"
DATASET_STR = "soc_gran"
RUN_TAG = datetime.now().strftime("%y%m%d") + "_soc_v2_small_ctrl"
RUN_DIR = Path("runs") / PROJECT_STR / DATASET_STR / RUN_TAG
OVERWRITE = True  # Safe — only overwrites failed runs with same tag
# ------------------------------

if RUN_DIR.exists() and OVERWRITE:
    shutil.rmtree(RUN_DIR)
elif RUN_DIR.exists():
    print(f"ERROR: Run directory already exists: {RUN_DIR}")
    print("Set OVERWRITE = True or delete manually.")
    sys.exit(1)

RUN_DIR.mkdir(parents=True)
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
os.chdir(RUN_DIR)

print(f"{'='*60}")
print(f"  SOC-LFADS v2: Reduced Controller")
print(f"  ci_enc_dim=64, con_dim=32, co_dim=8, batch_size=64")
print(f"  Run dir: {RUN_DIR.resolve()}")
print(f"  GPU: {GPU_ID}")
print(f"{'='*60}")

run_model(
    overrides={
        "datamodule": DATASET_STR,
        "model": DATASET_STR,
        # --- Reduced controller ---
        "model.ci_enc_dim": 64,
        "model.con_dim": 32,
        "model.co_dim": 8,
        # --- Larger batch ---
        "datamodule.batch_size": 64,
        # --- Don't try to reload best ckpt (we skip posterior sampling) ---
        "posterior_sampling.use_best_ckpt": False,
    },
    config_path="../configs/soc_gran.yaml",
    do_posterior_sample=False,
)
