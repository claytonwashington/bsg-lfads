#!/usr/bin/env python3
"""Run PBT hyperparameter search for SOC-LFADS.

Usage:
    conda run -n lfads-torch-cuda12 python scripts/run_soc_pbt.py

Uses all 8 GPUs (1 per trial), 8 parallel trials.
Tunes: lr_init, dropout_rate, kl_co_scale, kl_ic_scale, soc_tau.

CPU budget: 5 per trial × 8 trials = 40 CPUs, leaving 8 for other users.
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

# ---- Limit CPU usage on shared system ----
os.environ["OMP_NUM_THREADS"] = "5"
os.environ["MKL_NUM_THREADS"] = "5"

from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.basic_variant import BasicVariantGenerator

from lfads_torch.extensions.tune import (
    BinaryTournamentPBT,
    HyperParam,
    ImprovementRatioStopper,
)
from lfads_torch.run_model import run_model

# ---------- OPTIONS ----------
PROJECT_STR = "bsg-lfads"
DATASET_STR = "soc_gran"
RUN_TAG = datetime.now().strftime("%y%m%d") + "_soc_pbt"
RUN_DIR = Path("/snel/share/runs") / PROJECT_STR / DATASET_STR / RUN_TAG

# SOC-specific + standard LFADS hyperparameter search space
HYPERPARAM_SPACE = {
    # --- Standard LFADS hyperparams ---
    "model.lr_init": HyperParam(
        1e-5, 5e-3, explore_wt=0.3, enforce_limits=True, init=4e-3
    ),
    "model.dropout_rate": HyperParam(
        0.0, 0.6, explore_wt=0.3, enforce_limits=True, sample_fn="uniform"
    ),
    "model.kl_co_scale": HyperParam(1e-6, 1e-4, explore_wt=0.8),
    "model.kl_ic_scale": HyperParam(1e-6, 1e-3, explore_wt=0.8),
    # --- SOC-specific hyperparams ---
    # Membrane time constant (biophysically plausible: 10–200 ms)
    "model.soc_tau": HyperParam(
        10.0, 200.0, explore_wt=0.3, enforce_limits=True, init=50.0,
        sample_fn="uniform",
    ),
    # Baseline firing rate (biophysically plausible: 5–30 Hz)
    "model.soc_r0": HyperParam(
        5.0, 30.0, explore_wt=0.3, enforce_limits=True, init=20.0,
        sample_fn="uniform",
    ),
    # Max firing rate (biophysically plausible: 50–200 Hz)
    "model.soc_rmax": HyperParam(
        50.0, 200.0, explore_wt=0.3, enforce_limits=True, init=100.0,
        sample_fn="uniform",
    ),
}

# --- Resource allocation ---
# 8 GPUs available, 1 per trial, 5 CPUs per trial (40 total, leaving 8 for others)
NUM_TRIALS = 8
CPUS_PER_TRIAL = 5
GPUS_PER_TRIAL = 1.0
# ------------------------------


def clip_config_rates(config):
    """Keep dropout rate in-bounds."""
    return {k: min(v, 0.99) if "_rate" in k else v for k, v in config.items()}


init_space = {
    name: tune.sample_from(hp.init) for name, hp in HYPERPARAM_SPACE.items()
}

mandatory_overrides = {
    "datamodule": DATASET_STR,
    "model": DATASET_STR,
    "logger.wandb_logger.project": PROJECT_STR,
    "logger.wandb_logger.tags.1": DATASET_STR,
    "logger.wandb_logger.tags.2": RUN_TAG,
}

RUN_DIR.mkdir(parents=True, exist_ok=True)
# Copy this script into the run directory for reproducibility
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)

metric = "valid/recon_smth"
perturbation_interval = 25
burn_in_period = 80 + 25

print(f"{'='*60}")
print(f"  SOC-LFADS PBT Training")
print(f"  Trials: {NUM_TRIALS}")
print(f"  GPUs per trial: {GPUS_PER_TRIAL}")
print(f"  CPUs per trial: {CPUS_PER_TRIAL}")
print(f"  Run dir: {RUN_DIR}")
print(f"{'='*60}")

analysis = tune.run(
    tune.with_parameters(
        run_model,
        config_path="../configs/soc_pbt.yaml",
        do_posterior_sample=False,
    ),
    metric=metric,
    mode="min",
    name=RUN_DIR.name,
    stop=ImprovementRatioStopper(
        num_trials=NUM_TRIALS,
        perturbation_interval=perturbation_interval,
        burn_in_period=burn_in_period,
        metric=metric,
        patience=4,
        min_improvement_ratio=5e-4,
    ),
    config={**mandatory_overrides, **init_space},
    resources_per_trial=dict(cpu=CPUS_PER_TRIAL, gpu=GPUS_PER_TRIAL),
    num_samples=NUM_TRIALS,
    local_dir=RUN_DIR.parent,
    search_alg=BasicVariantGenerator(random_state=0),
    scheduler=BinaryTournamentPBT(
        perturbation_interval=perturbation_interval,
        burn_in_period=burn_in_period,
        hyperparam_mutations=HYPERPARAM_SPACE,
    ),
    keep_checkpoints_num=1,
    verbose=1,
    progress_reporter=CLIReporter(
        metric_columns=[metric, "cur_epoch"],
        sort_by_metric=True,
    ),
    trial_dirname_creator=lambda trial: str(trial),
)

# Copy the best model to a new folder
best_model_dir = RUN_DIR / "best_model"
shutil.copytree(analysis.best_logdir, best_model_dir)
# Switch working directory and run posterior sampling on best model
os.chdir(best_model_dir)
best_ckpt_dir = best_model_dir / Path(analysis.best_checkpoint._local_path).name
run_model(
    overrides=mandatory_overrides,
    checkpoint_dir=best_ckpt_dir,
    config_path="../configs/soc_pbt.yaml",
    do_train=False,
)
