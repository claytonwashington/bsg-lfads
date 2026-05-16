# AGENTS.md — bsg-lfads

Instructions and context for AI coding agents (Codex, Antigravity, etc.) working in this repository.

---

## Project Overview

This repo extends [`lfads-torch`](https://github.com/arsedler9/lfads-torch) with a biophysically-constrained **SOC-LFADS** model that replaces the GRU generator with a Stability-Optimized Circuit (SOC) E/I network. It models cat granule cell neural population dynamics and EMG jointly across 6 recording sessions.

See `CLAUDE.md` for full architecture details, data flow, and key file locations.

---

## Environments

**Do not run code without specifying the correct conda environment.**

```bash
# Model training, inference, and most Python work
conda run -n lfads-torch-cuda12 python <script>

# Data preparation from NWB pickles (requires snel-toolkit)
conda run -n stkit-nwb python scripts/prepare_soc_data.py
```

---

## Code Conventions

### The SOC extension lives in `lfads_torch/`

New files added to the upstream `lfads-torch` codebase:
- `lfads_torch/modules/recurrent.py` — `SOCCell` is appended at the bottom; do not modify the upstream GRU classes above it
- `lfads_torch/modules/soc_decoder.py` — SOCDecoder
- `lfads_torch/modules/readout.py` — DualReadout, MultisessionDualReadout
- `lfads_torch/soc_model.py` — LFADS_SOC LightningModule
- `lfads_torch/soc_datamodules.py` — SOCDataModule

**Do not modify** other `lfads_torch/` files (encoders, upstream decoder, etc.) unless intentional.

### Comments
- Comments should explain *why*, not *what*. Avoid over-commenting obvious lines.
- No `# Step N:` style banners. No inline comments on self-explanatory code.
- Scientific/mathematical comments (e.g., explaining the SOC dynamics equation) are encouraged.

### Typing
- Use `from __future__ import annotations` at the top of new files for Python 3.9 compatibility.
- Avoid `tuple[X, Y]` or `dict[str, T]` in type hints without the future import.

---

## Key Invariants — Do Not Break

1. **`W` is frozen**: The SOC weight matrix is a `buffer`, not a `Parameter`. Never change `requires_grad=True` on it.
2. **Dale's law**: Excitatory columns of W are non-negative, inhibitory columns are non-positive. Scripts that generate W (`scripts/save_soc_weights.py`) enforce this.
3. **Readin weights are frozen**: `MultisessionReadin` weights are initialized from PCR matrices and frozen. Do not set them as trainable.
4. **EMG readout uses excitatory units only**: `DualReadout` indexes `rates[:, :, :N//2]` for EMG. This is intentional.

---

## Running Training

```bash
# Standard single-GPU run (from repo root)
conda run -n lfads-torch-cuda12 python scripts/run_soc.py

# Or directly via Hydra
conda run -n lfads-torch-cuda12 python -m lfads_torch.run_model \
    --config-path ../configs --config-name soc_gran
```

Output lands in `runs/bsg-lfads/soc_gran/<run_name>/`. This directory is **gitignored** — sync to NAS via:
```bash
rsync -avh --inplace --no-perms --no-owner --no-group \
    runs/ /home/cbwash2/nay-storage/soc/runs/
```

---

## Data Pipeline

```
NWBDataset pickles
  → conda run -n stkit-nwb python scripts/prepare_soc_data.py
  → datasets/soc_gran/{neural,emg}/*.h5   (gitignored)
  → SOCDataModule (lfads-torch-cuda12)
  → LFADS_SOC training
```

The `datasets/` directory is gitignored. Regenerate from the NWBDataset pickles at:
`/snel/share/share/tmp/scratch/cbwash2/auyong_nwb/binsize_10ms_pcr_high_reg_ALL_cw/`

---

## Analysis Workflow

After training:
1. `scripts/soc_posterior_sampling.py` — run posterior sampling, save `soc_output_all_chops.h5`
2. `scripts/merge_soc_outputs.py` — merge chops, compute R², save `soc_merged_analysis.pkl`
3. `scripts/plot_validation_only.py` — heatmaps, traces, R² bars for validation bins
4. `scripts/plot_readout_weights.py` — E/I readout weight analysis

Edit `scripts/analysis_config.py` to point to the run you want to analyze.

---

## Gitignored Directories

| Directory | Why gitignored | Where it lives |
|---|---|---|
| `runs/` | Large checkpoints + outputs | `/home/cbwash2/nay-storage/soc/runs/` |
| `datasets/` | Regenerable from NWB pickles | Regenerate with `prepare_soc_data.py` |

---

## Open Research Directions

See `docs/future_plans.md` for details. Key items:

1. **Readout regularization** — fix linear regime collapse via L2 on readout, LayerNorm, or sparse identity readout
2. **Controller regularization** — enable AR prior on controller outputs (`kl_co_scale`)
3. **iLQR-VAE** — replace LFADS encoder/controller with structured inference over (I_e, g)
