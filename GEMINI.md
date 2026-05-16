# GEMINI.md — bsg-lfads

This file gives Gemini (and similar AI assistants) the context needed to work in this repo without re-reading everything from scratch.

---

## What This Repo Is

`bsg-lfads` is a fork of [`lfads-torch`](https://github.com/arsedler9/lfads-torch), extended to replace the standard GRU generator with a biophysically-constrained **Stability-Optimized Circuit (SOC)** E/I recurrent network. The SOC dynamics follow Hennequin et al. (Nature 2014) and Stroud et al. (Nature Neuroscience 2018).

The goal is to model both neural population dynamics and EMG jointly, using LFADS's encoder/controller infrastructure but with the generator replaced by a fixed-connectivity E/I circuit.

---

## Repository Layout

```text
bsg-lfads/
├── lfads_torch/             # Upstream lfads-torch (largely unmodified)
│   ├── modules/
│   │   ├── recurrent.py     # SOCCell appended here
│   │   ├── soc_decoder.py   # NEW: Controller GRU + SOCCell temporal loop
│   │   └── readout.py       # NEW: DualReadout + MultisessionDualReadout
│   ├── soc_model.py         # NEW: LFADS_SOC LightningModule
│   └── soc_datamodules.py   # NEW: paired neural+EMG H5 data loading
├── scripts/
│   ├── prepare_soc_data.py  # PCR alignment + chop → H5 files
│   ├── save_soc_weights.py  # Generate SOC W matrix (.pt)
│   ├── run_soc.py           # Training entry point
│   ├── soc_posterior_sampling.py
│   ├── merge_soc_outputs.py
│   ├── analysis_config.py   # Shared paths/constants — edit this to switch runs
│   └── ...
├── configs/
│   ├── soc_gran.yaml        # Top-level Hydra config
│   ├── model/soc_gran.yaml  # Model hyperparameters
│   └── datamodule/soc_gran.yaml
├── weights/
│   └── W_soc_200.pt         # Pre-computed SOC connectivity matrix (frozen)
├── datasets/
│   └── soc_gran/            # Chopped H5 files (gitignored)
│       ├── neural/          # lfads_torch_readin{sid}_neural.h5
│       └── emg/             # lfads_torch_readin{sid}_emg.h5
├── runs/                    # Training artifacts (gitignored, synced to NAS)
│   └── bsg-lfads/soc_gran/
│       ├── 260419_soc_mvp/           # Run 1: co_dim=16
│       └── 260419_soc_v2_small_ctrl/ # Run 2: co_dim=4
├── analysis_plots/          # Output figures
├── auto-nn/                 # Submodule: SOC generator + standalone test
│   └── auto_nn/test.py      # Standalone SOC replication (Stroud et al.)
└── docs/
    ├── soc_mvp_walkthrough.md
    ├── soc_mvp_implementation_plan.md
    ├── future_plans.md
    ├── run_log.md
    └── snel_toolkit_reference.md
```

---

## Conda Environments

| Task | Environment |
|---|---|
| Model training / inference | `lfads-torch-cuda12` (torch 1.13.1, PTL 1.6.0, Python 3.9) |
| Data preparation (NWB → H5) | `stkit-nwb` |

Always prefix commands with `conda run -n <env>` unless already activated.

---

## SOC-LFADS Architecture

Data flow for one session, one trial:

```text
spikes (B, T, D_raw)
  → Readin[s] (PCR, frozen): (B, T, 20)
  → IC Encoder (BiGRU):       ic_mean (B, 64)
  → CI Encoder (BiGRU):       ci (B, T, 256)

ic_mean → Linear → v₀ (B, 200)   # initial SOC membrane voltages

For t = 0..T-1:
  [ci_t, r_{t-1}] → Controller GRU → co (B, 16)
  co → Linear → I_e (B, 200)      # tonic input drive
  co → Linear + softplus → g (B, 200)  # per-neuron gain

  SOCCell:
    r = f(v, g)    # piecewise asymmetric tanh (r0=20 Hz, rmax=100 Hz)
                   # r0·tanh(g·v/r0)           if v < 0
                   # (rmax-r0)·tanh(g·v/(rmax-r0))  if v ≥ 0
    dv = W @ r + I_e   # W is frozen (Dale's law: E cols positive, I cols negative)
    v_next = v + (dt/τ) * (-v + dv)

rates (B, T, 200) → DualReadout[s]:
  all 200 → Linear → neural_pred (B, T, D_raw_s)
  first 100 (excitatory) → Linear + exp() → emg_pred (B, T, 14)

Loss = MSE(neural) + MSE(emg) + KL(ic) + KL(co) + L2
```

**Key constraint**: `W` is loaded from `weights/W_soc_200.pt` and registered as a non-trainable `buffer`. Never pass it as a `nn.Parameter`.

---

## Key Design Decisions

1. **Reconstruction targets** are smooth LFADS outputs (`lfads_rates`, `deEMG_mean`), not raw spikes/EMG. MSE loss, not Poisson NLL.
2. **PCR alignment** computes readin matrices from cycle-averaged (trial-aligned) PSTHs, not raw continuous data. See `auyong_pcr_alignment_v4.py` for the reference implementation.
3. **Multisession**: per-session `MultisessionReadin` (frozen PCR weights) + per-session `MultisessionDualReadout`. SOC population is shared across sessions.
4. **N=200** (100 excitatory, 100 inhibitory). EMG readout uses excitatory units only.
5. **Chops**: window=100 bins (1 s at 10 ms/bin), overlap=20 bins (200 ms), stride=80, every 5th chop held out for validation.

---

## Known Finding: Linear Regime Collapse

The trained SOC operates in a narrow band around r0=20 Hz (~16–25 Hz). The dense readout compensates with large weights (sum |w| per target ≈ 10–30×), effectively making the system linear. See `docs/future_plans.md` §1 for proposed fixes (readout L2, LayerNorm, sparse identity readout).

---

## Important File Locations (External)

These are on shared NAS/cluster storage, not in this repo:

| What | Path |
|---|---|
| NWBDataset pickles | `/snel/share/share/tmp/scratch/cbwash2/auyong_nwb/binsize_10ms_pcr_high_reg_ALL_cw/` |
| PCR alignment matrices | `/snel/share/share/derived/auyong/nwb_lfads/runs/binsize_10ms_pcr_high_reg_ALL_cw/gran/alignment_matrices/` |
| Training runs (NAS backup) | `/home/cbwash2/nay-storage/soc/runs/` |
| PCR alignment script | `/home/cbwash2/emg-dynamics/emg_paper/nwb_conversion/auyong_pcr_alignment_v4.py` |
| Post-analysis script | `/home/cbwash2/emg-dynamics/emg_paper/nwb_conversion/auyong_post_path_length_notebook_v3.py` |

---

## Gitignore Notes

- `runs/` — gitignored; large checkpoints + analysis outputs synced to `/home/cbwash2/nay-storage/soc/runs/`
- `datasets/` — gitignored; regenerate with `scripts/prepare_soc_data.py`

---

## Where to Read More

- Full architectural detail: `docs/soc_mvp_walkthrough.md`
- Future directions and known issues: `docs/future_plans.md`
- snel-toolkit API reference: `docs/snel_toolkit_reference.md`
- Run history and notes: `docs/run_log.md`
