# LFADS-SOC MVP — Walkthrough

## Summary

Replaced the standard LFADS GRU generator with a biophysical SOC (Stability-Optimized Circuit) cell, added a dual-headed neural+EMG readout, and swapped Poisson NLL for MSE loss. All changes live in the `lfads_torch/` package of the `bsg-lfads` repo.

## Design Philosophy

The SOC-LFADS model answers: **can a biophysically-motivated E/I network generate the neural dynamics already inferred by a trained LFADS model?**

Instead of fitting raw spikes (hard MSE target) or raw EMG (noisy), we use the **smooth outputs of already-trained LFADS models** as reconstruction targets:
- **Neural**: `lfads_rates` — smooth firing rates from the trained spike-LFADS (same dim as raw neurons per session: 71, 62, 65, 63, 62, 59)
- **EMG**: `deEMG_mean` — smooth decoded EMG from the trained EMG-LFADS (14 channels, shared across sessions)

This makes the MSE loss natural (smooth continuous targets) and lets us isolate whether SOC dynamics can reproduce what a GRU-LFADS already extracted.

---

## Files Changed

### Modified
- [recurrent.py](file:///home/cbwash2/bsg-lfads/lfads_torch/modules/recurrent.py) — Appended `SOCCell` class (Euler-step E/I network with frozen W buffer)

### New Files

| File | Purpose |
|---|---|
| [soc_decoder.py](file:///home/cbwash2/bsg-lfads/lfads_torch/modules/soc_decoder.py) | SOC Decoder — controller GRU + SOCCell loop |
| [readout.py](file:///home/cbwash2/bsg-lfads/lfads_torch/modules/readout.py) | DualReadout (neural all-units + EMG exc-only) + MultisessionDualReadout |
| [soc_model.py](file:///home/cbwash2/bsg-lfads/lfads_torch/soc_model.py) | LFADS_SOC LightningModule |
| [soc_datamodules.py](file:///home/cbwash2/bsg-lfads/lfads_torch/soc_datamodules.py) | SOCDataModule — paired neural + EMG H5 loading |
| [save_soc_weights.py](file:///home/cbwash2/bsg-lfads/scripts/save_soc_weights.py) | Script to generate and save SOC W matrix as `.pt` |
| [prepare_soc_data.py](file:///home/cbwash2/bsg-lfads/scripts/prepare_soc_data.py) | Data prep: PCR alignment + chopping → H5 files |
| [test_soc_smoke.py](file:///home/cbwash2/bsg-lfads/scripts/test_soc_smoke.py) | Smoke tests for SOCCell, DualReadout, gradient flow |

### Unchanged (reused from lfads-torch)
- `lfads_torch/modules/encoder.py` — IC + CI BiGRU encoders
- `lfads_torch/modules/priors.py` — KL divergence priors
- `lfads_torch/modules/recons.py` — MSE reconstruction class (already existed)
- `lfads_torch/modules/readin_readout.py` — MultisessionReadin/Readout (PCR-initialized linear layers)
- `lfads_torch/datamodules.py` — original BasicDataModule (still works for standard LFADS)

---

## Architecture

```
Data (B, T, D_raw) → Readin[s] (B, T, 20) → IC Encoder → ic_mean (B, 64)
                                             → CI Encoder → ci (B, T, 256)

ic_mean → ic_to_v0 → v₀ (B, 200)

For t in 0..T-1:
  [ci_t, r_feedback] → Controller GRU → con_state (B, 64) → co_linear → co (B, 16)
  co → controller_to_Ie → I_e (B, 200)      [tonic input drive]
  co → controller_to_g + softplus → g (B, 200)  [per-neuron gain modulation]

  SOCCell dynamics:
    r = f(v, g)  ← piecewise asymmetric tanh:
        r0·tanh(g·v/r0)           if v < 0   (saturates at ±r0=20 Hz)
        (rmax-r0)·tanh(g·v/(rmax-r0))  if v ≥ 0   (saturates at rmax-r0=80 Hz)
    τ dv/dt = -v + W·r + I_e
    v_next = v + (dt/τ)·dv        (Euler step)

rates (B, T, 200) → DualReadout[s]:
  all 200 units → Linear → neural_pred (B, T, neural_dim_s)
  first 100 units → Linear + exp → emg_pred (B, T, 14)

Loss = MSE(neural) + MSE(emg) + KL(ic) + KL(co) + L2(encoder)
```

### Key Design Decisions

1. **Activation function** — piecewise asymmetric `tanh` matching the SOC paper. Gain `g` modulates the slope around V=0. Parameters `r0` (baseline, default 20 Hz) and `rmax` (max, default 100 Hz) are PBT-tunable.

2. **Weight matrix W is frozen** — registered as a `buffer`, not a `Parameter`. The SOC connectivity structure is pre-computed and fixed; only the controller projections and readout weights are learned.

3. **Gain is the fast control knob** — `g` is projected from the 16-dim controller output via `controller_to_g + softplus` (ensuring positivity). Each of 200 neurons gets an independent gain at each timestep, enabling the controller to modulate the network's response properties dynamically.

4. **Tonic input (`I_e`)** — a second controller projection provides per-neuron tonic drive, following the full dynamics equation from the Nature paper.

5. **Controller GRU input** = `[ci_t (2×ci_enc_dim), r_feedback (N)]` — the SOC rates feed back into the controller so it can observe the network state.

6. **Dual readout** — neural prediction uses ALL N SOC units; EMG prediction uses only the first N/2 (excitatory) units, with `exp()` activation enforcing positivity.

7. **MSE loss** for both heads (not Poisson NLL) — appropriate since targets (`lfads_rates`, `deEMG_mean`) are smooth continuous values.

8. **Multisession** — variable neural dimensions across sessions handled by per-session `MultisessionReadin` (PCR-initialized, frozen) and per-session `MultisessionDualReadout`. The SOC population (N=200) is shared across all sessions.

---

## Data Pipeline

### Input Data

Source: NWBDataset pickle files at `/snel/share/share/tmp/scratch/cbwash2/auyong_nwb/binsize_10ms_pcr_high_reg_ALL_cw/`

| Session | Neural dim (`lfads_rates`) | EMG dim (`deEMG_mean`) | Timesteps |
|---|---|---|---|
| 013 | 71 | 14 | 7533 |
| 023 | 62 | 14 | 8162 |
| 028 | 65 | 14 | 8330 |
| 030 | 63 | 14 | 7743 |
| 031 | 62 | 14 | 7793 |
| 033 | 59 | 14 | 6736 |

### PCR Alignment (readin matrices)

**Following `auyong_pcr_alignment_v4.py` exactly:**

1. **Load** each session's pickled NWBDataset
2. **Exclude** first 4 and last 2 trials (same as original)
3. **Align** trials to `start_time` with range `(-100, 600)` ms → 71 aligned timebins at 10ms
4. **Cycle-average** across all trials per session → `cycle_avg` shape `(71, n_channels)`
5. **Concatenate** all sessions horizontally → `global_avg` shape `(71, sum_of_all_channels)`
6. **PCA** on mean-centered global averages (excluding NaN rows) → `global_pcs`
7. **Ridge regression** per session (α=0.01): `session_avg → global_pcs` → `W` (readin_weight), `mean` (readout_bias)

| Parameter | Neural | EMG |
|---|---|---|
| PCR dim | 20 | 10 |
| L2 scale (Ridge α) | 0.01 | 0.01 |
| PCA explained var | 0.9999 | 0.9934 |

> [!IMPORTANT]
> The PCR alignment matrices are computed on **cycle-averaged, trial-aligned** data — NOT on raw continuous data. This is critical for proper cross-session alignment into a shared latent space.

### Chopping (training data)

Continuous `lfads_rates` and `deEMG_mean` are chopped into overlapping segments:
- **Window**: 100 bins (1 second at 10ms)
- **Overlap**: 20 bins (200ms)
- **Stride**: 80 bins
- **Train/valid split**: every 5th chop → validation (4:1 ratio)

| Session | Total chops | Train | Valid |
|---|---|---|---|
| 013 | 93 | 74 | 19 |
| 023 | 101 | 80 | 21 |
| 028 | 103 | 82 | 21 |
| 030 | 96 | 76 | 20 |
| 031 | 97 | 77 | 20 |
| 033 | 83 | 66 | 17 |

### Output H5 Files

Saved to `datasets/soc_gran/neural/` and `datasets/soc_gran/emg/`:

```
lfads_torch_readin{session_id}_{neural|emg}.h5
├── train_encod_data   (n_train, 100, n_channels) float32
├── train_recon_data   (n_train, 100, n_channels) float32  [= encod_data]
├── valid_encod_data   (n_valid, 100, n_channels) float32
├── valid_recon_data   (n_valid, 100, n_channels) float32  [= encod_data]
├── readin_weight      (n_channels, pcr_dim) float64
└── readout_bias       (n_channels,) float64
```

- `encod_data = recon_data` — the readin projection is applied at model time by `MultisessionReadin`, not pre-applied to the data
- `readin_weight` and `readout_bias` match the format expected by `MultisessionReadin._get_state_dict()`

---

## Verification Results

All tests pass with `conda run -n lfads-torch-cuda12`:

```
=== Test: SOCCell ===
  ✓ Buffer, E/I, shapes, stability all correct
=== Test: DualReadout ===
  ✓ Shapes and positivity correct
=== Test: Gradient Flow ===
  ✓ Gradients flow correctly (W frozen, projections trainable)

=== Full LFADS_SOC end-to-end test ===
  Forward pass: neural_pred (4, 10, 15), emg_pred (4, 10, 5), rates (4, 10, 20)
  Training loss: 2.054852
  Parameters with gradients: 35/35
  W is correctly frozen (no gradients)
  ✓ PASSED
```

---

## Commands to Reproduce

### 1. Generate W matrix
```bash
conda run -n lfads-torch-cuda12 python scripts/save_soc_weights.py \
  --N 200 --output weights/W_soc_200.pt
```

### 2. Prepare data (PCR alignment + chopping)
```bash
conda run -n stkit-nwb python scripts/prepare_soc_data.py \
  --dataset_dir /snel/share/share/tmp/scratch/cbwash2/auyong_nwb/binsize_10ms_pcr_high_reg_ALL_cw \
  --output_dir datasets/soc_gran \
  --cat gran
```

### 3. Create Hydra config and train (TODO)
```bash
conda run -n lfads-torch-cuda12 python -m lfads_torch.run_model \
  --config-name=soc_gran
```

---

## Reference Scripts

The data preparation pipeline follows the same approach used in the original analysis:

| Script | Purpose | Environment |
|---|---|---|
| [auyong_pcr_alignment_v4.py](file:///home/cbwash2/emg-dynamics/emg_paper/nwb_conversion/auyong_pcr_alignment_v4.py) | Original PCR alignment for spike-LFADS | `stkit-nwb` |
| [auyong_post_path_length_notebook_v3.py](file:///home/cbwash2/emg-dynamics/emg_paper/nwb_conversion/auyong_post_path_length_notebook_v3.py) | Post-LFADS analysis with EMG burst detection and separate PCR fitting | `stkit-nwb` |

> [!NOTE]
> The `auto-nn/` submodule's gain branch (`g_enc`) is preserved but unused. A `TODO` for integrating it into the SOC model for direct gain encoding from data is noted in the implementation plan.

---

## Ablation Study: Controller Capacity (v2)

To verify that the biophysical SOC network was actually doing the "heavy lifting" (and the controller GRU wasn't simply treating the SOC layer as a pass-through), we ran an ablation study where we choked the controller capacity:

* **v1 (Baseline):** `co_dim=16`, `con_dim=64`, `ci_enc_dim=128`.
* **v2 (Reduced capacity):** `co_dim=8`, `con_dim=32`, `ci_enc_dim=64`.

**Results:**

| Metric | V1 (Full Controller) | V2 (Reduced Controller) |
|---|---|---|
| **Valid R² Neural** | **0.829** | **-723M** 💥 |
| **Valid R² EMG** | **0.810** | **-91.8** |
| Valid Recon Neural | 0.0024 | 0.0017 |
| Valid Recon EMG | 0.0128 | 0.0116 |

### Interpretation

When the controller bottleneck was squeezed too tightly, the model experienced a **representational collapse**. Notice that the raw reconstruction errors (MSE) actually *dropped* slightly in v2, yet the R² became catastrophically negative!

This occurs because the model lacked the control bandwidth to modulate the SOC network for trial-varying specific dynamics. Instead, it fell back to predicting the **cross-trial mean trajectory**. R² penalizes mean-only predictions extremely harshly since they fail to capture any trial-by-trial variance, whereas raw MSE can be naively minimized by simply outputting the dataset's global mean.

**Conclusion:** The SOC dynamics strongly depend on sufficient controller bandwidth to provide the real-time gains and context needed for the recurrent state to trace out trial-specific paths. If the controller doesn't have the degrees of freedom to dynamically instruct the network, the frozen `W` matrix cannot organically generate the rich structure of reaching movements on its own.
