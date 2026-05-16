# snel-toolkit Reference for LFADS Pipeline

> Conda env: `stkit-nwb`
> Location: `/home/cbwash2/snel-scpu-analysis/snel-toolkit/snel_toolkit/`

## Key Concepts

### NWBDataset

Core data container. Loaded from pickle files.

```python
import pickle
ds = pickle.load(open('nlb_gran_013.pkl', 'rb'))
# type: snel_toolkit.datasets.nwb.NWBDataset
```

**`ds.data`** — MultiIndex DataFrame, continuous time series. All fields share the same time index.

| Field | Shape (timesteps, channels) | Description |
|---|---|---|
| `spikes` | (7533, 71) | Binned spike counts (0-5 range) |
| `lfads_rates` | (7533, 71) | LFADS smoothed firing rates (0-2.65 range) |
| `lfads_factors` | (7533, 20) | LFADS latent factors |
| `lfads_gen_inputs` | (7533, 6) | LFADS generator inputs |
| `deEMG_mean` | (7533, 14) | Decoded EMG mean from EMG-LFADS (0.001-2.83) |
| `deEMG_factors` | (7533, 10) | EMG LFADS factors |
| `emg` | (7533, 14) | Raw rectified EMG (-0.67 to 7.33) |
| `model_emg` | (7533, 14) | Modeled EMG (-0.35 to 3.50) |
| `model_emg_smooth_30ms` | (7533, 14) | Smoothed modeled EMG |
| `spikes_smooth_30ms` | (7533, 71) | Smoothed spikes (30ms Gaussian) |

**`ds.trial_info`** — trial metadata DataFrame with columns:
- `trial_id` — integer trial identifier
- `start_time` — pandas Timedelta of trial onset
- `end_time` — pandas Timedelta of trial offset
- `condition_id` — condition grouping (1 in the raw NWB; reassigned downstream)

Session 013 has 105 trials in the raw NWB.

**`ds.bin_width`** — bin size in ms (10 for these datasets)

**`ds.unit_info`** — neuron metadata: `group_name`, `location`, etc.

### DataWrangler

From `snel_toolkit.datasets.base`. Used for trial alignment and cycle averaging.

```python
from snel_toolkit.datasets.base import DataWrangler

dw = DataWrangler(ds)
dw.make_trial_data(
    name="onset",
    align_field="start_time",      # align to trial onset
    align_range=(-100, 600),       # ms relative to align point
    ignored_trials=ignore_mask,    # boolean mask on trial_info
    allow_overlap=True,
    set_t_df=True,                 # store in dw._t_df
)
```

#### How it works internally

`make_trial_data` does the following for each non-ignored trial:

1. Reads `trial_info[align_field]` for the trial → gets the alignment timestamp
2. Computes the absolute time window: `[align_time + align_range[0], align_time + align_range[1])`
3. Indexes into the continuous data (`ds.data`, `ds.lfads_rates`, etc.) using the time index
4. Assigns relative `align_time` values (e.g. -100, -90, ..., 590 for 10ms bins)
5. Stacks all trials into a long-format DataFrame (`_t_df`) with columns:
   - `trial_id`: which trial this row belongs to
   - `align_time`: relative time within the trial (the index for pivoting)
   - All data fields from the dataset (spikes, lfads_rates, deEMG_mean, etc.)

#### Key attributes after `make_trial_data`

- `dw._t_df` — long-format DataFrame with all trial-aligned data
- `dw._t_df.align_time.unique()` — the shared time axis (e.g. 71 timepoints for `(-100, 600)` at 10ms)
- `dw.pivot_trial_df(dw._t_df, values=(field, channel))` — pivot from long to wide format:
  - Rows = `align_time` (n_timepoints)
  - Columns = `trial_id` (n_trials)
  - Values = the specified (field, channel) data

#### NaN handling

- If a trial's time window extends outside the available data, those bins are filled with NaN
- The first 4 and last 2 trials are typically excluded via `ignored_trials` (standard practice for locomotion)
- When computing cycle averages, `np.nanmean` or `skipna=True` handles any remaining NaN values

### NWBDataset.make_trial_data

Alternative to DataWrangler (used in post-analysis notebook):

```python
aligned_df = ds.make_trial_data(
    align_field="start_time",
    align_range=(-100, 220),
    allow_overlap=True,
)
# Returns a DataFrame with columns: trial_id, align_time, event_id, + all data fields
```

### Cycle Averaging

Used in PCR alignment to compute condition-averaged traces:

```python
def aligned_cycle_averaging(dataset, dw, field, field_names):
    n_time = dw._t_df.align_time.unique().size
    cycle_avg = np.full((n_time, len(field_names)), np.nan)
    for i, fname in enumerate(field_names):
        pivot = dw.pivot_trial_df(dw._t_df, values=(field, fname))
        cycle_avg[:, i] = pivot.mean(axis=1, skipna=True)
    return cycle_avg  # (n_timepoints, n_channels)
```

### LFADSInterface

Handles the **chop → LFADS → merge** pipeline. This is the critical object that must be preserved between chopping and merging.

Location: `snel_toolkit/interfaces.py`

- `window`: 1000ms = 100 bins
- `overlap`: 200ms = 20 bins
- `chop_fields_map`: `{'spikes': 'data'}` — maps NWBDataset field → H5 key
- `merge_fields_map`: `{'rates': 'lfads_rates', ...}` — maps LFADS output → dataset field

```python
# Chop and save to H5
intf.chop_and_save(neural_df, fname, valid_ratio=0.2, valid_block=1)

# Merge LFADS outputs back to continuous
intf.load_and_merge(fname, orig_df, smooth_pwr=2)
```

#### Full Merge Workflow

The standard post-training pipeline (from `merge_chopped_torch_outputs_and_kinematics.py`):

1. **Load the interface object** (pickled alongside the H5 data)
2. **Load posterior sampling H5** (contains `train_rates`, `valid_rates`, `train_inds`, `valid_inds`, etc.)
3. **Reconstruct ordered chops**: `combine_train_valid_outputs()` interleaves train/valid outputs back into their original chop order using `train_inds` and `valid_inds`
4. **Call `interface.merge(data_dict, smooth_pwr=1)`**: this calls `SegmentRecord.rebuild_segment()` for each continuous segment, which calls `merge_chops()`
5. **Result**: a continuous DataFrame indexed by the original `clock_time`, with smooth transitions at chop boundaries

#### merge_chops Algorithm (the key function)

`merge_chops(data, overlap, orig_len, smooth_pwr)` reconstructs continuous data from overlapping chops using **power-function blending**:

```
For each chop, split into 3 regions:
  [first overlap] [middle (non-overlapping)] [last overlap]

At overlap boundaries between chop_i and chop_{i+1}:
  result = chop_i_last * ramp + chop_{i+1}_first * (1 - ramp)

where ramp = 1 - x^smooth_pwr, x ∈ (0, 1) linspaced across the overlap
```

- `smooth_pwr=1`: linear interpolation (equal blending)
- `smooth_pwr=2` (default): slightly prefers the "end" of each chop (where the model has seen more context)
- `smooth_pwr=np.inf`: only keep chop ends, discard beginnings

**This is why naive concatenation produces jumps** — each chop starts with fresh initial conditions from the IC encoder, and the model needs "warm-up" time. The ramp function down-weights the beginning of each chop (where IC initialization artifacts are strongest) and up-weights the end (where dynamics have stabilized).

#### SegmentRecord

Stored by the interface during chopping, one per continuous segment. Tracks:
- `seg_id`: segment identifier
- `clock_time`: original time index (for reconstructing the DataFrame)
- `offset`: random offset applied during chopping
- `n_chops`: how many chops this segment produced
- `overlap`: overlap in bins

### Chopping scheme

Continuous data sliced into overlapping windows:
- Window = 100 bins, Stride = 80 bins
- `n_chops = floor((7533 - 100) / 80) + 1 = 93`
- Train/valid: **4 train, 1 valid** (every 5th chop → valid)
- Session 013: 74 train + 19 valid = 93 total

## PCR Alignment Process

### Overview

PCR (Principal Component Regression) maps each session's neural activity into a shared low-dimensional space. This is used to initialize the `MultisessionReadin` layers in lfads-torch.

### Procedure (from `auyong_pcr_alignment_v4.py`)

1. **Load** each session's NWBDataset pickle
2. **Preprocess**: xcorr rejection, Gaussian smoothing (30ms), EMG clipping/scaling
3. **Align** trials to `start_time` with `align_range=(-100, 600)` ms → 71 timebins
4. **Exclude** first 4 and last 2 trials per session
5. **Cycle-average** within conditions → `cycle_avg` (n_timepoints, n_neurons) per session
6. **Concatenate** horizontally across sessions → `global_avg` (n_timepoints, total_neurons)
7. **PCA** on mean-centered global data (excluding NaN rows) → `global_pcs` (n_valid_times, n_pcs)
8. **Ridge regression** per session: fit `(session_avg - session_mean) → global_pcs`
   - `W = lr.coef_.T` → shape `(n_neurons, n_pcs)` → this is `readin_weight`
   - `readout_bias = session_mean` → shape `(n_neurons,)` — channel means

### Parameters

| Parameter | Spike PCR | EMG PCR |
|---|---|---|
| n_pcs | 20 | 10 |
| L2 (Ridge α) | 0.01 | 0.01 |
| align_range | (-100, 600) ms | (-100, 600) ms |
| smoothing | 30ms Gaussian | 30ms Gaussian |

### How lfads-torch uses PCR matrices

In `MultisessionReadin._get_state_dict()`:
```python
weight = h5file["readin_weight"][()]          # (n_neurons, n_pcs)
bias = -np.dot(h5file["readout_bias"][()], weight)  # (n_pcs,)
# nn.Linear weight is transposed: (n_pcs, n_neurons)
```

In `MultisessionReadout._get_state_dict()`:
```python
weight = np.linalg.pinv(h5file["readin_weight"][()])  # pseudoinverse
bias = h5file["readout_bias"][()]
```

### PCR alignment output format

```
pcr_alignment_torch.h5
├── lfads_gran_013_ALL_spikes_10.h5/
│   ├── matrix: (71, 20)    # readin_weight
│   └── bias: (71,)         # readout_bias = channel means
├── lfads_gran_023_ALL_spikes_10.h5/
│   └── ...
```

## Trial Detection in Post-Analysis

The post-analysis notebook (`auyong_post_path_length_notebook_v3.py`) **reconstructs** trials from EMG data:

1. Compute differentiation of `deEMG_mean[RBA]` (right biceps anterior — extensor)
2. Find positive peaks (onsets) and negative peaks (offsets) in the differentiated signal
3. Find troughs in the EMG envelope (change points between steps)
4. Between consecutive change points: onset = first positive peak, offset = last negative peak
5. Filter by burst duration: 50ms < duration < 800ms
6. **Overwrite** `ds.trial_info` with these newly-detected step cycles

This produces more physiologically meaningful trials than the NWB's original trial_info.

The post-analysis then fits **its own PCR** (separate from the one used for LFADS training) on `lfads_factors` with different parameters.

## Data File Locations

| What | Path |
|---|---|
| Raw datasets (pkl) | `/snel/share/share/tmp/scratch/cbwash2/auyong_nwb/binsize_10ms_pcr_high_reg_ALL_cw/nlb_gran_*.pkl` |
| Chopped H5s | `.../gran/datasets/lfads_gran_*_ALL_spikes_10.h5` and `*_emg_10.h5` |
| Interface objects | `.../gran/spikes/lfads_input/gran_*_interface.pkl` |
| PCR alignment (spikes) | `.../gran/alignment_matrices/spikes/pcr_alignment_torch.h5` |
| PCR alignment (EMG) | `.../gran/alignment_matrices/emg/pcr_alignment_torch.h5` |
| Trained spike model | `.../gran/spikes/run_pcr_freeze_3/pbt_run/best_model/posterior_samples.h5` |
| Trained EMG model | `.../gran/emg/run_pcr_freeze/pbt_run/best_model/posterior_samples.h5` |
| lfads-torch readin H5s | `~/lfads-torch-fork/lfads-torch/datasets/spikes_gran/` and `emg_gran/` |
| **SOC data (NEW)** | `~/bsg-lfads/datasets/soc_gran/neural/` and `emg/` |

## Dimension Summary

| Quantity | Session 013 | All Sessions |
|---|---|---|
| Raw neurons | 71 | 71, 62, 65, 63, 62, 59 |
| EMG channels | 14 | 14 (shared) |
| Continuous timesteps | 7533 | varies |
| Bin size | 10 ms | 10 ms |
| Chop window | 100 bins | 100 bins |
| Train/valid chops | 74 / 19 | varies |
| LFADS factors dim | 20 | 20 |
| LFADS gen_dim | 100 | 100 |
| Neural PCR readin dim | 20 | 20 |
| EMG PCR readin dim | 10 | 10 |

## Environment Notes

| Environment | Purpose |
|---|---|
| `stkit-nwb` | snel-toolkit, NWBDataset loading, PCR alignment, data prep |
| `lfads-torch-cuda12` | Model code, training, evaluation |

The two environments are needed because snel-toolkit has different dependency versions than lfads-torch.
- `stkit-nwb` has Python 3.7 + torch 1.13 — cannot import bsg-lfads model code (needs `Literal` from Python 3.8+)
- `lfads-torch-cuda12` has Python 3.9 + torch 2.x — cannot import `snel_toolkit` (dependency conflicts)

## SOC-LFADS Analysis Pipeline

### Scripts (in `~/bsg-lfads/scripts/`)

| Script | Env | Purpose |
|---|---|---|
| `prepare_soc_data.py` | `stkit-nwb` | Loads NWBDataset pkls, runs PCR alignment, chops into overlapping windows (100 bins, 20 overlap), saves H5s |
| `run_soc.py` | `lfads-torch-cuda12` | Trains SOC-LFADS v1 model |
| `run_soc_v2.py` | `lfads-torch-cuda12` | Trains SOC-LFADS v2 (reduced controller) for ablation |
| `soc_posterior_sampling.py` | `lfads-torch-cuda12` | **Step 1**: Runs inference on ALL chops (train+valid in original order), saves raw chopped outputs to H5 |
| `merge_soc_outputs.py` | `stkit-nwb` | **Step 2**: Loads pkl + H5, applies `merge_chops()` for smooth blending, saves merged analysis pkl, generates plots |
| `analyze_soc.py` | `lfads-torch-cuda12` | (OLD) Naive chop concatenation — **superseded by the two-step pipeline above** |
| `analyze_soc_merged.py` | — | (UNUSED) Attempted single-script approach — doesn't work due to env conflicts |

### Data Artifacts

| What | Path |
|---|---|
| SOC training data (neural) | `~/bsg-lfads/datasets/soc_gran/neural/lfads_torch_readin*_neural.h5` |
| SOC training data (EMG) | `~/bsg-lfads/datasets/soc_gran/emg/lfads_torch_readin*_emg.h5` |
| SOC weight matrix | `~/bsg-lfads/weights/W_soc_200.pt` |
| v1 checkpoint (best) | `~/bsg-lfads/runs/bsg-lfads/soc_gran/260419_soc_mvp/lightning_checkpoints/536-537.ckpt` |
| v2 checkpoint (ablation) | `~/bsg-lfads/runs/bsg-lfads/soc_gran/260420_soc_v2_small_controller/lightning_checkpoints/` |
| Raw chopped outputs (H5) | `~/bsg-lfads/runs/bsg-lfads/soc_gran/260419_soc_mvp/soc_output_all_chops.h5` |
| **Merged analysis (pkl)** | `~/bsg-lfads/runs/bsg-lfads/soc_gran/260419_soc_mvp/soc_merged_analysis.pkl` |
| Presentation plots | `~/bsg-lfads/analysis_plots/*_merged.png` |
| Original NWBDataset pkls | `/snel/share/share/tmp/scratch/cbwash2/auyong_nwb/binsize_10ms_pcr_high_reg_ALL_cw/nlb_gran_*.pkl` |

### Merged Analysis Object Structure

`soc_merged_analysis.pkl` is a dict keyed by session ID (str: "013", "023", ...):

```python
merged_results[sid] = {
    "neural_pred":         np.array (T, n_neurons),  # SOC model's neural prediction (continuous, merged)
    "emg_pred":            np.array (T, 14),          # SOC model's EMG prediction
    "rates":               np.array (T, 200),         # SOC population firing rates (100 E + 100 I)
    "gen_inputs":          np.array (T, 16),           # Controller outputs (co_dim=16, pre gain/I_e split)
    "co_means":            np.array (T, 16),           # Same as gen_inputs (posterior mean, no sampling)
    "neural_target":       np.array (T, n_neurons),   # Original LFADS rates from pkl (ground truth)
    "emg_target":          np.array (T, 14),           # Original deEMG_mean from pkl (ground truth)
    "neural_target_merged": np.array (T, n_neurons),  # Target after chop→merge (slightly smoothed)
    "emg_target_merged":   np.array (T, 14),           # Target after chop→merge
    "time_index":          pd.TimedeltaIndex (T,),     # Original time index from NWBDataset
}
```

**Session dimensions**: 013=7533, 023=8162, 028=8330, 030=7743, 031=7793, 033=6736 bins

### Two-Step Inference + Merge Workflow

**Why two steps?** The model code requires `lfads-torch-cuda12` (Python 3.9+), but `merge_chops` and the NWBDataset pkls require `snel_toolkit` which only works in `stkit-nwb` (Python 3.7).

1. **Step 1** (`soc_posterior_sampling.py` in `lfads-torch-cuda12`):
   - Loads the best checkpoint
   - Reads the chopped H5 data, reconstructs original chop order from train/valid indices
   - Runs model.forward() on all chops in batches of 64
   - Saves all chopped outputs to `soc_output_all_chops.h5`

2. **Step 2** (`merge_soc_outputs.py` in `stkit-nwb`):
   - Loads the original NWBDataset pkls (for ground truth and time indices)
   - Loads the chopped H5 from Step 1
   - Calls `merge_chops(chops, overlap=20, orig_len=T, smooth_pwr=2)` for each output field
   - Saves `soc_merged_analysis.pkl`
   - Generates all presentation plots

### Current Model Performance (v1, merged)

| Session | Neural R² | EMG R² |
|---|---|---|
| 013 | 0.833 | 0.908 |
| 023 | 0.855 | 0.917 |
| 028 | 0.898 | 0.923 |
| 030 | 0.877 | 0.917 |
| 031 | 0.893 | 0.923 |
| 033 | 0.888 | 0.907 |
| **Mean** | **0.874** | **0.916** |

Per-muscle EMG R² range: 0.89 (DeltM) to 0.94 (SupSp, TLaLo)

### Plot Inventory (`~/bsg-lfads/analysis_plots/`)

| File | Description |
|---|---|
| `emg_traces_merged.png` | 8 EMG muscles, target vs SOC pred, 15s window |
| `neural_traces_merged.png` | 6 neurons, target vs SOC pred, 15s window |
| `soc_population_heatmap_merged.png` | 3-row heatmap: E units / I units / LFADS rates |
| `r2_bar_chart_merged.png` | Per-session R² bars (neural + EMG) |
| `emg_r2_per_muscle_merged.png` | Per-muscle R² for all 14 muscles |
| `controller_emg_merged.png` | All 16 CO dims + BicL + DeltA traces |
| `pc_comparison_procrustes.png` | Procrustes-aligned PCs: controller vs EMG (disp=0.44) and neural (disp=0.33) |
| `gain_covariance_matrix.png` | 200×200 gain covariance with E/I partition + per-neuron gain variance |
| `peak_sorted_cascade.png` | Peak-time-sorted heatmaps: E units / I units / LFADS neurons |
| `gain_time_series.png` | Population-average gain traces (E vs I) + per-neuron gain heatmaps |

### Additional Scripts

| Script | Env | Purpose |
|---|---|---|
| `extract_gain_weights.py` | `lfads-torch-cuda12` | Extracts `controller_to_g` and `controller_to_Ie` weights from checkpoint → saves `gain_projection_weights.npz` |
| `soc_advanced_analyses.py` | `stkit-nwb` | Generates PC comparison, gain covariance, cascade, and gain time series plots from merged pkl |

### Additional Data Artifacts

| What | Path |
|---|---|
| Gain projection weights | `~/bsg-lfads/runs/bsg-lfads/soc_gran/260419_soc_mvp/gain_projection_weights.npz` |

### Key Findings from Advanced Analyses

- **Gains are sparse**: Only ~5 of 200 neurons receive strong gain modulation; the controller uses a sparse strategy
- **E gain > I gain**: Mean E gain (~0.38) consistently exceeds mean I gain (~0.27), both with rhythmic modulation
- **Gain covariance clusters by type**: Mean |cov| E↔E (0.018) > E↔I (0.011) > I↔I (0.007) — gains are more correlated within E population
- **Controller is more neural-like**: Procrustes disparity to neural (0.33) < EMG (0.44) — the controller drives like neural activity, not like EMG
- **Cascading activation**: Peak-sorted plots show clear sequential activation cascades in both E and I SOC populations during locomotion

