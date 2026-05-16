# SOC-LFADS Run Log

## Run 1: `260419_soc_mvp`

**Status**: ✅ Complete (best run so far)

| Parameter | Value |
|---|---|
| Run Dir | `runs/bsg-lfads/soc_gran/260419_soc_mvp` |
| Checkpoint | `536-537.ckpt` |
| ic_enc_dim | 128 |
| ci_enc_dim | 128 |
| con_dim | 64 |
| co_dim | 16 |
| kl_co_scale | 0.0 |
| kl_ic_scale | 0.0 |
| dropout_rate | 0.02 |
| batch_size | 256 |

**Validation R² (pure validation bins):**
| Session | R²_neural | R²_emg |
|---|---|---|
| 013 | 0.743 | 0.772 |
| 023 | 0.817 | 0.808 |
| 028 | 0.876 | 0.857 |
| 030 | 0.865 | 0.846 |
| 031 | 0.850 | 0.815 |
| 033 | 0.864 | 0.773 |

**Notes:**
- Strong reconstruction but controller is doing most of the work (KL penalties disabled)
- SOC neurons operate in narrow linear regime (~16–25 Hz around r0=20 Hz baseline)
- Readout weights are large (sum |w| per target ~10–30x), compensating for tiny SOC fluctuations
- Inhibitory units heavily leveraged by readout (sign-flipped to match excitatory targets)
- A few "puppet neurons" in E population dominated by controller signal

---

## Run 2: `260419_soc_v2_small_ctrl`

**Status**: ⚠️ Complete but poor performance

| Parameter | Value |
|---|---|
| Run Dir | `runs/bsg-lfads/soc_gran/260419_soc_v2_small_ctrl` |
| Checkpoint | `533-4272.ckpt` (+ `last.ckpt`) |
| ic_enc_dim | 128 |
| ci_enc_dim | 64 |
| con_dim | 32 |
| co_dim | 8 |
| kl_co_scale | 0.0 |
| kl_ic_scale | 0.0 |
| dropout_rate | 0.02 |
| batch_size | 64 |

**Validation R² (pure validation bins):**
| Session | R²_neural | R²_emg |
|---|---|---|
| 013 | 0.838 | 0.797 |
| 023 | 0.904 | 0.813 |
| 028 | 0.929 | 0.845 |
| 030 | 0.917 | 0.878 |
| 031 | 0.920 | 0.836 |
| 033 | 0.906 | 0.745 |

**Notes:**
- Reduced controller capacity (ci_enc 128→64, con 64→32, co 16→8) to force SOC dynamics to work harder
- Ran for many more steps (4272 vs 537)
- **TensorBoard showed valid/r2_neural ≈ -929 million** throughout training, which made this run look catastrophic
- Post-hoc merged R² is actually strong (0.84–0.93) — see caveat below
- The discrepancy is due to **terrible initial transients**: early chops (beginning of session) have garbage predictions (chop 0: R²=-1195, chop 5: R²=-2.7) while later chops are fine (chop 15: R²=0.64). TB averages across all chops including these disasters
- Longer training + smaller controller may have led to better generalization (less overfitting via controller)
- Still needs investigation: does the smaller controller push neurons into a more nonlinear regime?

---

## ⚠️ R² Metric Caveat

**TensorBoard R² and post-hoc merged R² measure very different things:**

- **TensorBoard (during training):** Computes R² on individual 100-bin validation chops, then averages. A single chop with R²=-1000 (e.g., from bad initial conditions at the start of the session) destroys the average. This is what showed -929 million for v2.
- **Post-hoc merged R²:** Uses `merge_chops()` to blend overlapping predictions into a smooth continuous trace, then computes R² over the full session (~7000+ bins). The few hundred bad initial bins are diluted by thousands of good bins.

Both metrics are "correct" but answer different questions. The TB metric is overly sensitive to initial transients and should not be used to judge overall model quality. A better TB metric would exclude the first N chops or use a warm-up masking strategy.
