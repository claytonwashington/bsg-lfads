# Future Plans & Ideas

> Notes collected during the LFADS-SOC MVP development.

## 1. Gain Branch Integration (`auto-nn` → LFADS-SOC)

The `auto-nn/` submodule has a dedicated gain encoder (`g_enc`) that learns per-neuron gains from data. Currently in LFADS-SOC, gains are produced by the controller output → `controller_to_g` linear projection. 

**Future idea**: Replace the controller-to-g pathway with a dedicated encoder that directly infers gains from neural data, similar to `autoNN_v2.encoder` in `auto-nn/auto_nn/model.py`.

## 2. Initial State from Observability Gramian

The `auto-nn` code computes the initial state `x0` from the weight matrix using the observability Gramian (Stroud et al., 2018, Nature Neuroscience):

```python
def compute_initial_state(self, W, over_tau):
    A = (W - torch.eye(N)) * over_tau
    Q = la.solve_continuous_lyapunov(A.T, -torch.eye(N))
    _, eigvecs = np.linalg.eigh(Q)
    x0 = eigvecs[:, -1]  # eigenvec with largest eigenvalue
    x0 = x0 * (1.5 * np.sqrt(N)) / np.linalg.norm(x0)
    return x0
```

Currently in LFADS-SOC, `ic_to_v0` maps the IC latent vector to the initial voltage. We could optionally bias this initialization toward the Gramian-derived x0.

## 3. PBT Hyperparameter Ranges

Biophysically plausible ranges for Population-Based Training:

| Param | Default | PBT Range | Notes |
|---|---|---|---|
| `soc_tau` | 50 ms | 10–200 ms | Membrane time constant |
| `soc_r0` | 20 Hz | 5–30 Hz | Baseline/spontaneous firing rate |
| `soc_rmax` | 100 Hz | 50–200 Hz | Maximum firing rate |
| `co_dim` | 16 | 4–32 | Controller output dimensionality |
| `soc_N` | 200 | 100–800 | SOC population size (must be even) |

## 4. EMG Readout Improvements

Current: Linear(N//2, 14) + exp() from excitatory units only.

Ideas:
- **Log-transform EMG targets** during preprocessing (as done in post-analysis script with `log_emg`), then use linear readout without exp()
- **Shared EMG readout** across sessions (since EMG dim=14 is constant) to reduce parameters
- **Separate L/R EMG readouts** for left/right leg muscles (7 channels each)

## 5. Warm-Starting from Trained LFADS

Load pre-trained LFADS encoder weights (`ic_enc`, `ci_enc`) to initialize the SOC-LFADS encoder. This could dramatically speed up convergence since the encoder task (compressing neural data into latents) is the same.

## 6. Excitatory/Inhibitory Readout Constraint

Currently the EMG readout reads from the first N/2 (excitatory) units. We could additionally enforce that the neural readout respects the E/I structure — e.g., separate excitatory and inhibitory readout heads for neural reconstruction.

## 7. RK4 Integration

The `auto-nn` code uses `torchdiffeq.odeint_adjoint` with RK4 for ODE integration:
```python
output = odeint(self.dynamics, self.x0, self.t, method='rk4')
```

Currently we use simple Euler stepping. RK4 would be more numerically accurate but slower per step. Worth benchmarking if Euler introduces instability at larger dt values.

## 8. Gain Clamping

From `auto-nn/model.py` line 644-648, gains are clamped to be non-negative after each optimizer step:
```python
def optimizer_step(self, ...):
    optimizer.step(closure=optimizer_closure)
    with torch.no_grad():
        for name, param in self.named_parameters():
            if "gains" in name:
                param.clamp_(min=0.0)
```

In our case, gains come from `softplus(controller_to_g(...))` which guarantees positivity by construction, so no clamping is needed. But if we ever switch to learned static gains (like auto-nn), we'd need this.

## 9. Linear Regime Collapse & Readout Regularization

### The Problem

The trained SOC network operates in a narrow band around the r0=20 Hz baseline (~16–25 Hz). All neurons sit deep in the **linear regime** of the piecewise tanh activation. The readout weights compensate by scaling up these tiny fluctuations (sum |w| per target reaches 10–30x). Consequences:

- The nonlinearity is effectively unused — the SOC behaves as a linear dynamical system with a linear readout
- Gain modulation reduces to simple multiplicative scaling (a linear operation) rather than nonlinear response shaping
- The biophysical constraints (0 Hz floor, 100 Hz ceiling from rmax) are never engaged
- Inhibitory units are heavily leveraged by the readout (sign-flipped via negative weights to match excitatory targets)

Evidence: `analysis_plots/soc_readout_weights.png`

### Readout Regularization Strategies (in order of increasing constraint)

#### a. Heavy L2 on Readout Weights
- Add strong weight decay specifically on `readout_neural.weight`
- Forces smaller weights → SOC must produce larger internal fluctuations to match targets
- Easiest to implement; may not fully solve the problem

#### b. Layer Normalization Before Readout
- Apply LayerNorm to SOC rates before the linear projection
- Removes the network's ability to use the readout for scaling — it can only mix/rotate
- Forces the SOC population to produce target-scale fluctuations internally

#### c. Sparse Identity Readout (Direct Subpopulation Readout)
- Replace the dense learned readout with a sparse binary selection matrix (71×200, each row is one-hot)
- Each target neuron is modeled as **exactly one** SOC unit — no scaling, no mixing, no sign-flipping
- The SOC unit must physically match the target's firing rate scale (0–150 Hz), forcing neurons deep into the nonlinear regime
- Can allow selection from both E and I pools — the model learns which recorded neurons are excitatory vs inhibitory
- Most biologically literal interpretation: recorded neurons *are* a subset of this circuit
- Fits naturally: 71 target neurons < 200 SOC units

### Note on r0 Estimation
Collaborator suggested estimating r0 from the neural population before running. This **will not help** — the issue is not the baseline value itself but the fact that the network can stay near *any* baseline and let the readout handle the scaling. Changing r0 to match the data would just shift where the network hovers without forcing it into the nonlinear regime.

## 10. iLQR-VAE with Structured Priors

Replace the LFADS encoder/controller architecture with **iLQR-VAE** for more principled inference over the SOC control inputs:

- **Sparse input prior on I_e**: Enforce that tonic external inputs are sparse in time (most timesteps near zero), reflecting the biological assumption that external drive is intermittent, not continuously modulated
- **Autoregressive prior on gains (g)**: Enforce smooth, slow gain modulation via an AR(1) or AR(p) prior — gains should change on behavioral timescales (hundreds of ms), not bin-by-bin (10 ms)
- **PBT for prior hyperparameters**: Use Population-Based Training to tune the AR prior parameters (tau, scale) and sparsity penalties, avoiding expensive manual grid search
- **Advantage**: iLQR-VAE performs amortized inference over the control inputs directly, which is a natural fit for the SOC framework where I_e and g *are* the control inputs to the dynamics

