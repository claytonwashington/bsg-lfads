from __future__ import annotations

"""SOC Decoder — replaces the standard LFADS GRU-based Decoder.

Reuses the Controller GRU from lfads-torch but replaces the GRU generator
with an SOCCell Euler-step loop. The controller output is projected through
two linear layers to produce per-neuron tonic input (I_e) and gain (g).

The controller receives feedback from the SOC firing rates (dim N) instead
of the standard LFADS factor representation (dim fac_dim).
"""

import torch
import torch.nn.functional as F
from torch import nn

from .initializers import init_linear_
from .recurrent import ClippedGRUCell, SOCCell


class SOCDecoder(nn.Module):
    """Decoder that drives an SOCCell from LFADS encoder outputs.

    Parameters
    ----------
    hparams : SimpleNamespace or similar
        Must contain: ic_dim, ci_enc_dim, con_dim, co_dim, co_prior,
        cell_clip, dropout_rate, ext_input_dim, recon_seq_len.
    soc_cell : SOCCell
        Pre-constructed SOCCell instance (holds the frozen W matrix).
    """

    def __init__(self, hparams, soc_cell: SOCCell):
        super().__init__()
        self.hparams = hps = hparams
        self.soc_cell = soc_cell
        N = soc_cell.N

        # Map IC latent → initial SOC voltage
        self.ic_to_v0 = nn.Linear(hps.ic_dim, N)  # (ic_dim) → (N)
        init_linear_(self.ic_to_v0)

        # Decide whether to use the controller
        self.use_con = all(
            [hps.ci_enc_dim > 0, hps.con_dim > 0, hps.co_dim > 0]
        )

        if self.use_con:
            # Controller GRU — input is [ci_t, rate_feedback]
            # ci_t: (2 * ci_enc_dim),  rate_feedback: (N)
            con_input_dim = 2 * hps.ci_enc_dim + N
            self.con_cell = ClippedGRUCell(
                con_input_dim, hps.con_dim, clip_value=hps.cell_clip
            )
            # Mapping from controller state → controller output dist params
            self.co_linear = nn.Linear(hps.con_dim, hps.co_dim * 2)
            init_linear_(self.co_linear)

        # Controller output → SOC physical parameters
        self.controller_to_Ie = nn.Linear(hps.co_dim, N)  # (co_dim) → (N)
        self.controller_to_g = nn.Linear(hps.co_dim, N)   # (co_dim) → (N)
        init_linear_(self.controller_to_Ie)
        init_linear_(self.controller_to_g)

        # Learnable initial hidden state for controller
        self.con_h0 = nn.Parameter(
            torch.zeros((1, hps.con_dim), requires_grad=True)
        )

        self.dropout = nn.Dropout(hps.dropout_rate)

    def forward(
        self,
        ic_samp: torch.Tensor,
        ci: torch.Tensor,
        ext_input: torch.Tensor,
        sample_posteriors: bool = True,
    ) -> tuple:
        """Unroll the SOC decoder over the full temporal sequence.

        Parameters
        ----------
        ic_samp : Tensor, shape (B, ic_dim)
            Sampled (or mean) initial condition from the IC encoder.
        ci : Tensor, shape (B, T, 2 * ci_enc_dim)
            Controller input sequence from the CI encoder.
        ext_input : Tensor, shape (B, T, ext_input_dim)
            External inputs (unused in SOC MVP, placeholder for compatibility).
        sample_posteriors : bool
            If True, sample from the controller output posterior; else use mean.

        Returns
        -------
        gen_init : Tensor, shape (B, N)
            Initial SOC voltage v_0.
        rates : Tensor, shape (B, T, N)
            SOC firing rates at each timestep.
        con_states : Tensor, shape (B, T, con_dim)
            Controller hidden states.
        co_means : Tensor, shape (B, T, co_dim)
            Controller output means.
        co_stds : Tensor, shape (B, T, co_dim)
            Controller output standard deviations.
        gen_inputs : Tensor, shape (B, T, co_dim)
            Controller outputs (the gen_input signal).
        """
        hps = self.hparams
        B = ic_samp.shape[0]
        T = ci.shape[1]
        N = self.soc_cell.N
        device = ic_samp.device

        # --- Initial SOC voltage from IC latent ---
        gen_init = self.ic_to_v0(ic_samp)  # (B, N)
        v = gen_init

        # --- Initialize controller state and rate feedback ---
        con_state = self.con_h0.expand(B, -1).contiguous()  # (B, con_dim)
        r_feedback = torch.zeros(B, N, device=device)  # (B, N)

        # --- Storage for outputs ---
        all_rates = []
        all_con_states = []
        all_co_means = []
        all_co_stds = []
        all_gen_inputs = []

        # --- Temporal loop ---
        for t in range(T):
            ci_t = ci[:, t, :]  # (B, 2 * ci_enc_dim)

            if self.use_con:
                # Controller input: [ci_t, rate_feedback]
                con_input = torch.cat([ci_t, r_feedback], dim=1)
                con_input_drop = self.dropout(con_input)
                # Step controller GRU
                con_state = self.con_cell(con_input_drop, con_state)  # (B, con_dim)
                # Controller output distribution
                co_params = self.co_linear(con_state)  # (B, 2 * co_dim)
                co_mean, co_logvar = torch.split(co_params, hps.co_dim, dim=1)
                co_std = torch.sqrt(torch.exp(co_logvar))
                # Sample or use mean
                co_post = hps.co_prior.make_posterior(co_mean, co_std)
                con_output = (
                    co_post.rsample() if sample_posteriors else co_mean
                )  # (B, co_dim)
            else:
                con_output = torch.zeros(B, hps.co_dim, device=device)
                co_mean = torch.zeros(B, hps.co_dim, device=device)
                co_std = torch.ones(B, hps.co_dim, device=device)

            # --- Map controller output → SOC parameters ---
            I_e = self.controller_to_Ie(con_output)  # (B, N)
            # CRITICAL: softplus ensures gain is strictly positive
            g = F.softplus(self.controller_to_g(con_output))  # (B, N)

            # --- SOC Euler step ---
            v, r = self.soc_cell(v, I_e, g)  # (B, N) each

            # Rate feedback to controller for next timestep
            r_feedback = r

            # Store
            all_rates.append(r)
            all_con_states.append(con_state)
            all_co_means.append(co_mean)
            all_co_stds.append(co_std)
            all_gen_inputs.append(con_output)

        # --- Stack outputs ---
        rates = torch.stack(all_rates, dim=1)            # (B, T, N)
        con_states = torch.stack(all_con_states, dim=1)  # (B, T, con_dim)
        co_means = torch.stack(all_co_means, dim=1)      # (B, T, co_dim)
        co_stds = torch.stack(all_co_stds, dim=1)        # (B, T, co_dim)
        gen_inputs = torch.stack(all_gen_inputs, dim=1)  # (B, T, co_dim)

        return gen_init, rates, con_states, co_means, co_stds, gen_inputs
