from __future__ import annotations

import torch
from torch import nn

from .initializers import init_gru_cell_


class ClippedGRUCell(nn.GRUCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
        is_encoder: bool = False,
    ):
        super().__init__(input_size, hidden_size, bias=True)
        self.bias_hh.requires_grad = False
        self.clip_value = clip_value
        scale_dim = input_size + hidden_size if is_encoder else None
        init_gru_cell_(self, scale_dim=scale_dim)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        x_all = input @ self.weight_ih.T + self.bias_ih
        x_z, x_r, x_n = torch.chunk(x_all, chunks=3, dim=1)
        split_dims = [2 * self.hidden_size, self.hidden_size]
        weight_hh_zr, weight_hh_n = torch.split(self.weight_hh, split_dims)
        bias_hh_zr, bias_hh_n = torch.split(self.bias_hh, split_dims)
        h_all = hidden @ weight_hh_zr.T + bias_hh_zr
        h_z, h_r = torch.chunk(h_all, chunks=2, dim=1)
        z = torch.sigmoid(x_z + h_z)
        r = torch.sigmoid(x_r + h_r)
        h_n = (r * hidden) @ weight_hh_n.T + bias_hh_n
        n = torch.tanh(x_n + h_n)
        hidden = z * hidden + (1 - z) * n
        hidden = torch.clamp(hidden, -self.clip_value, self.clip_value)
        return hidden


class ClippedGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
    ):
        super().__init__()
        self.cell = ClippedGRUCell(
            input_size, hidden_size, clip_value=clip_value, is_encoder=True
        )

    def forward(self, input: torch.Tensor, h_0: torch.Tensor):
        hidden = h_0
        input = torch.transpose(input, 0, 1)
        output = []
        for input_step in input:
            hidden = self.cell(input_step, hidden)
            output.append(hidden)
        output = torch.stack(output, dim=1)
        return output, hidden


class BidirectionalClippedGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
    ):
        super().__init__()
        self.fwd_gru = ClippedGRU(input_size, hidden_size, clip_value=clip_value)
        self.bwd_gru = ClippedGRU(input_size, hidden_size, clip_value=clip_value)

    def forward(self, input: torch.Tensor, h_0: torch.Tensor):
        h0_fwd, h0_bwd = h_0
        input_fwd = input
        input_bwd = torch.flip(input, [1])
        output_fwd, hn_fwd = self.fwd_gru(input_fwd, h0_fwd)
        output_bwd, hn_bwd = self.bwd_gru(input_bwd, h0_bwd)
        output_bwd = torch.flip(output_bwd, [1])
        output = torch.cat([output_fwd, output_bwd], dim=2)
        h_n = torch.stack([hn_fwd, hn_bwd])
        return output, h_n


class SOCCell(nn.Module):
    """Discretized Euler-step SOC (Stability-Optimized Circuit) cell.

    Implements a fixed E/I recurrent network driven by time-varying tonic
    inputs (I_e) and gain modulation (g). The weight matrix W is frozen
    (registered as a buffer, never updated by the optimizer).

    Dynamics (from Hennequin et al., Nature 2018):
        τ V̇ᵢ(t) = -Vᵢ(t) + Σⱼ Wᵢⱼ r[gⱼ(t), Vⱼ(t)] + Iₑ(t)

    Activation function (piecewise asymmetric tanh):
        r(g, V) = r0 * tanh(g*V / r0),              if V < 0
                  (rmax - r0) * tanh(g*V / (rmax-r0)), if V >= 0

    Where r0 is the baseline firing rate and rmax is the maximum firing rate.
    The gain g modulates the slope of the activation around V=0.

    Convention: W[i, j] = connection from neuron j → neuron i (post, pre).
    First N//2 columns are excitatory (positive), last N//2 are inhibitory (negative).

    Parameters
    ----------
    N : int
        Number of neurons in the SOC population. Must be even.
    dt : float
        Euler integration step size (ms).
    tau : float
        Membrane time constant (ms). Default 50ms, tunable via PBT.
    r0 : float
        Baseline firing rate (Hz). Default 20.
    rmax : float
        Maximum firing rate (Hz). Default 100.
    W_init_path : str
        Path to a .pt file containing the pre-computed SOC weight matrix
        of shape (N, N), saved via ``torch.save(W_tensor, path)``.
    """

    def __init__(
        self,
        N: int,
        dt: float,
        tau: float,
        W_init_path: str,
        r0: float = 20.0,
        rmax: float = 100.0,
    ):
        super().__init__()
        assert N % 2 == 0, f"SOCCell requires even N, got {N}"
        self.N = N
        self.dt = dt
        self.tau = tau
        self.r0 = r0
        self.rmax = rmax

        # Load pre-computed SOC weight matrix
        W = torch.load(W_init_path, map_location="cpu", weights_only=True)
        assert W.shape == (N, N), (
            f"Expected W shape ({N}, {N}), got {W.shape}"
        )
        W = W.float()

        # Enforce E/I sign convention
        W[:, : N // 2] = torch.abs(W[:, : N // 2])   # excitatory columns
        W[:, N // 2 :] = -torch.abs(W[:, N // 2 :])   # inhibitory columns

        # CRITICAL: register as buffer — NOT nn.Parameter
        self.register_buffer("W", W)  # (N, N), requires_grad=False

    def _activation(self, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Piecewise asymmetric tanh activation with gain modulation.

        Matches auto_nn/model.py f_nonlinear and the formulation in
        Hennequin et al., Nature 2018 (see also Stroud et al., 2018).

        Parameters
        ----------
        v : Tensor, shape (B, N)
            Membrane voltage.
        g : Tensor, shape (B, N)
            Per-neuron gain (strictly positive).

        Returns
        -------
        r : Tensor, shape (B, N)
            Firing rates.
        """
        scaled = g * v  # (B, N)
        neg_branch = self.r0 * torch.tanh(scaled / self.r0)
        pos_branch = (self.rmax - self.r0) * torch.tanh(
            scaled / (self.rmax - self.r0)
        )
        return torch.where(v < 0, neg_branch, pos_branch)

    def forward(
        self,
        v_prev: torch.Tensor,
        I_e: torch.Tensor,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One Euler step of the SOC dynamics.

        Parameters
        ----------
        v_prev : Tensor, shape (B, N)
            Previous membrane voltage.
        I_e : Tensor, shape (B, N)
            Tonic input current from the controller.
        g : Tensor, shape (B, N)
            Per-neuron gain (must be strictly positive; apply softplus upstream).

        Returns
        -------
        v_next : Tensor, shape (B, N)
            Updated membrane voltage.
        r_prev : Tensor, shape (B, N)
            Firing rates at the *current* step (used for readout).
        """
        # Compute firing rates via piecewise asymmetric tanh
        r_prev = self._activation(v_prev, g)  # (B, N)

        # Recurrent input: W @ r  (using W[i,j] = j→i convention)
        # (N, N) @ (B, N, 1) → (B, N, 1) → (B, N)
        Wr = (self.W @ r_prev.unsqueeze(-1)).squeeze(-1)  # (B, N)

        # Total drive: τ V̇ = -V + W·r + I_e
        dv = -v_prev + Wr + I_e  # (B, N)

        # Euler integration
        alpha = self.dt / self.tau
        v_next = v_prev + alpha * dv  # (B, N)

        return v_next, r_prev
