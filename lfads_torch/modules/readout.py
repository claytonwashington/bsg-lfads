from __future__ import annotations

"""Dual-headed readout for the SOC-LFADS model.

Provides two parallel readout heads:
  - **Neural**: Linear(N → neural_dim) — uses all N SOC units
  - **EMG**:   Linear(N//2 → emg_dim) + exp — uses only excitatory units

This enforces the biological constraint that muscles (EMG) are only driven
by excitatory projections from the SOC population.
"""

import copy
from glob import glob

import h5py
import torch
from torch import nn

from .initializers import init_linear_


class DualReadout(nn.Module):
    """Single-session dual readout: neural (all N units) + EMG (excitatory N//2).

    Parameters
    ----------
    N : int
        SOC population size (must be even).
    neural_dim : int
        Number of neural output channels (session-specific).
    emg_dim : int
        Number of EMG output channels (typically 14).
    """

    def __init__(self, N: int, neural_dim: int, emg_dim: int):
        super().__init__()
        self.N = N
        self.neural_dim = neural_dim
        self.emg_dim = emg_dim

        self.readout_neural = nn.Linear(N, neural_dim)
        self.readout_emg = nn.Linear(N // 2, emg_dim)
        init_linear_(self.readout_neural)
        init_linear_(self.readout_emg)

    def forward(
        self, r: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Apply dual readout to the SOC rate tensor.

        Parameters
        ----------
        r : Tensor, shape (B, T, N)
            Full firing rate tensor from the SOC decoder.

        Returns
        -------
        dict with keys:
            "neural" : Tensor, shape (B, T, neural_dim)
            "emg"    : Tensor, shape (B, T, emg_dim) — positive via exp()
        """
        neural_out = self.readout_neural(r)  # (B, T, neural_dim)

        # Slice excitatory units only for EMG readout
        r_exc = r[:, :, : self.N // 2]  # (B, T, N//2)
        emg_linear = self.readout_emg(r_exc)  # (B, T, emg_dim)
        emg_out = torch.exp(emg_linear)  # (B, T, emg_dim), strictly positive

        return {"neural": neural_out, "emg": emg_out}


class MultisessionDualReadout(nn.ModuleList):
    """Per-session DualReadout, created from paired spike/EMG data files.

    Each session gets its own DualReadout with session-specific neural_dim
    (inferred from the spike H5 file) and shared emg_dim (inferred from
    the EMG H5 file).

    Parameters
    ----------
    spike_file_pattern : str
        Glob pattern for spike H5 files (e.g., "datasets/spikes_gran/*").
    emg_file_pattern : str
        Glob pattern for EMG H5 files (e.g., "datasets/emg_gran/*").
    soc_N : int
        SOC population size.
    """

    def __init__(
        self,
        spike_file_pattern: str,
        emg_file_pattern: str,
        soc_N: int,
    ):
        modules = []
        spike_paths = sorted(glob(spike_file_pattern))
        emg_paths = sorted(glob(emg_file_pattern))
        assert len(spike_paths) == len(emg_paths), (
            f"Mismatched number of spike ({len(spike_paths)}) and "
            f"EMG ({len(emg_paths)}) data files."
        )
        for spike_path, emg_path in zip(spike_paths, emg_paths):
            with h5py.File(spike_path, "r") as f_spike:
                neural_dim = f_spike["train_recon_data"].shape[-1]
            with h5py.File(emg_path, "r") as f_emg:
                emg_dim = f_emg["train_recon_data"].shape[-1]
            modules.append(DualReadout(soc_N, neural_dim, emg_dim))
        super().__init__(modules)
