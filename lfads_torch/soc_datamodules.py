"""SOC-specific DataModule that loads paired spike and EMG H5 files.

Unlike the standard lfads-torch BasicDataModule (which loads a single set of
H5 files for reconstruction), this module loads spike and EMG files in parallel
and pairs them by session. The encoder receives neural data only; reconstruction
targets include both neural and EMG data.

Expected H5 file format (per session):
  Spike file:
    train_encod_data  (n_trials, T, raw_neural_dim)
    train_recon_data  (n_trials, T, raw_neural_dim)
    valid_encod_data  (n_trials, T, raw_neural_dim)
    valid_recon_data  (n_trials, T, raw_neural_dim)
    readin_weight     (raw_neural_dim, encod_data_dim)
    readout_bias      (raw_neural_dim,)

  EMG file:
    train_recon_data  (n_trials, T, emg_dim)
    valid_recon_data  (n_trials, T, emg_dim)
"""

from glob import glob

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .tuples import SessionBatch


def _to_tensor(array):
    return torch.tensor(array, dtype=torch.float)


class SOCSessionDataset(Dataset):
    """Dataset for a single session with both neural and EMG targets.

    Each item returns:
        (model_batch, (emg_recon_target,))

    where model_batch is a SessionBatch (encod_data, recon_data, ext_input,
    truth, sv_mask) following the standard lfads-torch convention, and
    emg_recon_target is the EMG reconstruction target tensor.
    """

    def __init__(
        self,
        model_tensors: SessionBatch,
        emg_recon: Tensor,
    ):
        all_tensors = [*model_tensors, emg_recon]
        assert all(
            all_tensors[0].size(0) == t.size(0) for t in all_tensors
        ), "Size mismatch between tensors"
        self.model_tensors = model_tensors
        self.emg_recon = emg_recon

    def __getitem__(self, index):
        model_tensors = SessionBatch(*[t[index] for t in self.model_tensors])
        emg = self.emg_recon[index]
        return model_tensors, (emg,)

    def __len__(self):
        return len(self.model_tensors[0])


class SOCDataModule(pl.LightningDataModule):
    """DataModule that loads paired spike and EMG H5 files for SOC-LFADS.

    Parameters
    ----------
    spike_file_pattern : str
        Glob pattern for spike H5 files.
    emg_file_pattern : str
        Glob pattern for EMG H5 files.
    batch_size : int
        Batch size (divided across sessions).
    sv_rate : float
        Sample validation rate (0 = no sample validation).
    sv_seed : int
        Seed for sample validation mask.
    dm_ic_enc_seq_len : int
        Number of IC encoder-only timesteps to remove from ext_input/truth.
    """

    def __init__(
        self,
        spike_file_pattern: str,
        emg_file_pattern: str,
        batch_size: int = 64,
        sv_rate: float = 0.0,
        sv_seed: int = 0,
        dm_ic_enc_seq_len: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        hps = self.hparams
        spike_paths = sorted(glob(hps.spike_file_pattern))
        emg_paths = sorted(glob(hps.emg_file_pattern))
        assert len(spike_paths) == len(emg_paths), (
            f"Mismatched spike ({len(spike_paths)}) and EMG ({len(emg_paths)}) files."
        )

        sv_gen = torch.Generator().manual_seed(hps.sv_seed)
        all_train_data, all_valid_data = [], []

        for spike_path, emg_path in zip(spike_paths, emg_paths):
            # Load spike data
            with h5py.File(spike_path, "r") as f_spike:
                spike_dict = {k: v[()] for k, v in f_spike.items()}
            # Load EMG data
            with h5py.File(emg_path, "r") as f_emg:
                emg_dict = {k: v[()] for k, v in f_emg.items()}

            for prefix in ["train", "valid"]:
                # --- Neural SessionBatch (encoder input + neural recon target) ---
                encod_data = _to_tensor(spike_dict[f"{prefix}_encod_data"])
                n_samps, n_steps, _ = encod_data.shape

                # Neural reconstruction target
                recon_data = _to_tensor(spike_dict[f"{prefix}_recon_data"])

                # Sample validation mask
                if hps.sv_rate > 0:
                    bern_p = 1 - hps.sv_rate if prefix != "test" else 1.0
                    sv_mask = (
                        torch.rand(encod_data.shape, generator=sv_gen) < bern_p
                    ).float()
                else:
                    sv_mask = torch.ones(n_samps, 0, 0)

                # External inputs (placeholder — not used in SOC MVP)
                ext_input = torch.zeros(n_samps, n_steps, 0)
                # Truth (placeholder)
                truth = torch.full((n_samps, 0, 0), float("nan"))

                # Remove IC encoder-only segment from non-encoder tensors
                sv_mask = sv_mask[:, hps.dm_ic_enc_seq_len:]
                ext_input = ext_input[:, hps.dm_ic_enc_seq_len:]
                truth = truth[:, hps.dm_ic_enc_seq_len:, :]

                session_batch = SessionBatch(
                    encod_data=encod_data,
                    recon_data=recon_data,
                    ext_input=ext_input,
                    truth=truth,
                    sv_mask=sv_mask,
                )

                # --- EMG reconstruction target ---
                emg_recon = _to_tensor(emg_dict[f"{prefix}_recon_data"])
                assert emg_recon.shape[0] == n_samps, (
                    f"Trial count mismatch: spikes={n_samps}, "
                    f"EMG={emg_recon.shape[0]} for {prefix}"
                )

                if prefix == "train":
                    all_train_data.append((session_batch, emg_recon))
                else:
                    all_valid_data.append((session_batch, emg_recon))

        # Build datasets
        self.train_ds = [
            SOCSessionDataset(sb, emg) for sb, emg in all_train_data
        ]
        self.valid_ds = [
            SOCSessionDataset(sb, emg) for sb, emg in all_valid_data
        ]
        # Store raw data for later access
        self.train_data = all_train_data
        self.valid_data = all_valid_data

    def train_dataloader(self, shuffle=True):
        batch_size = int(self.hparams.batch_size / len(self.train_ds))
        dataloaders = {
            i: DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=True,
            )
            for i, ds in enumerate(self.train_ds)
        }
        return CombinedLoader(dataloaders, mode="max_size_cycle")

    def val_dataloader(self):
        batch_size = int(self.hparams.batch_size / len(self.valid_ds))
        dataloaders = {
            i: DataLoader(ds, batch_size=batch_size)
            for i, ds in enumerate(self.valid_ds)
        }
        return CombinedLoader(dataloaders, mode="max_size_cycle")

    def predict_dataloader(self):
        dataloaders = {
            s: {
                "train": DataLoader(
                    self.train_ds[s],
                    batch_size=self.hparams.batch_size,
                    shuffle=False,
                ),
                "valid": DataLoader(
                    self.valid_ds[s],
                    batch_size=self.hparams.batch_size,
                    shuffle=False,
                ),
            }
            for s in range(len(self.train_ds))
        }
        return dataloaders
