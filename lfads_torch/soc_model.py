"""LFADS-SOC LightningModule.

Replaces the standard LFADS GRU generator with a biophysical SOC (Stability-
Optimized Circuit) cell. The encoder and controller GRU are reused from
lfads-torch; the generator, readout, and loss are replaced.

Architecture:
  Encoder (IC+CI BiGRUs) → SOCDecoder (controller GRU + SOCCell) → DualReadout
  Loss: MSE (neural) + MSE (EMG), no Poisson/bits-per-spike.

See implementation_plan.md for full tensor dimension mapping.
"""

from typing import Any, Dict, Literal, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from .metrics import ExpSmoothedMetric, r2_score
from .modules.encoder import Encoder
from .modules.l2 import compute_l2_penalty
from .modules.priors import Null
from .modules.readout import DualReadout, MultisessionDualReadout
from .modules.recurrent import SOCCell
from .modules.soc_decoder import SOCDecoder
from .tuples import SessionBatch, SessionOutput
from .utils import transpose_lists


class LFADS_SOC(pl.LightningModule):
    """LFADS with SOC generator replacing the standard GRU generator.

    Parameters
    ----------
    encod_data_dim : int
        Shared encoder input dimension (after per-session readin).
    encod_seq_len : int
        Length of the encoding sequence.
    recon_seq_len : int
        Length of the reconstruction sequence.
    ext_input_dim : int
        External input dimension (0 for SOC MVP).
    ic_enc_seq_len : int
        Timesteps used only for IC encoder (0 = use full sequence).
    ic_enc_dim : int
        IC encoder BiGRU hidden size (per direction).
    ci_enc_dim : int
        CI encoder BiGRU hidden size (per direction).
    ci_lag : int
        Lag applied to controller input.
    con_dim : int
        Controller GRU hidden state dimension.
    co_dim : int
        Controller output dimension.
    ic_dim : int
        Initial condition latent dimension.
    soc_N : int
        SOC population size (must be even for E/I split).
    soc_dt : float
        SOC Euler integration step size (ms).
    soc_tau : float
        SOC membrane time constant (ms). Starting value ~50ms, tunable via PBT.
    soc_r0 : float
        SOC baseline firing rate (Hz). Default 20.
    soc_rmax : float
        SOC maximum firing rate (Hz). Default 100.
    soc_W_path : str
        Path to pre-computed SOC weight matrix (.pt file).
    readin : nn.ModuleList
        Per-session linear readin layers.
    readout : nn.ModuleList
        Per-session DualReadout modules (MultisessionDualReadout).
    variational : bool
        Whether to sample from posterior distributions.
    co_prior : nn.Module
        Prior for controller outputs.
    ic_prior : nn.Module
        Prior for initial conditions.
    ic_post_var_min : float
        Minimum variance for IC posterior.
    cell_clip : float
        Gradient clipping value for GRU cells.
    dropout_rate : float
        Dropout probability.
    loss_scale : float
        Scaling factor for the total loss.
    recon_reduce_mean : bool
        Whether to reduce reconstruction loss by mean.
    lr_init, lr_stop, lr_decay, lr_patience : float/int
        Learning rate scheduler parameters.
    lr_adam_beta1, lr_adam_beta2, lr_adam_epsilon : float
        Adam optimizer parameters.
    weight_decay : float
        Weight decay for AdamW.
    l2_start_epoch, l2_increase_epoch : int
        L2 regularization ramping schedule.
    l2_ic_enc_scale, l2_ci_enc_scale : float
        L2 penalty scale for encoder recurrent weights.
    kl_start_epoch, kl_increase_epoch : int
        KL divergence ramping schedule.
    kl_ic_scale, kl_co_scale : float
        KL penalty scale for IC and CO posteriors.
    lr_scheduler : bool
        Whether to use a learning rate scheduler.
    """

    def __init__(
        self,
        encod_data_dim: int,
        encod_seq_len: int,
        recon_seq_len: int,
        ext_input_dim: int,
        ic_enc_seq_len: int,
        ic_enc_dim: int,
        ci_enc_dim: int,
        ci_lag: int,
        con_dim: int,
        co_dim: int,
        ic_dim: int,
        # SOC-specific
        soc_N: int,
        soc_dt: float,
        soc_tau: float,
        soc_r0: float,
        soc_rmax: float,
        soc_W_path: str,
        # Multisession layers (passed in, not constructed here)
        readin: nn.ModuleList,
        readout: nn.ModuleList,
        # Priors / posteriors
        variational: bool,
        co_prior: nn.Module,
        ic_prior: nn.Module,
        ic_post_var_min: float,
        # Misc
        cell_clip: float,
        dropout_rate: float,
        loss_scale: float,
        recon_reduce_mean: bool,
        # Learning rate
        lr_scheduler: bool,
        lr_init: float,
        lr_stop: float,
        lr_decay: float,
        lr_patience: int,
        lr_adam_beta1: float,
        lr_adam_beta2: float,
        lr_adam_epsilon: float,
        weight_decay: float,
        # Regularization
        l2_start_epoch: int,
        l2_increase_epoch: int,
        l2_ic_enc_scale: float,
        l2_ci_enc_scale: float,
        kl_start_epoch: int,
        kl_increase_epoch: int,
        kl_ic_scale: float,
        kl_co_scale: float,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["ic_prior", "co_prior", "readin", "readout"],
        )
        # Store co_prior on hparams so SOCDecoder can access it
        self.hparams.co_prior = co_prior

        # Validate readin/readout list lengths match
        assert len(readin) == len(readout), (
            f"readin ({len(readin)}) and readout ({len(readout)}) must have "
            f"same number of sessions."
        )

        # Validate non-variational models use null priors
        if not variational:
            assert isinstance(ic_prior, Null) and isinstance(co_prior, Null)

        # --- Per-session layers ---
        self.readin = readin
        self.readout = readout  # MultisessionDualReadout

        # --- Controller flag ---
        self.use_con = all([ci_enc_dim > 0, con_dim > 0, co_dim > 0])

        # --- Encoder (reused from lfads-torch, unchanged) ---
        self.encoder = Encoder(hparams=self.hparams)

        # --- SOC Decoder (replaces standard Decoder) ---
        soc_cell = SOCCell(
            N=soc_N, dt=soc_dt, tau=soc_tau, W_init_path=soc_W_path,
            r0=soc_r0, rmax=soc_rmax,
        )
        self.soc_decoder = SOCDecoder(hparams=self.hparams, soc_cell=soc_cell)

        # --- Priors ---
        self.ic_prior = ic_prior
        self.co_prior = co_prior

        # --- Validation smoothing metric ---
        self.valid_recon_smth = ExpSmoothedMetric(coef=0.3)

    def forward(
        self,
        batch: Dict[int, SessionBatch],
        emg_targets: Dict[int, torch.Tensor] = None,
        sample_posteriors: bool = False,
    ) -> Dict[int, dict]:
        """Forward pass through the SOC-LFADS model.

        Parameters
        ----------
        batch : Dict[int, SessionBatch]
            Per-session neural data batches.
        emg_targets : Dict[int, Tensor], optional
            Per-session EMG reconstruction targets (not used in forward,
            but stored for convenience in evaluation).
        sample_posteriors : bool
            Whether to sample from posterior distributions.

        Returns
        -------
        Dict[int, dict] with per-session outputs containing:
            "neural_pred", "emg_pred", "rates", "gen_init",
            "ic_mean", "ic_std", "co_means", "co_stds",
            "gen_inputs", "con_states"
        """
        # Allow single-session input without dict wrapping
        if isinstance(batch, SessionBatch) and len(self.readin) == 1:
            batch = {0: batch}

        sessions = sorted(batch.keys())
        batch_sizes = [len(batch[s].encod_data) for s in sessions]

        # --- Pass through per-session readin ---
        encod_data = torch.cat(
            [self.readin[s](batch[s].encod_data) for s in sessions]
        )

        # --- Collect external inputs ---
        ext_input = torch.cat([batch[s].ext_input for s in sessions])

        # --- Encode ---
        ic_mean, ic_std, ci = self.encoder(encod_data)

        # --- IC posterior ---
        ic_post = self.ic_prior.make_posterior(ic_mean, ic_std)
        ic_samp = ic_post.rsample() if sample_posteriors else ic_mean

        # --- SOC Decode ---
        (
            gen_init,
            rates,
            con_states,
            co_means,
            co_stds,
            gen_inputs,
        ) = self.soc_decoder(
            ic_samp, ci, ext_input, sample_posteriors=sample_posteriors
        )

        # --- Split by session and apply per-session DualReadout ---
        rates_split = torch.split(rates, batch_sizes)
        output = {}
        for s, r_s in zip(sessions, rates_split):
            readout_result = self.readout[s](r_s)
            output[s] = {
                "neural_pred": readout_result["neural"],
                "emg_pred": readout_result["emg"],
                "rates": r_s,
            }

        # --- Attach shared latent variables (split by session) ---
        ic_mean_split = torch.split(ic_mean, batch_sizes)
        ic_std_split = torch.split(ic_std, batch_sizes)
        co_means_split = torch.split(co_means, batch_sizes)
        co_stds_split = torch.split(co_stds, batch_sizes)
        gen_init_split = torch.split(gen_init, batch_sizes)
        gen_inputs_split = torch.split(gen_inputs, batch_sizes)
        con_states_split = torch.split(con_states, batch_sizes)

        for s, idx in zip(sessions, range(len(sessions))):
            output[s].update(
                {
                    "gen_init": gen_init_split[idx],
                    "ic_mean": ic_mean_split[idx],
                    "ic_std": ic_std_split[idx],
                    "co_means": co_means_split[idx],
                    "co_stds": co_stds_split[idx],
                    "gen_inputs": gen_inputs_split[idx],
                    "con_states": con_states_split[idx],
                }
            )

        return output

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        hps = self.hparams
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=hps.lr_init,
            betas=(hps.lr_adam_beta1, hps.lr_adam_beta2),
            eps=hps.lr_adam_epsilon,
            weight_decay=hps.weight_decay,
        )
        if hps.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=hps.lr_decay,
                patience=hps.lr_patience,
                threshold=0.0,
                min_lr=hps.lr_stop,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "valid/recon_smth",
            }
        else:
            return optimizer

    def _compute_ramp(self, start: int, increase: int):
        ramp = (self.current_epoch + 1 - start) / (increase + 1)
        return torch.clamp(torch.tensor(ramp), 0, 1)

    def on_before_optimizer_step(
        self, optimizer: torch.optim.Optimizer, optimizer_idx: int
    ):
        hps = self.hparams
        l2_ramp = self._compute_ramp(hps.l2_start_epoch, hps.l2_increase_epoch)
        optimizer.param_groups[0]["weight_decay"] = l2_ramp * hps.weight_decay

    def _shared_step(
        self,
        batch: Dict[int, Tuple[SessionBatch, Tuple[torch.Tensor]]],
        batch_idx: int,
        split: Literal["train", "valid"],
    ) -> torch.Tensor:
        hps = self.hparams
        assert split in ["train", "valid"]

        sessions = sorted(batch.keys())

        # --- Unpack batch: (SessionBatch, (emg_recon_target,)) ---
        neural_batch = {s: b[0] for s, b in batch.items()}
        emg_targets = {s: b[1][0] for s, b in batch.items()}

        # --- Forward pass ---
        output = self.forward(
            neural_batch,
            emg_targets=emg_targets,
            sample_posteriors=hps.variational,
        )

        # --- Compute reconstruction losses (MSE, per session) ---
        sess_neural_loss = []
        sess_emg_loss = []
        batch_sizes = []

        for s in sessions:
            neural_pred = output[s]["neural_pred"]
            neural_target = neural_batch[s].recon_data
            emg_pred = output[s]["emg_pred"]
            emg_target = emg_targets[s]

            # MSE losses (element-wise then reduced)
            if hps.recon_reduce_mean:
                loss_neural = F.mse_loss(neural_pred, neural_target)
                loss_emg = F.mse_loss(emg_pred, emg_target)
            else:
                loss_neural = F.mse_loss(
                    neural_pred, neural_target, reduction="none"
                ).mean()
                loss_emg = F.mse_loss(
                    emg_pred, emg_target, reduction="none"
                ).mean()

            sess_neural_loss.append(loss_neural)
            sess_emg_loss.append(loss_emg)
            batch_sizes.append(len(neural_batch[s].encod_data))

        # Aggregate across sessions
        recon_neural = torch.mean(torch.stack(sess_neural_loss))
        recon_emg = torch.mean(torch.stack(sess_emg_loss))
        recon = recon_neural + recon_emg

        # --- L2 penalty on encoder recurrent weights ---
        l2 = self._compute_l2_penalty()

        # --- KL divergence ---
        ic_mean = torch.cat([output[s]["ic_mean"] for s in sessions])
        ic_std = torch.cat([output[s]["ic_std"] for s in sessions])
        co_means = torch.cat([output[s]["co_means"] for s in sessions])
        co_stds = torch.cat([output[s]["co_stds"] for s in sessions])

        ic_kl = self.ic_prior(ic_mean, ic_std) * hps.kl_ic_scale
        co_kl = self.co_prior(co_means, co_stds) * hps.kl_co_scale

        # --- Ramping coefficients ---
        l2_ramp = self._compute_ramp(hps.l2_start_epoch, hps.l2_increase_epoch)
        kl_ramp = self._compute_ramp(hps.kl_start_epoch, hps.kl_increase_epoch)

        # --- Total loss ---
        loss = hps.loss_scale * (recon + l2_ramp * l2 + kl_ramp * (ic_kl + co_kl))

        # --- R² scores for monitoring ---
        r2_neural_list = []
        r2_emg_list = []
        for s in sessions:
            neural_pred = output[s]["neural_pred"]
            neural_target = neural_batch[s].recon_data
            emg_pred = output[s]["emg_pred"]
            emg_target = emg_targets[s]
            r2_neural_list.append(r2_score(neural_pred, neural_target))
            r2_emg_list.append(r2_score(emg_pred, emg_target))

        r2_neural = torch.mean(torch.stack(r2_neural_list))
        r2_emg = torch.mean(torch.stack(r2_emg_list))

        # --- Logging ---
        total_batch_size = sum(batch_sizes)
        metrics = {
            f"{split}/loss": loss,
            f"{split}/recon": recon,
            f"{split}/recon_neural": recon_neural,
            f"{split}/recon_emg": recon_emg,
            f"{split}/r2_neural": r2_neural,
            f"{split}/r2_emg": r2_emg,
            f"{split}/wt_l2": l2,
            f"{split}/wt_l2/ramp": l2_ramp,
            f"{split}/wt_kl": ic_kl + co_kl,
            f"{split}/wt_kl/ic": ic_kl,
            f"{split}/wt_kl/co": co_kl,
            f"{split}/wt_kl/ramp": kl_ramp,
        }

        # Per-session logging
        for s, nl, el, bs in zip(
            sessions, sess_neural_loss, sess_emg_loss, batch_sizes
        ):
            self.log(f"{split}/recon_neural/sess{s}", nl, batch_size=bs)
            self.log(f"{split}/recon_emg/sess{s}", el, batch_size=bs)

        if split == "valid":
            self.valid_recon_smth.update(recon, total_batch_size)
            metrics.update(
                {
                    "valid/recon_smth": self.valid_recon_smth,
                    "hp_metric": recon,
                    "cur_epoch": float(self.current_epoch),
                }
            )

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            batch_size=total_batch_size,
        )

        return loss

    def _compute_l2_penalty(self) -> torch.Tensor:
        """Compute L2 penalty on encoder recurrent weights.

        Note: The SOC W matrix is a frozen buffer, so it has no L2 penalty.
        The SOC decoder's controller GRU weights could optionally be penalized,
        but we skip that for MVP.
        """
        hps = self.hparams
        recurrent_kernels_and_weights = [
            (self.encoder.ic_enc.fwd_gru.cell.weight_hh, hps.l2_ic_enc_scale),
            (self.encoder.ic_enc.bwd_gru.cell.weight_hh, hps.l2_ic_enc_scale),
        ]
        if self.use_con:
            recurrent_kernels_and_weights.extend(
                [
                    (
                        self.encoder.ci_enc.fwd_gru.cell.weight_hh,
                        hps.l2_ci_enc_scale,
                    ),
                    (
                        self.encoder.ci_enc.bwd_gru.cell.weight_hh,
                        hps.l2_ci_enc_scale,
                    ),
                ]
            )
        recurrent_penalty = 0.0
        recurrent_size = 0
        for kernel, weight in recurrent_kernels_and_weights:
            if weight > 0:
                recurrent_penalty += weight * 0.5 * torch.norm(kernel, 2) ** 2
                recurrent_size += kernel.numel()
        recurrent_penalty /= recurrent_size + 1e-8
        return recurrent_penalty

    def training_step(
        self,
        batch: Dict[int, Tuple[SessionBatch, Tuple[torch.Tensor]]],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(
        self,
        batch: Dict[int, Tuple[SessionBatch, Tuple[torch.Tensor]]],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "valid")

    def predict_step(
        self,
        batch: Dict[int, Tuple[SessionBatch, Tuple[torch.Tensor]]],
        batch_idx: int,
        sample_posteriors: bool = True,
    ) -> Dict[int, "SessionOutput"]:
        """Predict step returning SessionOutput for callback compatibility.

        Maps SOC outputs to the standard lfads-torch SessionOutput:
            output_params  ← neural_pred
            factors        ← rates (SOC firing rates)
            gen_states     ← rates (same as factors for SOC)
            gen_inputs     ← gen_inputs (controller output)
        """
        neural_batch = {s: b[0] for s, b in batch.items()}
        raw_output = self.forward(
            neural_batch,
            sample_posteriors=self.hparams.variational and sample_posteriors,
        )
        # Wrap in SessionOutput namedtuples for callback compatibility
        result = {}
        for s, out in raw_output.items():
            result[s] = SessionOutput(
                output_params=out["neural_pred"],
                factors=out["rates"],
                ic_mean=out["ic_mean"],
                ic_std=out["ic_std"],
                co_means=out["co_means"],
                co_stds=out["co_stds"],
                gen_states=out["rates"],     # SOC rates serve as gen_states
                gen_init=out["gen_init"],
                gen_inputs=out["gen_inputs"],
                con_states=out["con_states"],
            )
        return result

    def on_validation_epoch_end(self):
        self.log_dict(
            {
                "hp/lr_init": self.hparams.lr_init,
                "hp/dropout_rate": self.hparams.dropout_rate,
                "hp/l2_ic_enc_scale": self.hparams.l2_ic_enc_scale,
                "hp/l2_ci_enc_scale": self.hparams.l2_ci_enc_scale,
                "hp/kl_co_scale": self.hparams.kl_co_scale,
                "hp/kl_ic_scale": self.hparams.kl_ic_scale,
                "hp/weight_decay": self.hparams.weight_decay,
            }
        )
