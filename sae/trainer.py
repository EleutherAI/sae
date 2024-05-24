import contextlib
from dataclasses import dataclass
from typing import Any, Sized, cast

import torch
import wandb
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from . import __version__
from .checkpointing import save_checkpoint
from .config import SaeConfig
from .optim import L1Scheduler, get_lr_scheduler
from .sae import SparseAutoencoder
from .utils import assert_type

# used to map between parameters which are updated during finetuning and the config str.
FINETUNING_PARAMETERS = {
    "scale": ["scaling_factor"],
    "decoder": ["scaling_factor", "W_dec", "b_dec"],
    "unrotated_decoder": ["scaling_factor", "b_dec"],
}


def _log_feature_sparsity(
    feature_sparsity: Tensor, eps: float = 1e-10
) -> Tensor:
    return torch.log10(feature_sparsity + eps).detach().cpu()


@dataclass
class TrainSaeOutput:
    sae: SparseAutoencoder
    checkpoint_path: str
    log_feature_sparsities: Tensor


@dataclass
class TrainStepOutput:
    sae_in: Tensor
    sae_out: Tensor
    feature_acts: Tensor
    loss: Tensor
    mse_loss: Tensor
    l1_loss: Tensor
    ghost_grad_loss: Tensor
    ghost_grad_neuron_mask: Tensor


class SaeTrainer:
    def __init__(self, cfg: SaeConfig, dataset: Dataset, model: PreTrainedModel):
        N = model.config.num_hidden_layers

        self.cfg = cfg
        self.dataset = dataset

        assert isinstance(dataset, Sized)
        self.num_examples = len(dataset)

        device = model.device
        self.model = model
        self.saes = nn.ModuleList([SparseAutoencoder(cfg, device) for _ in range(N)])

        self.n_training_steps: int = 0
        self.n_training_tokens: int = 0
        self.started_fine_tuning: bool = False

        self.cfg.sae_lens_training_version = __version__

        self.act_freq_scores = torch.zeros(
            cast(int, cfg.d_sae),
            device=device,
        )
        self.n_forward_passes_since_fired = torch.zeros(
            cast(int, cfg.d_sae),
            device=device,
        )
        self.n_frac_active_tokens = 0
        # we don't train the scaling factor (initially)
        # set requires grad to false for the scaling factor
        for name, param in self.saes.named_parameters():
            if "scaling_factor" in name:
                param.requires_grad = False

        self.optimizer = Adam(
            self.saes.parameters(),
            lr=cfg.lr,
            betas=(
                cfg.adam_beta1,  # type: ignore
                cfg.adam_beta2,  # type: ignore
            ),
        )
        assert cfg.lr_end is not None  # this is set in config post-init
        self.lr_scheduler = get_lr_scheduler(
            cfg.lr_scheduler_name,
            lr=cfg.lr,
            optimizer=self.optimizer,
            warm_up_steps=cfg.lr_warm_up_steps,
            decay_steps=cfg.lr_decay_steps,
            training_steps=self.num_examples // cfg.batch_size,
            lr_end=cfg.lr_end,
            num_cycles=cfg.n_restart_cycles,
        )

        self.l1_scheduler = L1Scheduler(
            l1_warm_up_steps=cfg.l1_warm_up_steps,  # type: ignore
            total_steps=self.num_examples // cfg.batch_size,
            final_l1_coefficient=self.cfg.l1_coefficient,
        )

    @property
    def feature_sparsity(self) -> Tensor:
        return self.act_freq_scores / self.n_frac_active_tokens

    @property
    def log_feature_sparsity(self) -> Tensor:
        return _log_feature_sparsity(self.feature_sparsity)

    @property
    def current_l1_coefficient(self) -> float:
        return self.l1_scheduler.current_l1_coefficient

    def fit(self):
        wandb.init(name=self.cfg.run_name)

        device = self.model.device
        dl = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )
        pbar = tqdm(dl, desc="Training")

        for batch in pbar:
            # Forward pass on the model to get the next batch of activations
            with torch.no_grad():
                hidden_list = self.model(
                    batch["input_ids"].to(device),
                    output_hidden_states=True
                ).hidden_states[1:]

            for i, (hiddens, sae) in enumerate(zip(hidden_list, self.saes)):
                step_output = self._train_step(
                    assert_type(SparseAutoencoder, sae),
                    hiddens.flatten(0, 1)
                )

                if self.cfg.log_to_wandb:
                    with torch.no_grad():
                        if (self.n_training_steps + 1) % self.cfg.wandb_log_frequency == 0:
                            wandb.log(
                                self._build_train_step_log_dict(
                                    output=step_output,
                                    n_training_tokens=self.n_training_tokens,
                                    layer=i,
                                ),
                                step=self.n_training_steps,
                            )

                        # record loss frequently, but not all the time.
                        if (self.n_training_steps + 1) % (
                            self.cfg.wandb_log_frequency * self.cfg.eval_every_n_wandb_logs
                        ) == 0:
                            print("TODO: Implement eval")
                            # self.sae.eval()
                            # run_evals(
                            #     sparse_autoencoder=self.sae,
                            #     activation_store=self.activation_store,
                            #     model=self.model,
                            #     n_training_steps=self.n_training_steps,
                            #     suffix="",
                            #     n_eval_batches=self.cfg.n_eval_batches,
                            #     eval_batch_size_prompts=self.cfg.eval_batch_size_prompts,
                            # )
                            # self.sae.train()

            self.optimizer.step()
            if self.cfg.normalize_sae_decoder:
                for sae in self.saes:
                    sae.remove_gradient_parallel_to_decoder_directions()

            self.optimizer.zero_grad()
            self.lr_scheduler.step()
            self.l1_scheduler.step()

            # checkpoint if at checkpoint frequency
            # if (
            #     self.checkpoint_thresholds
            #     and self.n_training_tokens > self.checkpoint_thresholds[0]
            # ):
            #     save_checkpoint(
            #         trainer=self,
            #         checkpoint_name=self.n_training_tokens,
            #     )

            ###############
            self.n_training_steps += 1

        # save final sae group to checkpoints folder
        save_checkpoint(
            trainer=self,
            checkpoint_name=f"final_{self.n_training_tokens}",
            wandb_aliases=["final_model"],
        )
        pbar.close()

    def _train_step(
        self,
        sparse_autoencoder: SparseAutoencoder,
        sae_in: Tensor,
    ) -> TrainStepOutput:

        sparse_autoencoder.train()
        # Make sure the W_dec is still zero-norm
        if sparse_autoencoder.normalize_sae_decoder:
            sparse_autoencoder.set_decoder_norm_to_unit_norm()

        # log and then reset the feature sparsity every feature_sampling_window steps
        if (self.n_training_steps + 1) % self.cfg.feature_sampling_window == 0:
            feature_sparsity = self.feature_sparsity
            log_feature_sparsity = _log_feature_sparsity(feature_sparsity)

            if self.cfg.log_to_wandb:
                wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())
                wandb.log(
                    {
                        "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
                        "plots/feature_density_line_chart": wandb_histogram,
                        "sparsity/below_1e-5": (feature_sparsity < 1e-5).sum().item(),
                        "sparsity/below_1e-6": (feature_sparsity < 1e-6).sum().item(),
                    },
                    step=self.n_training_steps,
                )

            self.act_freq_scores = torch.zeros(
                sparse_autoencoder.cfg.d_sae,  # type: ignore
                device=sparse_autoencoder.device,
            )
            self.n_frac_active_tokens = 0

        ghost_grad_neuron_mask = (
            self.n_forward_passes_since_fired
            > sparse_autoencoder.cfg.dead_feature_window
        ).bool()

        # Setup autocast if using

        if self.cfg.autocast:
            autocast_if_enabled = torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.cfg.autocast,
            )
        else:
            autocast_if_enabled = contextlib.nullcontext()

        # temporary hack until we move this out of the SAE.
        sparse_autoencoder.l1_coefficient = self.l1_scheduler.current_l1_coefficient
        # Forward and Backward Passes
        # for documentation on autocasting see:
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        with autocast_if_enabled:
            (
                sae_out,
                feature_acts,
                loss,
                mse_loss,
                l1_loss,
                ghost_grad_loss,
            ) = sparse_autoencoder(
                sae_in.to(sparse_autoencoder.device),
                ghost_grad_neuron_mask,
            )

        did_fire = (feature_acts > 0).float().sum(-2) > 0
        self.n_forward_passes_since_fired += 1
        self.n_forward_passes_since_fired[did_fire] = 0

        with torch.no_grad():
            # Calculate the sparsities, and add it to a list, calculate sparsity metrics
            self.act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
            self.n_frac_active_tokens += self.num_examples // self.cfg.batch_size

        loss.backward()

        # TODO: Work out if grad norm clipping should be in config / how to test it.
        torch.nn.utils.clip_grad_norm_(sparse_autoencoder.parameters(), 1.0)

        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            loss=loss,
            mse_loss=mse_loss,
            l1_loss=l1_loss,
            ghost_grad_loss=ghost_grad_loss,
            ghost_grad_neuron_mask=ghost_grad_neuron_mask,
        )

    def _build_train_step_log_dict(
        self,
        output: TrainStepOutput,
        n_training_tokens: int,
        layer: int,
    ) -> dict[str, Any]:
        sae_in = output.sae_in
        sae_out = output.sae_out
        feature_acts = output.feature_acts
        mse_loss = output.mse_loss
        l1_loss = output.l1_loss
        ghost_grad_loss = output.ghost_grad_loss
        loss = output.loss
        ghost_grad_neuron_mask = output.ghost_grad_neuron_mask

        # metrics for currents acts
        l0 = (feature_acts > 0).float().sum(-1).mean()
        current_learning_rate = self.optimizer.param_groups[0]["lr"]

        per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
        total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
        explained_variance = 1 - per_token_l2_loss / total_variance

        if isinstance(ghost_grad_loss, Tensor):
            ghost_grad_loss = ghost_grad_loss.item()
        return {
            # losses
            f"mse_loss/layer_{layer}": mse_loss.item(),
            f"l1_loss/layer_{layer}": l1_loss.item()
            / self.current_l1_coefficient,  # normalize by l1 coefficient
            f"ghost_grad_loss/layer_{layer}": ghost_grad_loss,
            f"overall_loss/layer_{layer}": loss.item(),
            # variance explained
            f"explained_variance/layer_{layer}": explained_variance.mean().item(),
            f"explained_variance_std/layer_{layer}": explained_variance.std().item(),
            f"l0/layer_{layer}": l0.item(),
            # sparsity
            # "sparsity/mean_passes_since_fired": self.n_forward_passes_since_fired.mean().item(),
            # "sparsity/dead_features": ghost_grad_neuron_mask.sum().item(),
            # "details/current_learning_rate": current_learning_rate,
            # "details/current_l1_coefficient": self.current_l1_coefficient,
            # "details/n_training_tokens": n_training_tokens,
        }
