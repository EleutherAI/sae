import contextlib
from dataclasses import dataclass
from typing import Any, Sized, cast

import bitsandbytes as bnb
import torch
import wandb
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, PreTrainedModel

from . import __version__
from .checkpointing import save_checkpoint
from .config import SaeConfig
from .optim import L1Scheduler
from .sae import SparseAutoencoder
from .utils import assert_type


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
    fvu: Tensor
    sparsity_loss: Tensor


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
        # we don't train the scaling factor (initially)
        # set requires grad to false for the scaling factor
        for name, param in self.saes.named_parameters():
            if "scaling_factor" in name:
                param.requires_grad = False

        self.optimizer = bnb.optim.Adam8bit(self.saes.parameters(), lr=cfg.lr)

        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, cfg.lr_warm_up_steps, self.num_examples // cfg.batch_size
        )
        self.l1_scheduler = L1Scheduler(
            l1_warm_up_steps=cfg.l1_warm_up_steps,  # type: ignore
            total_steps=self.num_examples // cfg.batch_size,
            final_sparsity_weight=self.cfg.sparsity_weight,
        )

    @property
    def current_sparsity_weight(self) -> float:
        return self.l1_scheduler.current_sparsity_weight

    def fit(self):
        wandb.init(name=self.cfg.run_name)

        num_sae_params = sum(p.numel() for p in self.saes.parameters())
        num_model_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of SAE parameters: {num_sae_params:_}")
        print(f"Number of model parameters: {num_model_params:_}")

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
                    batch["input_ids"].to(device), output_hidden_states=True
                ).hidden_states[1:]

            for i, (hiddens, sae) in enumerate(zip(hidden_list, self.saes)):
                hiddens = hiddens.flatten(0, 1)

                step_output = self._train_step(
                    assert_type(SparseAutoencoder, sae), hiddens
                )

                if (
                    self.cfg.log_to_wandb
                    and (self.n_training_steps + 1) % self.cfg.wandb_log_frequency == 0
                ):
                    wandb.log(
                        self._build_train_step_log_dict(
                            output=step_output,
                            layer=i,
                        ),
                        step=self.n_training_steps,
                    )

            self.optimizer.step()
            if self.cfg.normalize_sae_decoder:
                for sae in self.saes:
                    sae.remove_gradient_parallel_to_decoder_directions()

            self.optimizer.zero_grad()
            self.lr_scheduler.step()
            self.l1_scheduler.step()

            ###############
            self.n_training_steps += 1

        # save final sae group to checkpoints folder
        save_checkpoint(
            trainer=self,
            checkpoint_name=f"final_{self.n_training_tokens}",
        )
        pbar.close()

    def _train_step(
        self,
        sparse_autoencoder: SparseAutoencoder,
        sae_in: Tensor,
    ) -> TrainStepOutput:
        sparse_autoencoder.train()

        # Make sure the W_dec is still unit-norm
        if sparse_autoencoder.normalize_sae_decoder:
            sparse_autoencoder.set_decoder_norm_to_unit_norm()

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
        sparse_autoencoder.sparsity_weight = self.l1_scheduler.current_sparsity_weight
        # Forward and Backward Passes
        # for documentation on autocasting see:
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        with autocast_if_enabled:
            (
                sae_out,
                feature_acts,
                loss,
                fvu,
                sparsity_loss,
            ) = sparse_autoencoder(
                sae_in.to(sparse_autoencoder.device),
            )

        did_fire = feature_acts.gt(0).float().sum(-2) > 0
        self.n_forward_passes_since_fired += 1
        self.n_forward_passes_since_fired[did_fire] = 0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(sparse_autoencoder.parameters(), 1.0)

        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            loss=loss,
            fvu=fvu,
            sparsity_loss=sparsity_loss,
        )

    @torch.no_grad()
    def _build_train_step_log_dict(
        self,
        output: TrainStepOutput,
        layer: int,
    ) -> dict[str, Any]:
        return {
            # losses
            f"fvu/layer_{layer}": output.fvu.item(),
            f"sparsity_loss/layer_{layer}": output.sparsity_loss.item()
            / self.current_sparsity_weight,  # normalize by l1 coefficient
            f"overall_loss/layer_{layer}": output.loss.item(),

            f"l0/layer_{layer}": output.feature_acts.gt(0).float().sum(-1).mean().item(),
        }
