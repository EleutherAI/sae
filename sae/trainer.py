from dataclasses import asdict, dataclass
from typing import Any, Sized

import torch
import wandb
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, PreTrainedModel

from . import __version__
from .checkpointing import save_checkpoint
from .config import SaeConfig
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

        d = assert_type(int, cfg.d_sae)
        self.n_fwd_passes_since_fired = torch.zeros(
            N, d, dtype=torch.long, device=device
        )

        self.optimizer = Adam(
            self.saes.parameters(), lr=cfg.lr, betas=(0.9, 0.95)
        )
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, cfg.lr_warm_up_steps, self.num_examples // cfg.batch_size
        )

    def fit(self):
        wandb.init(name=self.cfg.run_name, config=asdict(self.cfg), save_code=True)

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

        # This mask is zeroed out every training step
        did_fire = torch.zeros_like(self.n_fwd_passes_since_fired, dtype=torch.bool)

        fvu_avg = torch.zeros(len(self.saes), device=device)
        l0_avg = torch.zeros(len(self.saes), device=device)

        for batch in pbar:
            # Forward pass on the model to get the next batch of activations
            with torch.no_grad():
                hidden_list = self.model(
                    batch["input_ids"].to(device), output_hidden_states=True
                ).hidden_states[:-1]

            for i, (hiddens, sae) in enumerate(zip(hidden_list, self.saes)):
                hiddens = hiddens.flatten(0, 1)

                step_output = self._train_step(
                    assert_type(SparseAutoencoder, sae), hiddens
                )
                fvu_avg[i] += step_output.fvu / self.cfg.grad_acc_steps

                nonzero_mask = step_output.feature_acts.gt(0).detach()
                l0 = nonzero_mask.sum(dim=-1, dtype=l0_avg.dtype).mean()
                l0_avg[i] += l0 / self.cfg.grad_acc_steps

                # Update the did_fire mask
                did_fire[i] |= nonzero_mask.any(dim=0)

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

            # Check if we need to actually do a training step
            if pbar.n % self.cfg.grad_acc_steps == 0:
                # Update sparsity weights
                for fvu, l0, sae in zip(fvu_avg, l0_avg, self.saes):
                    # We get divergence if we don't do this
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)

                    prop_err = torch.clamp(l0 / self.cfg.target_l0 - 1, -0.2, 0.2)
                    sae.sparsity_weight *= 1 + 0.01 * prop_err

                if self.cfg.normalize_sae_decoder:
                    for sae in self.saes:
                        sae.remove_gradient_parallel_to_decoder_directions()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

                ###############
                self.n_training_steps += 1

                with torch.no_grad():
                    self.n_fwd_passes_since_fired += 1
                    self.n_fwd_passes_since_fired[did_fire] = 0

                    dead_pct = 1.0 - float(did_fire.mean(dtype=torch.float32))
                    pbar.set_postfix_str(f"{dead_pct:.1%} dead")

                    did_fire.zero_()    # reset the mask

                fvu_avg.zero_()
                l0_avg.zero_()
            
            # Check if we need to resample
            if self.n_training_steps % self.cfg.feature_sampling_window == 0:
                d = assert_type(int, self.cfg.d_sae)

                for i, sae in enumerate(self.saes):
                    thresh = self.cfg.feature_sampling_window // 2
                    resample_mask = self.n_fwd_passes_since_fired[i] >= thresh
                    num_to_resample = int(resample_mask.sum())
                    print(f"Resampling {num_to_resample / d:.1%} features for layer {i}")

                    if num_to_resample <= 0:
                        continue

                    # Get new parameters
                    sae.W_enc.data[:, resample_mask] = sae.W_enc.data.new_empty(
                        sae.d_in, num_to_resample
                    ).normal_(
                        std=sae.W_enc.data.std().item() * 0.2
                    )
                    sae.W_dec.data[resample_mask] = sae.W_dec.data.new_empty(
                        num_to_resample, sae.d_in
                    ).normal_(
                        std=sae.W_dec.data.std().item()
                    )
                    sae.b_enc.data[resample_mask] = 0.0

                    self.optimizer.state[sae.W_enc]['exp_avg'][:, resample_mask] = 0.0
                    self.optimizer.state[sae.W_enc]['exp_avg_sq'][:, resample_mask] = 1.0
                    #self.optimizer.state[sae.W_enc]['step'] = 0

                    self.optimizer.state[sae.W_dec]['exp_avg'][resample_mask] = 0.0
                    self.optimizer.state[sae.W_dec]['exp_avg_sq'][resample_mask] = 1.0
                    #self.optimizer.state[sae.W_dec]['step'] = 0
                
                self.n_fwd_passes_since_fired.zero_()

        # save final sae group to checkpoints folder
        save_checkpoint(
            trainer=self,
            checkpoint_name=f"final_{self.n_training_tokens}",
        )
        pbar.close()

    def _train_step(
        self,
        sae: SparseAutoencoder,
        sae_in: Tensor,
    ) -> TrainStepOutput:
        sae.train()

        # Make sure the W_dec is still unit-norm
        if sae.normalize_sae_decoder:
            sae.set_decoder_norm_to_unit_norm()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.cfg.autocast,
        ):
            (
                sae_out,
                feature_acts,
                loss,
                fvu,
                sparsity_loss,
            ) = sae(
                sae_in.to(sae.device),
            )

        denom = sae.cfg.grad_acc_steps
        loss.div(denom).backward()

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
        weight = self.saes[layer].sparsity_weight
        return {
            # losses
            f"fvu/layer_{layer}": output.fvu.item(),
            f"sparsity_loss/layer_{layer}": output.sparsity_loss.item()
            / weight,  # normalize by l1 coefficient
            f"overall_loss/layer_{layer}": output.loss.item(),
            f"sparsity_weight/layer_{layer}": weight,

            f"l0/layer_{layer}": output.feature_acts.gt(0).float().sum(-1).mean().item(),
        }
