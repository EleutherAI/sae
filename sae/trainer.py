from dataclasses import asdict
from typing import Sized

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, PreTrainedModel

from . import __version__
from .config import TrainConfig
from .sae import Sae
from .utils import geometric_median


class SaeTrainer:
    def __init__(self, cfg: TrainConfig, dataset: Dataset, model: PreTrainedModel):
        d_in = model.config.hidden_size
        N = model.config.num_hidden_layers

        self.cfg = cfg
        self.dataset = dataset

        assert isinstance(dataset, Sized)
        self.num_examples = len(dataset)

        device = model.device
        self.model = model
        self.saes = nn.ModuleList([Sae(d_in, cfg.sae, device) for _ in range(N)])

        self.n_training_steps: int = 0
        self.n_training_tokens: int = 0

        d = d_in * cfg.sae.expansion_factor
        self.num_tokens_since_fired = torch.zeros(N, d, dtype=torch.long, device=device)

        try:
            from bitsandbytes.optim import Adam8bit as Adam

            print("Using 8-bit Adam from bitsandbytes")
        except ImportError:
            from torch.optim import Adam

            print("bitsandbytes 8-bit Adam not available, using torch.optim.Adam")
            print("Run `pip install bitsandbytes` for less memory usage.")

        self.optimizer = Adam(self.saes.parameters(), lr=cfg.lr)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, cfg.lr_warm_up_steps, self.num_examples // cfg.batch_size
        )

    def fit(self):
        # Use Tensor Cores even for fp32 matmuls
        torch.set_float32_matmul_precision("high")

        if self.cfg.log_to_wandb:
            try:
                import wandb

                wandb.init(
                    name=self.cfg.run_name, config=asdict(self.cfg), save_code=True
                )
            except ImportError:
                print("Weights & Biases not installed, skipping logging.")

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
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        num_tokens_in_step = 0

        for batch in pbar:
            # Bookkeeping for dead feature detection
            num_tokens_in_step += batch["input_ids"].numel()

            # Forward pass on the model to get the next batch of activations
            with torch.no_grad():
                hidden_list = self.model(
                    batch["input_ids"].to(device), output_hidden_states=True
                ).hidden_states[:-1]

            for i, (hiddens, sae) in enumerate(zip(hidden_list, self.saes)):
                hiddens = hiddens.flatten(0, 1)

                # On the first iteration, initialize the decoder bias
                if pbar.n == 0:
                    sae.b_dec.data = geometric_median(hiddens)

                # Make sure the W_dec is still unit-norm
                if sae.cfg.normalize_decoder:
                    sae.set_decoder_norm_to_unit_norm()

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=torch.cuda.is_bf16_supported(),
                ):
                    out = sae(hiddens.to(sae.device))

                denom = self.cfg.grad_acc_steps
                out.fvu.div(denom).backward()

                # Update the did_fire mask
                fired_indices = out.latent_indices.unique()
                did_fire[i][fired_indices] = True

                if (
                    self.cfg.log_to_wandb
                    and (self.n_training_steps + 1) % self.cfg.wandb_log_frequency == 0
                ):
                    wandb.log(
                        {f"fvu/layer_{i}": out.fvu.item()},
                        step=self.n_training_steps,
                    )

            # Check if we need to actually do a training step
            if pbar.n % self.cfg.grad_acc_steps == 0:
                if self.cfg.sae.normalize_decoder:
                    for sae in self.saes:
                        sae.remove_gradient_parallel_to_decoder_directions()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

                ###############
                self.n_training_steps += 1

                with torch.no_grad():
                    self.num_tokens_since_fired += num_tokens_in_step
                    self.num_tokens_since_fired[did_fire] = 0

                    active_pct = float(did_fire.mean(dtype=torch.float32))
                    pbar.set_postfix_str(f"{active_pct:.1%} active")

                    did_fire.zero_()  # reset the mask
                    num_tokens_in_step = 0

            if self.n_training_steps % self.cfg.save_every == 0:
                self.save()

        self.save()
        pbar.close()
    
    def save(self):
        """Save the SAEs to disk."""
        for i, sae in enumerate(self.saes):
            assert isinstance(sae, Sae)

            # TODO: Make the path configurable
            sae.save_to_disk(f"checkpoints/layer_{i}.pt")
