from dataclasses import asdict
from typing import Sized

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, PreTrainedModel

from . import __version__
from .config import TrainConfig
from .sae import Sae
from .utils import geometric_median, maybe_all_cat, maybe_all_reduce


class SaeTrainer:
    def __init__(self, cfg: TrainConfig, dataset: Dataset, model: PreTrainedModel):
        d_in = model.config.hidden_size
        N = model.config.num_hidden_layers

        self.cfg = cfg
        self.dataset = dataset

        assert isinstance(dataset, Sized)
        num_examples = len(dataset)

        device = model.device
        self.model = model
        self.saes = nn.ModuleList([Sae(d_in, cfg.sae, device) for _ in range(N)])

        self.n_training_steps: int = 0
        self.n_training_tokens: int = 0

        d = d_in * cfg.sae.expansion_factor
        self.num_tokens_since_fired = torch.zeros(N, d, dtype=torch.long, device=device)

        # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
        if (lr := cfg.lr) is None:
            # Base LR is 2e-4 for num latents = 2 ** 14
            scale = d / (2 ** 14)

            lr = 2e-4 / scale ** 0.5
            print(f"Auto-selected LR: {lr:.2e}")

        try:
            from bitsandbytes.optim import Adam8bit as Adam

            print("Using 8-bit Adam from bitsandbytes")
        except ImportError:
            from torch.optim import Adam

            print("bitsandbytes 8-bit Adam not available, using torch.optim.Adam")
            print("Run `pip install bitsandbytes` for less memory usage.")

        self.optimizer = Adam(self.saes.parameters(), lr=lr)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, cfg.lr_warmup_steps, num_examples // cfg.batch_size
        )

    def fit(self):
        # Use Tensor Cores even for fp32 matmuls
        torch.set_float32_matmul_precision("high")

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        if self.cfg.log_to_wandb and rank_zero:
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
        pbar = tqdm(dl, desc="Training", disable=not rank_zero)

        # This mask is zeroed out every training step
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        num_tokens_in_step = 0

        for i, batch in enumerate(pbar):
            # Bookkeeping for dead feature detection
            num_tokens_in_step += batch["input_ids"].numel()

            # Forward pass on the model to get the next batch of activations
            with torch.no_grad():
                hidden_list = self.model(
                    batch["input_ids"].to(device), output_hidden_states=True
                ).hidden_states[:-1]

            # 'raw' never has a DDP wrapper
            for j, (hiddens, raw) in enumerate(zip(hidden_list, self.saes)):
                hiddens = hiddens.flatten(0, 1)

                # On the first iteration, initialize the decoder bias
                if i == 0:
                    # NOTE: The all-cat here could conceivably cause an OOM in some
                    # cases, but it's unlikely to be a problem with small world sizes.
                    # We could avoid this by "approximating" the geometric median
                    # across all ranks with the mean (median?) of the geometric medians
                    # on each rank. Not clear if that would hurt performance.
                    raw.b_dec.data = geometric_median(maybe_all_cat(hiddens))

                    # Wrap the SAEs with Distributed Data Parallel. We have to do this
                    # after we set the decoder bias, otherwise DDP will not register
                    # gradients flowing to the bias after the first step.
                    maybe_wrapped = [
                        DDP(sae, device_ids=[dist.get_rank()])
                        for sae in self.saes
                    ] if dist.is_initialized() else self.saes

                # Make sure the W_dec is still unit-norm
                if raw.cfg.normalize_decoder:
                    raw.set_decoder_norm_to_unit_norm()

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=torch.cuda.is_bf16_supported(),
                ):
                    wrapped = maybe_wrapped[j]

                    dead_mask = self.num_tokens_since_fired[j] > self.cfg.dead_feature_threshold
                    out = wrapped(
                        hiddens.to(wrapped.device),
                        dead_mask=dead_mask if self.cfg.auxk_alpha > 0 else None,
                    )

                loss = out.fvu + self.cfg.auxk_alpha * out.auxk_loss
                loss.div(self.cfg.grad_acc_steps).backward()

                # Update the did_fire mask
                did_fire[j][out.latent_indices.flatten()] = True
                maybe_all_reduce(did_fire[j], "max")    # max is boolean "any"

                if (
                    self.cfg.log_to_wandb
                    and (self.n_training_steps + 1) % self.cfg.wandb_log_frequency == 0
                ):
                    info = {f"fvu/layer_{j}": maybe_all_reduce(out.fvu).item()}
                    if self.cfg.auxk_alpha > 0:
                        info[f"auxk/layer_{j}"] = maybe_all_reduce(out.auxk_loss).item()

                    if rank_zero:
                        wandb.log(
                            info,
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

            if rank_zero and (self.n_training_steps + 1) % self.cfg.save_every == 0:
                self.save()

        self.save()
        pbar.close()
    
    def save(self):
        """Save the SAEs to disk."""
        for i, sae in enumerate(self.saes):
            assert isinstance(sae, Sae)

            # TODO: Make the path configurable
            sae.save_to_disk(f"checkpoints/layer_{i}.pt")
