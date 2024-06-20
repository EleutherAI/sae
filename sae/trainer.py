import os

import torch
import torch.distributed as dist
from datasets import Dataset
from safetensors.torch import load_model
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, PreTrainedModel

from .config import TrainConfig
from .sae import Sae
from .utils import ceil_div, geometric_median


class SaeTrainer:
    def __init__(self, cfg: TrainConfig, dataset: Dataset, model: PreTrainedModel):
        d_in = model.config.hidden_size

        # If no layers are specified, train on all of them
        if not cfg.layers:
            N = model.config.num_hidden_layers
            cfg.layers = list(range(0, N, cfg.layer_stride))
        
        self.cfg = cfg
        self.dataset = dataset.shuffle(seed=cfg.shuffle_seed)
        self.distribute_layers()

        N = len(self.layers)

        device = model.device
        self.model = model
        self.saes = nn.ModuleList([Sae(d_in, cfg.sae, device) for _ in range(N)])

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

        self.batch_idx = 0
        self.optimizer = Adam(self.saes.parameters(), lr=lr)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, cfg.lr_warmup_steps, len(dataset) // cfg.batch_size
        )

    @classmethod
    def resume_from(cls, path: str, dataset: Dataset, model: PreTrainedModel):
        cfg = TrainConfig.load_json(f"{path}/cfg.json")
        trainer = cls(cfg, dataset, model)

        rank = dist.get_rank() if dist.is_initialized() else 0
        state = torch.load(f"{path}/rank_{rank}.pt")
        trainer.batch_idx = state["batch_idx"]
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.lr_scheduler.load_state_dict(state["scheduler"])

        for layer, sae in zip(trainer.layers, trainer.saes):
            load_model(sae, f"{path}/layer_{layer}/sae.safetensors")

        print(f"Resuming from '{path}' at batch {trainer.batch_idx:_}")
        return trainer

    def fit(self):
        # Use Tensor Cores even for fp32 matmuls
        torch.set_float32_matmul_precision("high")

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        ddp = dist.is_initialized() and not self.cfg.distribute_layers

        if self.cfg.log_to_wandb and rank_zero:
            try:
                import wandb

                wandb.init(
                    name=self.cfg.run_name, config=self.cfg.to_dict(), save_code=True
                )
            except ImportError:
                print("Weights & Biases not installed, skipping logging.")
                self.cfg.log_to_wandb = False

        num_sae_params = sum(p.numel() for p in self.saes.parameters())
        num_model_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of SAE parameters: {num_sae_params:_}")
        print(f"Number of model parameters: {num_model_params:_}")

        bs = self.cfg.batch_size
        num_batches = ceil_div(len(self.dataset), bs)
        pbar = tqdm(
            desc="Training",
            disable=not rank_zero,
            initial=self.batch_idx,
            total=num_batches,
        )

        # This mask is zeroed out every training step
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        num_tokens_in_step = 0

        # For logging purposes
        device = self.model.device
        avg_auxk_loss = torch.zeros(len(self.saes), device=device)
        avg_fvu = torch.zeros(len(self.saes), device=device)

        while self.batch_idx < num_batches:
            # We manually slice into the dataset instead of using a DataLoader mainly
            # because it seemed easier to make it fully resumable this way.
            batch = self.dataset[self.batch_idx * bs : (self.batch_idx + 1) * bs]

            # Bookkeeping for dead feature detection
            num_tokens_in_step += batch["input_ids"].numel()

            # Forward pass on the model to get the next batch of activations
            with torch.no_grad():
                hidden_list = self.model(
                    batch["input_ids"].to(device), output_hidden_states=True
                ).hidden_states[:-1]

                if self.cfg.distribute_layers:
                    hidden_list = self.scatter_hiddens(hidden_list)
                else:
                    hidden_list = [hidden_list[i] for i in self.cfg.layers]

            # 'raw' never has a DDP wrapper
            for j, (hiddens, raw) in enumerate(zip(hidden_list, self.saes)):
                hiddens = hiddens.flatten(0, 1)

                # On the first iteration, initialize the decoder bias
                if self.batch_idx == 0:
                    # NOTE: The all-cat here could conceivably cause an OOM in some
                    # cases, but it's unlikely to be a problem with small world sizes.
                    # We could avoid this by "approximating" the geometric median
                    # across all ranks with the mean (median?) of the geometric medians
                    # on each rank. Not clear if that would hurt performance.
                    median = geometric_median(self.maybe_all_cat(hiddens))
                    raw.b_dec.data = median.to(raw.dtype)

                    # Wrap the SAEs with Distributed Data Parallel. We have to do this
                    # after we set the decoder bias, otherwise DDP will not register
                    # gradients flowing to the bias after the first step.
                    maybe_wrapped = [
                        DDP(sae, device_ids=[dist.get_rank()])
                        for sae in self.saes
                    ] if ddp else self.saes

                # Make sure the W_dec is still unit-norm
                if raw.cfg.normalize_decoder:
                    raw.set_decoder_norm_to_unit_norm()

                acc_steps = self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
                denom = acc_steps * self.cfg.wandb_log_frequency
                wrapped = maybe_wrapped[j]

                # Save memory by chunking the activations
                for chunk in hiddens.chunk(self.cfg.micro_acc_steps):
                    out = wrapped(
                        chunk,
                        dead_mask=(
                            self.num_tokens_since_fired[j] > self.cfg.dead_feature_threshold
                            if self.cfg.auxk_alpha > 0
                            else None
                        ),
                    )

                    avg_fvu[j] += self.maybe_all_reduce(out.fvu.detach()) / denom
                    if self.cfg.auxk_alpha > 0:
                        avg_auxk_loss[j] += self.maybe_all_reduce(out.auxk_loss.detach()) / denom

                    loss = out.fvu + self.cfg.auxk_alpha * out.auxk_loss
                    loss.div(acc_steps).backward()

                    # Update the did_fire mask
                    did_fire[j][out.latent_indices.flatten()] = True
                    self.maybe_all_reduce(did_fire[j], "max")    # max is boolean "any"

                # Clip gradient norm independently for each SAE
                torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)

            # Check if we need to actually do a training step
            step, substep = divmod(self.batch_idx + 1, self.cfg.grad_acc_steps)
            if substep == 0:
                if self.cfg.sae.normalize_decoder:
                    for sae in self.saes:
                        sae.remove_gradient_parallel_to_decoder_directions()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

                ###############
                with torch.no_grad():
                    self.num_tokens_since_fired += num_tokens_in_step
                    self.num_tokens_since_fired[did_fire] = 0

                    did_fire.zero_()  # reset the mask
                    num_tokens_in_step = 0

                if self.cfg.log_to_wandb and (step + 1) % self.cfg.wandb_log_frequency == 0:
                    info = {}

                    for j in range(len(self.saes)):
                        mask = self.num_tokens_since_fired[j] > self.cfg.dead_feature_threshold
                        layer_idx = self.layers[j]

                        info.update({
                            f"fvu/layer_{layer_idx}": avg_fvu[j].item(),
                            f"dead_pct/layer_{layer_idx}": mask.mean(dtype=torch.float32).item(),
                        })
                        if self.cfg.auxk_alpha > 0:
                            info[f"auxk/layer_{layer_idx}"] = avg_auxk_loss[j].item()

                    avg_auxk_loss.zero_()
                    avg_fvu.zero_()

                    if self.cfg.distribute_layers:
                        outputs = [{} for _ in range(dist.get_world_size())]
                        dist.gather_object(info, outputs if rank_zero else None)
                        info.update({k: v for out in outputs for k, v in out.items()})

                    if rank_zero:
                        wandb.log(info, step=step)

            if (step + 1) % self.cfg.save_every == 0 and substep == 0:
                if rank_zero:
                    pbar.write(f"Saving checkpoint...")

                self.save()

            self.batch_idx += 1
            pbar.update()

        self.save()
        pbar.close()
    
    def maybe_all_cat(self, x: Tensor) -> Tensor:
        """Concatenate a tensor across all processes."""
        if not dist.is_initialized() or self.cfg.distribute_layers:
            return x

        buffer = x.new_empty([dist.get_world_size() * x.shape[0], *x.shape[1:]])
        dist.all_gather_into_tensor(buffer, x)
        return buffer

    
    def maybe_all_reduce(self, x: Tensor, op: str = "mean") -> Tensor:
        if not dist.is_initialized() or self.cfg.distribute_layers:
            return x

        if op == "sum":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        elif op == "mean":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            x /= dist.get_world_size()
        elif op == "max":
            dist.all_reduce(x, op=dist.ReduceOp.MAX)
        else:
            raise ValueError(f"Unknown reduction op '{op}'")

        return x
    
    def distribute_layers(self):
        """Prepare a plan for distributing layers across ranks."""
        if not self.cfg.distribute_layers:
            self.layers = self.cfg.layers
            self.layer_plan = {}
            print(f"Training on layers: {self.cfg.layers}")
            return

        layers_per_rank, rem = divmod(len(self.cfg.layers), dist.get_world_size())
        assert rem == 0, "Number of layers must be divisible by world size"

        # Each rank gets a subset of the layers
        self.layer_plan = {
            rank: self.cfg.layers[start:start + layers_per_rank]
            for rank, start in enumerate(range(0, len(self.cfg.layers), layers_per_rank))
        }
        for rank, layers in self.layer_plan.items():
            print(f"Rank {rank} layers: {layers}")
        
        self.layers = self.layer_plan[dist.get_rank()]

    def scatter_hiddens(self, hidden_list: list[Tensor]) -> list[Tensor]:
        """Scatter & gather the hidden states across ranks."""
        outputs = [
            # Add a new leading "layer" dimension to each tensor
            torch.stack([hidden_list[i] for i in layers], dim=1)
            for layers in self.layer_plan.values()
        ]
        # Allocate one contiguous buffer to minimize memcpys
        buffer = outputs[0].new_empty(
            # The (micro)batch size times the world size
            hidden_list[0].shape[0] * dist.get_world_size(),
            # The number of layers we expect to receive
            len(self.layer_plan[dist.get_rank()]),
            # All other dimensions
            *hidden_list[0].shape[1:],
        )

        # Perform the all-to-all scatter
        inputs = buffer.split([len(output) for output in outputs])
        dist.all_to_all([x for x in inputs], outputs)

        # Return a list of results, one for each layerfi
        return list(buffer.unbind(1))

    def save(self):
        """Save the SAEs to disk."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        path = self.cfg.run_name or "checkpoints"
        os.makedirs(path, exist_ok=True)

        if rank == 0:
            self.cfg.save_json(f"{path}/cfg.json", indent=4)
        elif not self.cfg.distribute_layers:
            return

        state = {
            "batch_idx": self.batch_idx,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
        }
        torch.save(state, f"{path}/rank_{rank}.pt")

        for i, sae in zip(self.layers, self.saes):
            assert isinstance(sae, Sae)

            sae.save_to_disk(f"{path}/layer_{i}")
