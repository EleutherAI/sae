from collections import defaultdict
from dataclasses import asdict
from fnmatch import fnmatchcase
from typing import Sized

import torch
import torch.distributed as dist
from natsort import natsorted
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedModel, get_linear_schedule_with_warmup

from .config import TrainConfig
from .sae import Sae
from .utils import geometric_median, get_layer_list


class SaeTrainer:
    def __init__(self, cfg: TrainConfig, dataset: Dataset, model: PreTrainedModel):
        d_in = model.config.hidden_size

        if cfg.hookpoints:
            assert not cfg.layers, "Cannot specify both `hookpoints` and `layers`."

            # Replace wildcard patterns
            raw_hookpoints = []
            for name, _ in model.named_modules():
                if any(fnmatchcase(name, pat) for pat in cfg.hookpoints):
                    raw_hookpoints.append(name)

            # Natural sort to impose a consistent order
            cfg.hookpoints = natsorted(raw_hookpoints)
        else:
            # If no layers are specified, train on all of them
            if not cfg.layers:
                N = model.config.num_hidden_layers
                cfg.layers = list(range(0, N, cfg.layer_stride))

            # Now convert layers to hookpoints
            layers_name, _ = get_layer_list(model)
            cfg.hookpoints = [f"{layers_name}.{i}" for i in cfg.layers]

        self.cfg = cfg
        self.dataset = dataset
        self.distribute_modules()

        N = len(cfg.hookpoints)
        assert isinstance(dataset, Sized)
        num_examples = len(dataset)

        device = model.device
        self.model = model
        self.saes = {
            hook: Sae(d_in, cfg.sae, device) for hook in self.local_hookpoints()
        }
        self.num_latents = cfg.sae.num_latents or d_in * cfg.sae.expansion_factor

        # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
        if (lr := cfg.lr) is None:
            # Base LR is 1e-4 for num latents = 2 ** 13
            scale = self.num_latents / (2**14)

            lr = 2e-4 / scale**0.5
            print(f"Auto-selected LR: {lr:.2e}")

        try:
            from bitsandbytes.optim import Adam8bit as Adam

            print("Using 8-bit Adam from bitsandbytes")
        except ImportError:
            from torch.optim import Adam

            print("bitsandbytes 8-bit Adam not available, using torch.optim.Adam")
            print("Run `pip install bitsandbytes` for less memory usage.")

        self.optimizer = Adam(
            (p for s in self.saes.values() for p in s.parameters()), lr=lr
        )
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, cfg.lr_warmup_steps, num_examples // cfg.batch_size
        )

    def fit(self):
        # Use Tensor Cores even for fp32 matmuls
        torch.set_float32_matmul_precision("high")

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        ddp = dist.is_initialized() and not self.cfg.distribute_modules

        if self.cfg.log_to_wandb and rank_zero:
            try:
                import wandb

                wandb.init(
                    name=self.cfg.run_name,
                    project="sae",
                    config=asdict(self.cfg),
                    save_code=True,
                )
            except ImportError:
                print("Weights & Biases not installed, skipping logging.")
                self.cfg.log_to_wandb = False

        num_sae_params = sum(
            p.numel() for s in self.saes.values() for p in s.parameters()
        )
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

        did_fire = defaultdict(
            lambda: torch.zeros(self.num_latents, device=device, dtype=torch.bool)
        )
        num_tokens_since_fired = defaultdict(
            lambda: torch.zeros(self.num_latents, device=device, dtype=torch.long)
        )
        num_tokens_in_step = 0

        # For logging purposes
        avg_auxk_loss = defaultdict(float)
        avg_fvu = defaultdict(float)

        hidden_dict: dict[str, Tensor] = {}
        name_to_module = {
            name: self.model.get_submodule(name) for name in self.cfg.hookpoints
        }
        module_to_name = {v: k for k, v in name_to_module.items()}

        def hook(module: nn.Module, _, outputs):
            # Maybe unpack tuple outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            name = module_to_name[module]
            hidden_dict[name] = outputs.flatten(0, 1)

        for i, batch in enumerate(pbar):
            hidden_dict.clear()

            # Bookkeeping for dead feature detection
            num_tokens_in_step += batch["input_ids"].numel()

            # Forward pass on the model to get the next batch of activations
            with torch.no_grad():
                handles = [
                    mod.register_forward_hook(hook) for mod in name_to_module.values()
                ]
                try:
                    self.model(batch["input_ids"].to(device))
                finally:
                    for handle in handles:
                        handle.remove()

                if self.module_plan:
                    hidden_dict = self.scatter_hiddens(hidden_dict)

            for name, hiddens in hidden_dict.items():
                raw = self.saes[name]  # 'raw' never has a DDP wrapper

                # On the first iteration, initialize the decoder bias
                if i == 0:
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
                    maybe_wrapped = (
                        {
                            name: DDP(sae, device_ids=[dist.get_rank()])
                            for name, sae in self.saes.items()
                        }
                        if ddp
                        else self.saes
                    )

                # Make sure the W_dec is still unit-norm
                if raw.cfg.normalize_decoder:
                    raw.set_decoder_norm_to_unit_norm()

                acc_steps = self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
                denom = acc_steps * self.cfg.wandb_log_frequency
                wrapped = maybe_wrapped[name]

                # Save memory by chunking the activations
                for chunk in hiddens.chunk(self.cfg.micro_acc_steps):
                    out = wrapped(
                        chunk,
                        dead_mask=(
                            num_tokens_since_fired[name]
                            > self.cfg.dead_feature_threshold
                            if self.cfg.auxk_alpha > 0
                            else None
                        ),
                    )

                    avg_fvu[name] += float(
                        self.maybe_all_reduce(out.fvu.detach()) / denom
                    )
                    if self.cfg.auxk_alpha > 0:
                        avg_auxk_loss[name] += float(
                            self.maybe_all_reduce(out.auxk_loss.detach()) / denom
                        )

                    loss = out.fvu + self.cfg.auxk_alpha * out.auxk_loss
                    loss.div(acc_steps).backward()

                    # Update the did_fire mask
                    did_fire[name][out.latent_indices.flatten()] = True
                    self.maybe_all_reduce(did_fire[name], "max")  # max is boolean "any"

                # Clip gradient norm independently for each SAE
                torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)

            # Check if we need to actually do a training step
            step, substep = divmod(i + 1, self.cfg.grad_acc_steps)
            if substep == 0:
                if self.cfg.sae.normalize_decoder:
                    for sae in self.saes.values():
                        sae.remove_gradient_parallel_to_decoder_directions()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

                ###############
                with torch.no_grad():
                    # Update the dead feature mask
                    for name, counts in num_tokens_since_fired.items():
                        counts += num_tokens_in_step
                        counts[did_fire[name]] = 0

                    did_fire.clear()
                    num_tokens_in_step = 0

                if (
                    self.cfg.log_to_wandb
                    and (step + 1) % self.cfg.wandb_log_frequency == 0
                ):
                    info = {}

                    for name in self.saes:
                        mask = (
                            num_tokens_since_fired[name]
                            > self.cfg.dead_feature_threshold
                        )

                        info.update(
                            {
                                f"fvu/{name}": avg_fvu[name],
                                f"dead_pct/{name}": mask.mean(
                                    dtype=torch.float32
                                ).item(),
                            }
                        )
                        if self.cfg.auxk_alpha > 0:
                            info[f"auxk/{name}"] = avg_auxk_loss[name]

                    avg_auxk_loss.clear()
                    avg_fvu.clear()

                    if self.cfg.distribute_modules:
                        outputs = [{} for _ in range(dist.get_world_size())]
                        dist.gather_object(info, outputs if rank_zero else None)
                        info.update({k: v for out in outputs for k, v in out.items()})

                    if rank_zero:
                        wandb.log(info, step=step)

                if (step + 1) % self.cfg.save_every == 0:
                    self.save()

        self.save()
        pbar.close()

    def local_hookpoints(self) -> list[str]:
        return (
            self.module_plan[dist.get_rank()]
            if self.module_plan
            else self.cfg.hookpoints
        )

    def maybe_all_cat(self, x: Tensor) -> Tensor:
        """Concatenate a tensor across all processes."""
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        buffer = x.new_empty([dist.get_world_size() * x.shape[0], *x.shape[1:]])
        dist.all_gather_into_tensor(buffer, x)
        return buffer

    def maybe_all_reduce(self, x: Tensor, op: str = "mean") -> Tensor:
        if not dist.is_initialized() or self.cfg.distribute_modules:
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

    def distribute_modules(self):
        """Prepare a plan for distributing modules across ranks."""
        if not self.cfg.distribute_modules:
            self.module_plan = {}
            print(f"Training on modules: {self.cfg.hookpoints}")
            return

        layers_per_rank, rem = divmod(len(self.cfg.hookpoints), dist.get_world_size())
        assert rem == 0, "Number of modules must be divisible by world size"

        # Each rank gets a subset of the layers
        self.module_plan = {
            rank: self.cfg.hookpoints[start : start + layers_per_rank]
            for rank, start in enumerate(
                range(0, len(self.cfg.hookpoints), layers_per_rank)
            )
        }
        for rank, modules in self.module_plan.items():
            print(f"Rank {rank} modules: {modules}")

    def scatter_hiddens(self, hidden_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Scatter & gather the hidden states across ranks."""
        outputs = [
            # Add a new leading "layer" dimension to each tensor
            torch.stack([hidden_dict[hook] for hook in hookpoints], dim=1)
            for hookpoints in self.module_plan.values()
        ]
        local_hooks = self.module_plan[dist.get_rank()]
        shape = next(iter(hidden_dict.values())).shape

        # Allocate one contiguous buffer to minimize memcpys
        buffer = outputs[0].new_empty(
            # The (micro)batch size times the world size
            shape[0] * dist.get_world_size(),
            # The number of layers we expect to receive
            len(local_hooks),
            # All other dimensions
            *shape[1:],
        )

        # Perform the all-to-all scatter
        inputs = buffer.split([len(output) for output in outputs])
        dist.all_to_all([x for x in inputs], outputs)

        # Return a list of results, one for each layer
        return {hook: buffer[:, i] for i, hook in enumerate(local_hooks)}

    def save(self):
        """Save the SAEs to disk."""

        if (
            self.cfg.distribute_modules
            or not dist.is_initialized()
            or dist.get_rank() == 0
        ):
            print("Saving checkpoint")

            for hook, sae in zip(self.local_hookpoints(), self.saes):
                assert isinstance(sae, Sae)

                path = self.cfg.run_name or "checkpoints"
                sae.save_to_disk(f"{path}/{hook}")

        # Barrier to ensure all ranks have saved before continuing
        if dist.is_initialized():
            dist.barrier()
