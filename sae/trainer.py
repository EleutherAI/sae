from dataclasses import asdict, dataclass
from typing import Sized

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, PreTrainedModel

from . import __version__
from .config import SaeConfig
from .sae import Sae
from .utils import assert_type, geometric_median


@dataclass
class TrainStepOutput:
    sae_in: Tensor
    sae_out: Tensor
    feature_acts: Tensor
    fvu: Tensor


class SaeTrainer:
    def __init__(self, cfg: SaeConfig, dataset: Dataset, model: PreTrainedModel):
        N = model.config.num_hidden_layers

        self.cfg = cfg
        self.dataset = dataset

        assert isinstance(dataset, Sized)
        self.num_examples = len(dataset)

        device = model.device
        self.model = model
        self.saes = nn.ModuleList([Sae(cfg, device) for _ in range(N)])

        self.n_training_steps: int = 0
        self.n_training_tokens: int = 0

        d = assert_type(int, cfg.d_sae)
        self.num_tokens_since_fired = torch.zeros(
            N, d, dtype=torch.long, device=device
        )

        # The most performant configuration (with good quality output) is 8-bit Adam
        # while optimizing the SAE entirely in bf16. Regular Adam does not work well
        # with bf16 weights and gradients.
        # try:
        #     from bitsandbytes.optim import Adam8bit as Adam
        #     print("Using 8-bit Adam from bitsandbytes")
# 
        #     if torch.cuda.is_bf16_supported():
        #         print("Using bfloat16 for training")
        #     else:
        #         print("bfloat16 not supported on this device, using float32")
        # except ImportError:
        #     from torch.optim import Adam
        #     print("bitsandbytes 8-bit Adam not available, using torch.optim.Adam")
        #     print("Run `pip install bitsandbytes` for less memory usage.")

        #from torch.optim import Adam
        from bitsandbytes.optim import Adam8bit as Adam
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

                step_output = self._train_step(
                    assert_type(Sae, sae), hiddens
                )
                nonzero_mask = step_output.feature_acts.gt(0).detach()

                # Update the did_fire mask
                did_fire[i] |= nonzero_mask.any(dim=0)

                if (
                    self.cfg.log_to_wandb
                    and (self.n_training_steps + 1) % self.cfg.wandb_log_frequency == 0
                ):
                    wandb.log(
                        {f"fvu/layer_{i}": step_output.fvu.item()},
                        step=self.n_training_steps,
                    )

            # Check if we need to actually do a training step
            if pbar.n % self.cfg.grad_acc_steps == 0:
                if self.cfg.normalize_decoder:
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

                    dead_pct = 1.0 - float(did_fire.mean(dtype=torch.float32))
                    pbar.set_postfix_str(f"{dead_pct:.1%} dead")

                    did_fire.zero_()    # reset the mask
                    num_tokens_in_step = 0

        # save final sae group to checkpoints folder
        for i, sae in enumerate(self.saes):
            assert isinstance(sae, Sae)
            sae.save_to_disk(f"checkpoints/layer_{i}.pt")

        pbar.close()

    def _train_step(
        self,
        sae: Sae,
        sae_in: Tensor,
    ) -> TrainStepOutput:
        sae.train()

        # Make sure the W_dec is still unit-norm
        if sae.cfg.normalize_decoder:
            sae.set_decoder_norm_to_unit_norm()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.cfg.autocast,
        ):
            (
                sae_out,
                feature_acts,
                fvu,
            ) = sae(
                sae_in.to(sae.device),
            )

        denom = sae.cfg.grad_acc_steps
        fvu.div(denom).backward()

        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            fvu=fvu,
        )
