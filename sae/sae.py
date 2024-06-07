import json
from pathlib import Path
from typing import NamedTuple

import einops
import torch
from jaxtyping import Float
from safetensors.torch import load_model, save_model
from torch import nn, Tensor

from .config import SaeConfig


class ForwardOutput(NamedTuple):
    sae_out: Tensor
    feature_acts: Tensor
    fvu: Tensor


class Sae(nn.Module):
    d_sae: int

    def __init__(
        self,
        cfg: SaeConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in

        assert cfg.d_sae is not None  # keep pyright happy
        self.d_sae = cfg.d_sae

        # no config changes encoder bias init for now.
        self.b_enc = nn.Parameter(
            torch.zeros(self.d_sae, dtype=dtype, device=device)
        )
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_in, self.d_sae, dtype=dtype, device=device),
                nonlinearity="relu",
            )
        )
        self.W_dec = nn.Parameter(self.W_enc.data.mT.clone())

        if self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        # methods which change b_dec as a function of the dataset are implemented after init.
        self.b_dec = nn.Parameter(
            torch.zeros(self.d_in, dtype=dtype, device=device)
        )
    
    @staticmethod
    def load_from_disk(path: Path | str, device: str | torch.device = "cpu") -> "Sae":
        path = Path(path)
    
        with open(path / "cfg.json", "r") as f:
            cfg = SaeConfig(**json.load(f))

        sae = Sae(cfg, device=device)
        load_model(sae, str(path / "sae.safetensors"), device=str(device))
        return sae
    
    def save_to_disk(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(self.cfg.to_dict(), f)

    @property
    def device(self):
        return self.b_enc.device
    
    @property
    def dtype(self):
        return self.b_enc.dtype

    def encode(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_sae"]:
        x = x.to(self.dtype)
        sae_in = x - (
            self.b_dec * self.cfg.apply_b_dec_to_input
        )  # Remove decoder bias as per Anthropic

        hidden_pre = (
            einops.einsum(
                sae_in,
                self.W_enc,
                "... d_in, d_in d_sae -> ... d_sae",
            )
            + self.b_enc
        )
        # ReLU just to make sure the features are non-negative
        feats = nn.functional.relu(hidden_pre)

        # TODO: Implement kernels to make this more efficient
        top_vals, top_indices = feats.topk(self.cfg.k, sorted=False)
        out = torch.zeros_like(feats).scatter_(dim=-1, index=top_indices, src=top_vals)

        return out

    def decode(
        self, feature_acts: Float[Tensor, "... d_sae"]
    ) -> Float[Tensor, "... d_in"]:
        """Decodes SAE feature activation tensor into a reconstructed input activation tensor."""
        sae_out = (
            einops.einsum(
                feature_acts,
                self.W_dec,
                "... d_sae, d_sae d_in -> ... d_in",
            )
            + self.b_dec
        )
        return sae_out

    def forward(self, x: Tensor) -> ForwardOutput:
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)

        per_token_l2_loss = (sae_out - x).pow(2).sum(0)
        total_variance = (x - x.mean(0)).pow(2).sum(0)
        fvu = torch.mean(per_token_l2_loss / total_variance)

        return ForwardOutput(
            sae_out=sae_out,
            feature_acts=feature_acts,
            fvu=fvu,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= (norm + eps)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        """
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
