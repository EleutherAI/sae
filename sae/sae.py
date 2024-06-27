import json
from pathlib import Path
from typing import NamedTuple

import einops
import torch
from jaxtyping import Float, Int64
from safetensors.torch import load_model, save_model
from torch import nn, Tensor

from .config import SaeConfig
from .kernels import TritonDecoder


class ForwardOutput(NamedTuple):
    sae_out: Tensor

    latent_acts: Tensor
    """Activations of the top-k latents."""

    latent_indices: Tensor
    """Indices of the top-k features."""

    fvu: Tensor
    """Fraction of variance unexplained."""

    auxk_loss: Tensor
    """AuxK loss, if applicable."""


class Sae(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: SaeConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        d_sae = d_in * cfg.expansion_factor

        self.encoder = nn.Linear(d_in, d_sae, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        if self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    @staticmethod
    def load_from_disk(path: Path | str, device: str | torch.device = "cpu") -> "Sae":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SaeConfig(**cfg_dict)

        sae = Sae(d_in, cfg, device=device)
        load_model(sae, str(path / "sae.safetensors"), device=str(device))
        return sae

    def save_to_disk(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump({
                **self.cfg.to_dict(),
                "d_in": self.d_in,
            }, f)

    @property
    def device(self):
        return self.b_dec.device

    @property
    def dtype(self):
        return self.b_dec.dtype

    def encode(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_sae"]:
        # Remove decoder bias as per Anthropic
        sae_in = x.to(self.dtype) - self.b_dec
        out = self.encoder(sae_in)

        return nn.functional.relu(out) if not self.cfg.signed else out

    def decode(
        self,
        top_acts: Float[Tensor, "... d_sae"],
        top_indices: Int64[Tensor, "..."],
    ) -> Float[Tensor, "... d_in"]:
        y = TritonDecoder.apply(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        return y + self.b_dec

    def forward(self, x: Tensor, dead_mask: Tensor | None = None) -> ForwardOutput:
        latent_acts = self.encode(x)

        if self.cfg.signed:
            _, top_indices = latent_acts.abs().topk(self.cfg.k, sorted=False)
            top_acts = latent_acts.gather(dim=-1, index=top_indices)
        else:
            top_acts, top_indices = latent_acts.topk(self.cfg.k, sorted=False)

        # Decode and compute residual
        sae_out = self.decode(top_acts, top_indices)
        e = sae_out - x

        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (x - x.mean(0)).pow(2).sum(0)

        # Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # Heuristic from Appendix B.1 in the paper
            k_aux = x.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = torch.where(dead_mask[None], latent_acts, -torch.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents
            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e).pow(2).sum(0)
            auxk_loss = scale * torch.mean(auxk_loss / total_variance)
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum(0)
        fvu = torch.mean(l2_loss / total_variance)

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
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
