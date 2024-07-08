import json
from fnmatch import fnmatch
from natsort import natsorted
from pathlib import Path
from typing import NamedTuple

import einops
import torch
from huggingface_hub import snapshot_download
from jaxtyping import Float, Int64
from safetensors.torch import load_model, save_model
from torch import nn, Tensor

from .config import SaeConfig
from .kernels import TritonDecoder


class EncoderOutput(NamedTuple):
    top_acts: Float[Tensor, "... d_sae"]
    """Activations of the top-k latents."""

    top_indices: Int64[Tensor, "..."]
    """Indices of the top-k features."""


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
        *,
        decoder: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        d_sae = d_in * cfg.expansion_factor

        self.encoder = nn.Linear(d_in, d_sae, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

        self.W_dec = (
            nn.Parameter(self.encoder.weight.data.clone())
            if decoder
            else None
        )
        if decoder and self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    @staticmethod
    def load_many_from_hub(
        name: str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
        pattern: str | None = None,
    ) -> dict[str, "Sae"]:
        """Load SAEs for multiple hookpoints on a single model and dataset."""
        pattern = pattern + "/*" if pattern is not None else None
        repo_path = Path(snapshot_download(name, allow_patterns=pattern))

        files = [
            f for f in repo_path.iterdir()
            if f.is_dir() and (pattern is None or fnmatch(f.name, pattern))
        ]
        return {
            f.stem: Sae.load_from_disk(f, device=device, decoder=decoder)
            for f in natsorted(files, key=lambda f: f.stem)
        }

    @staticmethod
    def load_from_hub(
        name: str,
        layer: int | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "Sae":
        # Download from the HuggingFace Hub
        repo_path = Path(
            snapshot_download(
                name, allow_patterns=f"layer_{layer}/*" if layer is not None else None,
            )
        )
        if layer is not None:
            repo_path = repo_path / f"layer_{layer}"

        # No layer specified, and there are multiple layers
        elif not repo_path.joinpath("cfg.json").exists():
            raise FileNotFoundError(f"No config file found; try specifying a layer.")

        return Sae.load_from_disk(repo_path, device=device, decoder=decoder)

    @staticmethod
    def load_from_disk(
        path: Path | str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "Sae":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SaeConfig(**cfg_dict)

        sae = Sae(d_in, cfg, device=device, decoder=decoder)
        load_model(
            model=sae,
            filename=str(path / "sae.safetensors"),
            device=str(device),
            # TODO: Maybe be more fine-grained about this in the future?
            strict=decoder,
        )
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
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    def pre_acts(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_sae"]:
        # Remove decoder bias as per Anthropic
        sae_in = x.to(self.dtype) - self.b_dec
        out = self.encoder(sae_in)

        return nn.functional.relu(out) if not self.cfg.signed else out

    def select_topk(self, latents: Float[Tensor, "... d_sae"]) -> EncoderOutput:
        """Select the top-k latents."""
        if self.cfg.signed:
            _, top_indices = latents.abs().topk(self.cfg.k, sorted=False)
            top_acts = latents.gather(dim=-1, index=top_indices)

            return EncoderOutput(top_acts, top_indices)

        return EncoderOutput(*latents.topk(self.cfg.k, sorted=False))

    def encode(self, x: Float[Tensor, "... d_in"]) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        return self.select_topk(self.pre_acts(x))

    def decode(
        self,
        top_acts: Float[Tensor, "... d_sae"],
        top_indices: Int64[Tensor, "..."],
    ) -> Float[Tensor, "... d_in"]:
        assert self.W_dec is not None, "Decoder weight was not initialized."

        y = TritonDecoder.apply(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        return y + self.b_dec

    def forward(self, x: Tensor, dead_mask: Tensor | None = None) -> ForwardOutput:
        pre_acts = self.pre_acts(x)
        top_acts, top_indices = self.select_topk(pre_acts)

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
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)

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
        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
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
