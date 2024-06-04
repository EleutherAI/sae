"""Most of this is just copied over from Arthur's code and slightly simplified:
https://github.com/ArthurConmy/sae/blob/main/sae/model.py
"""

import json
import math
import os
from typing import Callable, NamedTuple, Optional

import einops
import torch
from jaxtyping import Float
from safetensors.torch import save_file
from torch import nn, Tensor

from .config import SaeConfig, load_pretrained_sae_lens_sae_components

SPARSITY_PATH = "sparsity.safetensors"
SAE_WEIGHTS_PATH = "sae_weights.safetensors"
SAE_CFG_PATH = "cfg.json"


class ForwardOutput(NamedTuple):
    sae_out: Tensor
    feature_acts: Tensor
    loss: Tensor
    fvu: Tensor
    sparsity_loss: Tensor


class SparseAutoencoder(nn.Module):
    sparsity_weight: float
    d_sae: int
    normalize_sae_decoder: bool
    dtype: torch.dtype
    noise_scale: float
    activation_fn: Callable[[Tensor], Tensor]

    def __init__(
        self,
        cfg: SaeConfig,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        if not isinstance(self.d_in, int):
            raise ValueError(
                f"d_in must be an int but was {self.d_in=}; {type(self.d_in)=}"
            )
        assert cfg.d_sae is not None  # keep pyright happy

        self.d_sae = cfg.d_sae
        self.sparsity_weight = cfg.sparsity_weight
        self.dtype = cfg.dtype
        self.normalize_sae_decoder = cfg.normalize_sae_decoder
        self.noise_scale = cfg.noise_scale
        self.activation_fn = nn.ReLU()

        # no config changes encoder bias init for now.
        self.b_enc = nn.Parameter(
            torch.zeros(self.d_sae, dtype=self.dtype, device=device)
        )

        # Start with the default init strategy:
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_sae, self.d_in, dtype=self.dtype, device=device)
            )
        )
        if self.cfg.decoder_orthogonal_init:
            self.W_dec.data = nn.init.orthogonal_(self.W_dec.data.T).T

        elif self.cfg.normalize_sae_decoder:
            self.set_decoder_norm_to_unit_norm()

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_in, self.d_sae, dtype=self.dtype, device=device),
                nonlinearity="relu",
            )
        )

        # Then we intialize the encoder weights (either as the transpose of decoder or not)
        if self.cfg.init_encoder_as_decoder_transpose:
            self.W_enc.data = self.W_dec.data.T.clone().contiguous()
        else:
            self.W_enc = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(self.d_in, self.d_sae, dtype=self.dtype, device=device),
                    a=math.sqrt(5)
                )
            )

        if self.normalize_sae_decoder:
            with torch.no_grad():
                # Anthropic normalize this to have unit columns
                self.set_decoder_norm_to_unit_norm()

        # methdods which change b_dec as a function of the dataset are implemented after init.
        self.b_dec = nn.Parameter(
            torch.zeros(self.d_in, dtype=self.dtype, device=device)
        )

    @property
    def device(self):
        return self.b_enc.device

    def encode(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_sae"]:
        feature_acts, _ = self._encode_with_hidden_pre(x)
        return feature_acts

    def _encode_with_hidden_pre(
        self, x: Float[Tensor, "... d_in"]
    ) -> tuple[Float[Tensor, "... d_sae"], Float[Tensor, "... d_sae"]]:
        """Encodes input activation tensor x into an SAE feature activation tensor."""
        # move x to correct dtype
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

        noisy_hidden_pre = hidden_pre
        if self.noise_scale > 0:
            noise = torch.randn_like(hidden_pre) * self.noise_scale
            noisy_hidden_pre = hidden_pre + noise
        feature_acts = self.activation_fn(noisy_hidden_pre)

        return feature_acts, hidden_pre

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
        feature_acts, _ = self._encode_with_hidden_pre(x)
        sae_out = self.decode(feature_acts)

        per_token_l2_loss = (sae_out - x).pow(2).sum(0)
        total_variance = (x - x.mean(0)).pow(2).sum(0)
        fvu = torch.mean(per_token_l2_loss / total_variance)

        sparsity = torch.norm(feature_acts, p=1, dim=-1).mean()# hoyer_measure(feature_acts).mean()
        sparsity_loss = (self.sparsity_weight * sparsity)
        loss = fvu + sparsity_loss

        return ForwardOutput(
            sae_out=sae_out,
            feature_acts=feature_acts,
            loss=loss,
            fvu=fvu,
            sparsity_loss=sparsity_loss,
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

    def save_model(self, path: str, sparsity: Optional[Tensor] = None):
        if not os.path.exists(path):
            os.mkdir(path)

        # generate the weights
        save_file(self.state_dict(), f"{path}/{SAE_WEIGHTS_PATH}")

        # save the config
        config = {
            **self.cfg.__dict__,
            # some args may not be serializable by default
            "dtype": str(self.cfg.dtype),
            "device": str(self.device),
        }

        with open(f"{path}/{SAE_CFG_PATH}", "w") as f:
            json.dump(config, f)

        if sparsity is not None:
            sparsity_in_dict = {"sparsity": sparsity}
            save_file(sparsity_in_dict, f"{path}/{SPARSITY_PATH}")  # type: ignore

    @classmethod
    def load_from_pretrained(
        cls, path: str, device: str = "cpu"
    ) -> "SparseAutoencoder":
        config_path = os.path.join(path, "cfg.json")
        weight_path = os.path.join(path, "sae_weights.safetensors")

        cfg, state_dict = load_pretrained_sae_lens_sae_components(
            config_path, weight_path, device
        )

        sae = cls(cfg)
        sae.load_state_dict(state_dict)

        return sae

    def get_name(self):
        sae_name = f"sae_{self.cfg.d_sae}"
        return sae_name


def hoyer_measure(z: Tensor, dim: int = -1) -> Tensor:
    """Hoyer sparsity measure."""
    eps = torch.finfo(z.dtype).eps
    scale = z.shape[dim] ** -0.5

    numer = z.norm(p=1, dim=dim)
    denom = z.norm(p=2, dim=dim)
    return numer.div(denom + eps) * scale
