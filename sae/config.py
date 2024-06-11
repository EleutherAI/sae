from dataclasses import dataclass
from simple_parsing import list_field, Serializable

from . import __version__


@dataclass
class SaeConfig(Serializable):
    """
    Configuration for training a sparse autoencoder on a language model.
    """
    expansion_factor: int = 32
    """Multiple of the input dimension to use as the SAE dimension."""

    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""

    k: int = 32
    """Number of nonzero features."""


@dataclass
class TrainConfig(Serializable):
    sae: SaeConfig

    batch_size: int = 8
    """Batch size measured in sequences."""

    grad_acc_steps: int = 1
    """Number of steps over which to accumulate gradients."""

    lr: float | None = None
    """Base LR. If None, it is automatically chosen based on the number of latents."""

    lr_warmup_steps: int = 1000
 
    auxk_alpha: float = 1 / 32
    """Weight of the auxiliary loss term."""

    dead_feature_threshold: int = 10_000_000
    """Number of tokens after which a feature is considered dead."""

    layers: list[int] = list_field()
    """List of layer indices to train SAEs on."""

    layer_stride: int = 1
    """Stride between layers to train SAEs on."""

    save_every: int = 1000
    """Save SAEs every `save_every` steps."""

    hf_token: str | None = None
    """Huggingface API token for downloading models."""

    load_in_8bit: bool = False
    """Load the model in 8-bit mode."""

    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_log_frequency: int = 10

    def __post_init__(self):
        assert not (self.layers and self.layer_stride != 1), "Cannot specify both `layers` and `layer_stride`."
