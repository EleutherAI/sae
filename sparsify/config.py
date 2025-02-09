from dataclasses import dataclass

from simple_parsing import Serializable, list_field


@dataclass
class SaeConfig(Serializable):
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    expansion_factor: int = 64
    """Multiple of the input dimension to use as the SAE dimension."""

    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""
    
    encoder_halut: bool = False
    """Whether to use hashed lookup tables for the encoder weights."""
    
    encoder_pkm: bool = True
    """Whether to use Product Key Memory for the encoder weights."""

    pkm_pad: bool = False
    """Pad the PKM encoder to a power of 2."""

    topk_separate: bool = True
    """Faster top-k for PKM by separating the top-k operation."""

    pkm_bias: bool = True
    """Non-decomposed bias for PKM."""
    
    pkm_init_scale: float = 1.0
    """Scale factor for PKM encoder initialization."""
    
    decoder_xformers: bool = False
    """Xformers implementation for the decoder."""

    num_latents: int = 0
    """Number of latents to use. If 0, use `expansion_factor`."""

    k: int = 32
    """Number of nonzero features."""

    multi_topk: bool = False
    """Use Multi-TopK loss."""

    skip_connection: bool = False
    """Include a linear skip connection."""


@dataclass
class TrainConfig(Serializable):
    sae: SaeConfig

    batch_size: int = 8
    """Batch size measured in sequences."""

    grad_acc_steps: int = 1
    """Number of steps over which to accumulate gradients."""

    micro_acc_steps: int = 1
    """Chunk the activations into this number of microbatches for SAE training."""

    lr: float | None = None
    """Base LR. If None, it is automatically chosen based on the number of latents."""

    lr_warmup_steps: int = 1000

    auxk_alpha: float = 0.0
    """Weight of the auxiliary loss term."""

    dead_feature_threshold: int = 10_000_000
    """Number of tokens after which a feature is considered dead."""

    hookpoints: list[str] = list_field()
    """List of hookpoints to train SAEs on."""

    init_seeds: list[int] = list_field(0)
    """List of random seeds to use for initialization. If more than one, train an SAE
    for each seed."""

    layers: list[int] = list_field()
    """List of layer indices to train SAEs on."""

    layer_stride: int = 1
    """Stride between layers to train SAEs on."""

    transcode: bool = False
    """Predict the output of a module given its input."""

    distribute_modules: bool = False
    """Store a single copy of each SAE, instead of copying them across devices."""

    save_every: int = 1000
    """Save SAEs every `save_every` steps."""

    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_log_frequency: int = 1

    def __post_init__(self):
        assert not (
            self.layers and self.layer_stride != 1
        ), "Cannot specify both `layers` and `layer_stride`."

        assert len(self.init_seeds) > 0, "Must specify at least one random seed."
