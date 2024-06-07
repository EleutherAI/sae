from dataclasses import dataclass
from simple_parsing import Serializable

from . import __version__


@dataclass
class SaeConfig(Serializable):
    """
    Configuration for training a sparse autoencoder on a language model.
    """
    # SAE Parameters
    d_in: int
    d_sae: int |  None = None
    expansion_factor: int = 32
    normalize_decoder: bool = True

    # Number of nonzero features
    k: int = 50

    apply_b_dec_to_input: bool = True

    autocast: bool = True  # autocast to autocast_dtype during training

    ## Batch size
    batch_size: int = 1
    grad_acc_steps: int = 1

    ## AuxK hparams
    auxk_alpha: float = 1 / 32
    dead_feature_threshold: int = 10_000_000

    ## Learning Rate Schedule
    lr: float = 1e-4
    lr_warm_up_steps: int = 1000

    # WANDB
    log_to_wandb: bool = True
    wandb_id: str | None = None
    run_name: str | None = None
    wandb_entity: str | None = None
    wandb_log_frequency: int = 10
    eval_every_n_wandb_logs: int = 100  # logs every 1000 steps.

    def __post_init__(self):
        if not isinstance(self.expansion_factor, list):
            self.d_sae = self.d_in * self.expansion_factor
