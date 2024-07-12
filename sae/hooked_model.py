from torch import nn, Tensor

from .sae import Sae, SaeConfig


class HookedModel(nn.Module):
    """Wrapper around a model to hook SAEs into its forward pass."""
    @classmethod
    def build(
        cls,
        model: nn.Module,
        cfg: SaeConfig,
        hookpoints: list[str],
    ):
        """Build SAEs at `hookpoints` on `model`, wrapping them in a `HookedModel`."""


    def __init__(
        self,
        model: nn.Module,
        cfg: SaeConfig,
        hookpoints: list[str],
    ):
        super().__init__()

        self.model = model
    
    def forward(self, *args, **kwargs) -> Tensor:
        # Add forward hooks
        try:
            pass
        finally:
            pass