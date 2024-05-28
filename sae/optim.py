"""
Took the LR scheduler from my previous work: https://github.com/jbloomAus/DecisionTransformerInterpretability/blob/ee55df35cdb92e81d689c72fb9dd5a7252893363/src/decision_transformer/utils.py#L425
"""

from typing import Any


class L1Scheduler:

    def __init__(
        self,
        l1_warm_up_steps: float,
        total_steps: int,
        final_sparsity_weight: float,
    ):

        self.l1_warmup_steps = l1_warm_up_steps
        # assume using warm-up
        if self.l1_warmup_steps != 0:
            self.current_sparsity_weight = 0.0
        else:
            self.current_sparsity_weight = final_sparsity_weight

        self.final_sparsity_weight = final_sparsity_weight

        self.current_step = 0
        self.total_steps = total_steps
        assert isinstance(self.final_sparsity_weight, float | int)

    def __repr__(self) -> str:
        return (
            f"L1Scheduler(final_l1_value={self.final_sparsity_weight}, "
            f"l1_warmup_steps={self.l1_warmup_steps}, "
            f"total_steps={self.total_steps})"
        )

    def step(self):
        """
        Updates the l1 coefficient of the sparse autoencoder.
        """
        step = self.current_step
        if step < self.l1_warmup_steps:
            self.current_sparsity_weight = self.final_sparsity_weight * (
                (1 + step) / self.l1_warmup_steps
            )  # type: ignore
        else:
            self.current_sparsity_weight = self.final_sparsity_weight  # type: ignore

        self.current_step += 1

    def state_dict(self):
        """State dict for serializing as part of an SAETrainContext."""
        return {
            "l1_warmup_steps": self.l1_warmup_steps,
            "total_steps": self.total_steps,
            "current_sparsity_weight": self.current_sparsity_weight,
            "final_sparsity_weight": self.final_sparsity_weight,
            "current_step": self.current_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Loads all state apart from attached SAE."""
        for k in state_dict:
            setattr(self, k, state_dict[k])
