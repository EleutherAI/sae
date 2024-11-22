from polyapprox.ols import ols, OlsResult
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXMLP
from transformers.models.llama.modeling_llama import LlamaMLP

from torch import nn


def linear_approx(mod: nn.Module, mean, cov) -> OlsResult | None:
    # NeoX architecture
    if isinstance(mod, GPTNeoXMLP):
        b1 = mod.dense_h_to_4h.bias.data.float()
        b2 = mod.dense_4h_to_h.bias.data.float()

        W1 = mod.dense_h_to_4h.weight.data.float()
        W2 = mod.dense_4h_to_h.weight.data.float()
        return ols(W1, b1, W2, b2, mean=mean, cov=cov)

    # Llama architecture (Gated Linear Unit)
    elif isinstance(mod, LlamaMLP):
        b1 = mod.dense_h_to_4h.bias.data.float()
        b2 = mod.dense_4h_to_h.bias.data.float()

        W1 = mod.dense_h_to_4h.weight.data.float()
        W2 = mod.dense_4h_to_h.weight.data.float()
        return ols(W1, b1, W2, b2, mean=mean, cov=cov)

    return None