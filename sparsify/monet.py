from transformers.activations import ACT2FN
from torch import nn
import torch
from dataclasses import dataclass


@dataclass
class MonetConfig:
    moe_heads: int = 6
    moe_experts: int = 512
    moe_dim: int = 12
    moe_topk: int = 8
    hidden_size: int = 1024
    hidden_act: str = "relu2"
    exact_topk: bool = True


class MonetRouter(nn.Module):
    def __init__(self, config: MonetConfig):
        super().__init__()
        self.config = config
        flatten_shape = config.moe_heads * config.moe_experts

        self.w1 = nn.Linear(config.hidden_size, flatten_shape, bias=False)
        self.w2 = nn.Linear(config.hidden_size, flatten_shape, bias=False)
        self.norm1 = nn.LayerNorm(config.moe_heads, bias=False)
        self.norm2 = nn.LayerNorm(config.moe_heads, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from scipy.stats import norm

        # b, moe_heads, moe_experts
        g1z = self.w1(x).unflatten(-1, (self.config.moe_heads, -1)).float()
        g2z = self.w2(x).unflatten(-1, (self.config.moe_heads, -1)).float()

        # (b moe_experts), moe_heads
        g1n = self.norm1(g1z.transpose(-1, -2).flatten(0, -2))
        g2n = self.norm2(g2z.transpose(-1, -2).flatten(0, -2))
        # b, moe_experts, moe_heads -> b, moe_heads, moe_experts
        g1n = g1n.view(g1z.size(0), g1z.size(2), -1).transpose(-1, -2)
        g2n = g2n.view(g2z.size(0), g2z.size(2), -1).transpose(-1, -2)

        if self.config.exact_topk:
            g1s = g1n.topk(self.config.moe_topk, dim=-1).values.amin(-1, keepdim=True)
            g2s = g2n.topk(self.config.moe_topk, dim=-1).values.amin(-1, keepdim=True)
        else:
            sigma = float(norm.ppf(1 - self.config.moe_topk / self.config.moe_experts))
            g1s = g1n.amax(-1, keepdim=True).clamp_max_(sigma)
            g2s = g2n.amax(-1, keepdim=True).clamp_max_(sigma)

        g1 = nn.functional.softmax(torch.where(g1n >= g1s, g1z, -1e10), dim=-1)
        g2 = nn.functional.softmax(torch.where(g2n >= g2s, g2z, -1e10), dim=-1)
        return g1, g2


class MonetMoVDE(nn.Module):
    def __init__(self, config: MonetConfig):
        super().__init__()
        self.config = config
        self.act_fn = ACT2FN[config.hidden_act]
        flatten_shape = config.moe_experts * config.moe_dim // 2

        self.u1 = nn.Linear(config.hidden_size, flatten_shape)
        self.u2 = nn.Linear(config.hidden_size, flatten_shape)

        self.v11 = nn.Linear(flatten_shape, config.hidden_size // 2, bias=False)
        self.v12 = nn.Linear(flatten_shape, config.hidden_size // 2, bias=False)
        self.v21 = nn.Linear(flatten_shape, config.hidden_size // 2, bias=False)
        self.v22 = nn.Linear(flatten_shape, config.hidden_size // 2, bias=False)

        self.b1 = nn.Parameter(torch.zeros(config.moe_experts, config.hidden_size // 2))
        self.b2 = nn.Parameter(torch.zeros(config.moe_experts, config.hidden_size // 2))

    def forward(
        self, x: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor
    ) -> torch.Tensor:
        g1, g2 = g1.type_as(x), g2.type_as(x)
        x1 = self.act_fn(self.u1(x).unflatten(-1, (self.config.moe_experts, -1)))
        x2 = self.act_fn(self.u2(x).unflatten(-1, (self.config.moe_experts, -1)))

        x11 = self.v11(torch.einsum("...im,...hi->...im", x1, g1).flatten(-2))
        x12 = self.v12(torch.einsum("...jm,...hj,...hi->...im", x2, g2, g1).flatten(-2))
        x13 = torch.einsum("...hi,id->...d", g1, self.b1.type_as(x))

        x21 = self.v21(torch.einsum("...im,...hi,...hj->...jm", x1, g1, g2).flatten(-2))
        x22 = self.v22(torch.einsum("...jm,...hj->...jm", x2, g2).flatten(-2))
        x23 = torch.einsum("...hj,jd->...d", g2, self.b2.type_as(x))

        return torch.cat((x11 + x12 + x13, x21 + x22 + x23), dim=-1)


class MonetMoHDE(nn.Module):
    def __init__(self, config: MonetConfig):
        super().__init__()
        self.config = config
        self.act_fn = ACT2FN[config.hidden_act]
        flatten_shape = config.moe_experts * config.moe_dim

        self.u = nn.Linear(config.hidden_size, flatten_shape)
        self.v = nn.Linear(flatten_shape, config.hidden_size, bias=False)
        self.b = nn.Parameter(torch.zeros(config.moe_experts, config.hidden_size))

    def forward(
        self, x: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor
    ) -> torch.Tensor:
        g1, g2 = g1.type_as(x), g2.type_as(x)
        x = self.act_fn(self.u(x).unflatten(-1, (self.config.moe_experts, -1)))
        x = self.v(torch.einsum("...im,...hi,...hj->...jm", x, g1, g2).flatten(-2))
        return x + torch.einsum("...hj,jd->...d", g2, self.b)


class Monet(nn.Module):
    def __init__(self, config: MonetConfig):
        super().__init__()
        self.config = config
        self.router = MonetRouter(config)
        self.movde = MonetMoVDE(config)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        g1, g2 = self.router(x)
        return torch.einsum("...hi,...hj->...ij", g1, g2).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g1, g2 = self.router(x)
        return self.movde(x, g1, g2)
