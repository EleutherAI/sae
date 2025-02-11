from dataclasses import dataclass
from sparsify.xformers_decoder import xformers_embedding_bag
from torch import nn
import torch


def decoder_impl(indices, weights, dictionary):
    return xformers_embedding_bag(
        indices, dictionary, weights, mode="sum"
    )


def grad_pursuit_update_step(signal, weights, dictionary, eps=1e-3):
    residual = signal - torch.einsum('fv,bf->bv', dictionary, weights)
    selected_features = (weights != 0)
    inner_products = torch.einsum('fv,bv->bf', dictionary, residual)
    idx = inner_products.argmax(dim=1)
    selected_features = torch.scatter(selected_features, 1, idx[:, None], torch.full((idx.shape[0], 1), True, device=idx.device))

    grad = selected_features * inner_products
    c = torch.einsum('bf,fv->bv', grad, dictionary)
    c_square_norm = torch.einsum('bv,bv->b', c, c)
    step_size = torch.einsum('bv,bv->b', c, residual) / torch.maximum(c_square_norm, torch.full_like(c_square_norm, eps))
    weights = weights + step_size[:, None] * grad
    weights = torch.nn.functional.relu(weights)
    return weights

def grad_pursuit(signal, dictionary, target_l0):
    if target_l0 < 0 or target_l0 >= dictionary.shape[0]:
        raise ValueError(f"target_l0 must be in [0, {dictionary.shape[0]}), got {target_l0}")
    weights = torch.zeros((signal.shape[0], dictionary.shape[0]), dtype=signal.dtype, device=signal.device)
    for i in range(target_l0):
        weights = grad_pursuit_update_step(signal, weights, dictionary)
    return weights.topk(target_l0, dim=-1)


@dataclass
class ITDAConfig:
    d_model: int
    target_l0: int
    loss_threshold: float
    add_error: bool = False
    subtract_mean: bool = False
    start_size: int = 16


@dataclass
class ITDAOutput:
    weights: torch.Tensor
    indices: torch.Tensor
    x_reconstructed: torch.Tensor
    y_reconstructed: torch.Tensor
    losses: torch.Tensor


class ITDA(nn.Module):
    def __init__(self, config: ITDAConfig):
        super().__init__()
        self.xs = nn.Parameter(
            torch.empty((config.start_size, config.d_model))
        )
        self.ys = nn.Parameter(
            torch.empty((config.start_size, config.d_model))
        )
        self.mean_x, self.mean_y = None, None
        self.dictionary_size = 0
        self.config = config
    
    def forward(self, x, y, target_l0=None):
        assert x.ndim == y.ndim == 2
        if self.config.subtract_mean:
            assert self.mean_x is not None
            assert self.mean_y is not None
        if self.dictionary_size == 0:
            if self.config.subtract_mean:
                x_reconstructed = self.mean_x.broadcast_to(*x.shape[:-1], -1)
                y_reconstructed = self.mean_y.broadcast_to(*y.shape[:-1], -1)
            else:
                x_reconstructed = y_reconstructed = \
                    torch.zeros_like(y)
            weights = torch.zeros((y.shape[0], self.config.target_l0), dtype=y.dtype, device=y.device)
            indices = torch.zeros((y.shape[0], self.config.target_l0), dtype=torch.long, device=y.device)
        else:
            if target_l0 is None:
                target_l0 = self.config.target_l0
            if self.config.subtract_mean:
                x = x - self.mean_x
            weights, indices = grad_pursuit(x, self.xs[:self.dictionary_size], min(target_l0, self.dictionary_size))
            x_reconstructed = decoder_impl(indices, weights, self.xs[:self.dictionary_size])
            y_reconstructed = decoder_impl(indices, weights, self.ys[:self.dictionary_size])
            if self.config.subtract_mean:
                x_reconstructed = x_reconstructed + self.mean_x
                y_reconstructed = y_reconstructed + self.mean_y
        return ITDAOutput(
            weights=weights,
            indices=indices,
            x_reconstructed=x_reconstructed,
            y_reconstructed=y_reconstructed,
            losses=(y_reconstructed - y).pow(2).sum(-1)
        )
        
    @torch.inference_mode()
    def step(self, x, y):
        assert x.ndim == y.ndim == 2
        if self.mean_x is None:
            self.mean_x = x.mean(0)
        if self.mean_y is None:
            self.mean_y = y.mean(0)
        out_0 = self(x, y)
        should_be_added = out_0.losses > self.config.loss_threshold
        if self.config.add_error:
            added_x = x - out_0.x_reconstructed
            added_y = y - out_0.y_reconstructed
        else:
            added_x = x
            added_y = y
        added_x = added_x[should_be_added]
        added_y = added_y[should_be_added]
        added_y = added_y / added_x.norm(dim=-1, keepdim=True)
        added_x = added_x / added_x.norm(dim=-1, keepdim=True)
        n_added = added_x.shape[0]
        while self.dictionary_size + n_added > self.xs.shape[0]:
            self.xs.data = torch.cat([self.xs, torch.empty_like(self.xs)])
            self.ys.data = torch.cat([self.ys, torch.empty_like(self.xs)])
        self.xs[self.dictionary_size:self.dictionary_size + n_added] = added_x
        self.ys[self.dictionary_size:self.dictionary_size + n_added] = added_y
        self.dictionary_size += n_added
        return out_0