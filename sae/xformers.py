# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
from typing import Tuple
import triton
from triton import language as tl
from torch.nn import functional as F


@triton.jit
def embedding_bag_k(
    out_ptr,  # [B, dim]
    indices_ptr,  # [B, bag_size]
    weight_ptr,  # [n_keys**2, dim]
    per_sample_weights,  # [B, bag_size]
    dim: tl.constexpr,
    bag_size: tl.constexpr,
):
    out_idx = tl.program_id(axis=0).to(tl.int64)
    out_value = tl.zeros([dim], dtype=tl.float32)
    for bag in range(0, bag_size):
        my_index = tl.load(indices_ptr + out_idx * bag_size + bag).to(tl.int64)
        my_scaling = tl.load(per_sample_weights + out_idx * bag_size + bag)
        my_weight = tl.load(weight_ptr + tl.arange(0, dim) + my_index * dim)
        out_value = out_value + my_weight.to(tl.float32) * my_scaling
    tl.store(out_ptr + out_idx * dim + tl.arange(0, dim), out_value)


def embedding_bag_triton(
    indices: torch.Tensor, weight: torch.Tensor, per_sample_weights: torch.Tensor
) -> torch.Tensor:
    trt_out = torch.empty(
        [indices.shape[0], weight.shape[1]], dtype=weight.dtype, device=weight.device
    )
    grid = (indices.shape[0],)

    embedding_bag_k[grid](
        trt_out,
        indices,
        weight,
        per_sample_weights,
        dim=weight.shape[-1],
        bag_size=indices.shape[1],
        #num_warps=1,
        #num_stages=1,
    )
    return trt_out


@triton.jit
def embedding_bag_bw_k2(
    weight_grad_signaling_ptr,  # [K]
    weight_grad_ptr,  # [K, dim]
    per_sample_weights_grad_ptr,  # [B, bag_size]
    indices_ptr,  # [B, bag_size]
    weight_ptr,  # [K, dim]
    per_sample_weights_ptr,  # [B, bag_size]
    gradient_ptr,  # [B, dim]
    dim: tl.constexpr,
    bag_size: tl.constexpr,
    USE_ATOMICS: tl.constexpr,
):
    batch_id = tl.program_id(axis=0).to(tl.int64)
    gradient = tl.load(gradient_ptr + batch_id * dim + tl.arange(0, dim))
    gradient = gradient.to(tl.float32)
    for bag in range(bag_size):
        my_index = tl.load(indices_ptr + batch_id * bag_size + bag).to(tl.int64)
        my_scaling = tl.load(per_sample_weights_ptr + batch_id * bag_size + bag)
        my_weight = tl.load(weight_ptr + my_index * dim + tl.arange(0, dim))
        my_weight = my_weight.to(tl.float32)
        weight_grad = gradient * my_scaling
        per_sample_weights_grad = gradient * my_weight
        per_sample_weights_grad = tl.sum(per_sample_weights_grad)
        if USE_ATOMICS:
            addr = weight_grad_ptr + my_index * dim + tl.arange(0, dim)
            tl.atomic_add(
                addr,
                weight_grad,
                sem="relaxed",
            )
        else:
            # Get lock for `weight.grad` row
            old_val_sign = tl.atomic_or(
                weight_grad_signaling_ptr + my_index, 1, sem="acquire"
            )
            while (old_val_sign & 1) == 1:
                old_val_sign = tl.atomic_or(
                    weight_grad_signaling_ptr + my_index, 1, sem="acquire"
                )
            if old_val_sign & 2:  # already initialized
                old_val = tl.load(
                    weight_grad_ptr + my_index * dim + tl.arange(0, dim)
                ).to(tl.float32)
            else:
                old_val = tl.zeros([dim], dtype=tl.float32)
            tl.store(
                weight_grad_ptr + my_index * dim + tl.arange(0, dim),
                weight_grad + old_val.to(tl.float32),
            )
            tl.atomic_xchg(
                weight_grad_signaling_ptr + my_index, 2, sem="release"
            )  # < Release lock
        tl.store(
            per_sample_weights_grad_ptr + batch_id * bag_size + bag,
            per_sample_weights_grad,
        )


@triton.jit
def embedding_bag_init_where_needed_k(
    weight_grad_signaling_ptr,  # [K]
    weight_grad_ptr,  # [K, dim]
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    key_id = tl.program_id(axis=0).to(tl.int64) * BLOCK_SIZE
    for i in range(BLOCK_SIZE):
        val = tl.load(weight_grad_signaling_ptr + key_id + i)
        if val == 0:
            tl.store(
                weight_grad_ptr + (key_id + i) * dim + tl.arange(0, dim),
                tl.zeros([dim], dtype=tl.float32),
            )


def embedding_bag_bw2(
    indices: torch.Tensor,
    weight: torch.Tensor,
    per_sample_weights: torch.Tensor,
    gradient: torch.Tensor,
    use_atomics: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns: [weight.grad, per_sample_weights.grad]
    K, dim = weight.shape
    B, bag_size = indices.shape
    weight_grad_signaling = torch.empty((K,), dtype=torch.uint32, device=indices.device)
    if not use_atomics:
        weight_grad_signaling.fill_(0)
    weight_grad = torch.empty_like(weight)
    if use_atomics:
        weight_grad.fill_(0)
    per_sample_weights_grad = torch.empty_like(per_sample_weights)
    assert indices.is_contiguous()
    assert weight.is_contiguous()
    assert per_sample_weights.is_contiguous()
    assert gradient.is_contiguous()
    assert indices.shape == (B, bag_size)
    assert weight.shape == (K, dim)
    assert per_sample_weights.shape == (B, bag_size)
    assert gradient.shape == (B, dim)
    embedding_bag_bw_k2[(B,)](
        weight_grad_signaling,
        weight_grad,
        per_sample_weights_grad,
        indices,
        weight,
        per_sample_weights,
        gradient,
        dim=dim,
        bag_size=bag_size,
        num_warps=1,
        USE_ATOMICS=True,  # use_atomics,
    )
    if not use_atomics:
        BLOCK_SIZE = 16
        if K % BLOCK_SIZE:
            BLOCK_SIZE = 1
        embedding_bag_init_where_needed_k[(K // BLOCK_SIZE,)](
            weight_grad_signaling,
            weight_grad,
            dim=dim,
            num_warps=1,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return weight_grad, per_sample_weights_grad


@triton.jit
def count_per_embedding_k(
    count_per_emb_ptr,  # [K+1] (out)
    indices_ptr,  # [B, bag_size]
    bag_size: tl.constexpr,
):
    batch_id = tl.program_id(axis=0).to(tl.int64)
    for i in range(bag_size):
        embedding_id = tl.load(indices_ptr + batch_id * bag_size + i)
        tl.atomic_add(
            count_per_emb_ptr + embedding_id + 1,
            1,
            sem="relaxed",
        )


@triton.jit
def map_embeddings_and_outputs_k(
    reverse_mapping_ptr,  # [B*bag_size] (out)
    mapping_write_pos_ptr,  # [K] (tmp)
    indices_ptr,  # [B, bag_size]
    bag_size: tl.constexpr,
):
    batch_id = tl.program_id(axis=0).to(tl.int64)
    for bag_id in range(bag_size):
        embedding_id = tl.load(indices_ptr + batch_id * bag_size + bag_id)
        write_pos = tl.atomic_add(
            mapping_write_pos_ptr + embedding_id, 1, sem="relaxed"
        )
        tl.store(reverse_mapping_ptr + write_pos, batch_id * bag_size + bag_id)


@triton.jit
def aggregate_gradient_for_embedding_k(
    weight_grad_ptr,  # [K, dim] (out)
    per_sample_weights_grad_ptr,  # [B, bag_size] (out)
    emb_argsorted_ptr,  # [K+1]
    weight_ptr,  # [K, dim] (out)
    emb_begin_pos_ptr,  # [K+1]
    reverse_mapping_ptr,  # [B*bag_size]
    per_sample_weights_ptr,  # [B, bag_size]
    gradient_ptr,  # [B, dim]
    dim: tl.constexpr,
    bag_size: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    first_embedding_id = tl.program_id(axis=0).to(tl.int64)
    for k in range(0, BLOCK_SIZE):
        embedding_id = first_embedding_id + (K // BLOCK_SIZE) * k
        # embedding_id = first_embedding_id * BLOCK_SIZE + k
        embedding_id = tl.load(emb_argsorted_ptr + embedding_id).to(tl.int64)
        weight_grad = tl.zeros([dim], dtype=tl.float32)
        begin = tl.load(emb_begin_pos_ptr + embedding_id)
        end = tl.load(emb_begin_pos_ptr + embedding_id + 1)
        weight = tl.load(
            weight_ptr + embedding_id * dim + tl.arange(0, dim),
        ).to(tl.float32)
        for idx in range(begin, end):
            output_indice_id = tl.load(reverse_mapping_ptr + idx).to(tl.int64)
            batch_id = output_indice_id // bag_size
            bag_id = output_indice_id % bag_size
            per_sample_w = tl.load(per_sample_weights_ptr + output_indice_id)
            gradient = tl.load(gradient_ptr + batch_id * dim + tl.arange(0, dim)).to(
                tl.float32
            )
            weight_grad = weight_grad + per_sample_w * gradient
            per_sample_weights_grad = gradient * weight
            per_sample_weights_grad = tl.sum(per_sample_weights_grad)
            tl.store(
                per_sample_weights_grad_ptr + output_indice_id, per_sample_weights_grad
            )
        tl.store(weight_grad_ptr + embedding_id * dim + tl.arange(0, dim), weight_grad)


def embedding_bag_bw_rev_indices(
    indices: torch.Tensor,
    weight: torch.Tensor,
    per_sample_weights: torch.Tensor,
    gradient: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns: [weight.grad, per_sample_weights.grad]

    K, dim = weight.shape
    B, bag_size = indices.shape
    count_per_emb = torch.zeros((K + 1,), dtype=torch.uint32, device=indices.device)
    count_per_embedding_k[(B,)](count_per_emb, indices, bag_size=bag_size, num_warps=1)
    emb_argsorted = count_per_emb[1:].int().argsort(descending=True)
    emb_begin_pos = count_per_emb.cumsum(0)
    reverse_mapping = torch.empty(
        [B * bag_size], dtype=torch.uint32, device=indices.device
    )
    assert B * bag_size < 2 ** (reverse_mapping.dtype.itemsize * 8 - 1)
    map_embeddings_and_outputs_k[(B,)](
        reverse_mapping_ptr=reverse_mapping,
        mapping_write_pos_ptr=emb_begin_pos.clone(),
        indices_ptr=indices,
        bag_size=bag_size,
        num_warps=1,
    )
    weight_grad = torch.empty_like(weight)
    per_sample_weights_grad = torch.empty_like(per_sample_weights)
    BLOCK_SIZE = 8
    assert (K % BLOCK_SIZE) == 0
    aggregate_gradient_for_embedding_k[(K // BLOCK_SIZE,)](
        weight_grad_ptr=weight_grad,
        emb_begin_pos_ptr=emb_begin_pos,
        emb_argsorted_ptr=emb_argsorted,
        per_sample_weights_grad_ptr=per_sample_weights_grad,
        weight_ptr=weight,
        reverse_mapping_ptr=reverse_mapping,
        per_sample_weights_ptr=per_sample_weights,
        gradient_ptr=gradient,
        dim=dim,
        bag_size=bag_size,
        B=B,
        K=K,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )
    return weight_grad, per_sample_weights_grad


class xFormersEmbeddingBag(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,
        weight: torch.Tensor,
        per_sample_weights: torch.Tensor,
        bw_algo: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(indices, weight, per_sample_weights)
        ctx.bw_algo = bw_algo
        return embedding_bag_triton(indices, weight, per_sample_weights)

    @staticmethod
    def backward(ctx, gradient):
        indices, weight, per_sample_weights = ctx.saved_tensors
        if ctx.bw_algo in ["lock", "atomics"]:
            # TODO : Remove lock and atomics and only keep reverse_indices
            weight_g, per_sample_weights_g = embedding_bag_bw2(
                indices,
                weight,
                per_sample_weights,
                gradient,
                use_atomics=ctx.bw_algo == "atomics",
            )
        else:
            assert ctx.bw_algo == "reverse_indices"
            weight_g, per_sample_weights_g = embedding_bag_bw_rev_indices(
                indices,
                weight,
                per_sample_weights,
                gradient,
            )
        return None, weight_g, per_sample_weights_g, None


def xformers_embedding_bag(
    indices: torch.Tensor,
    weight: torch.Tensor,
    per_sample_weights: torch.Tensor,
    mode: str = "sum",
    bw_algo: str = "reverse_indices",
) -> torch.Tensor:
    assert mode == "sum"
    return xFormersEmbeddingBag.apply(indices, weight, per_sample_weights, bw_algo)
