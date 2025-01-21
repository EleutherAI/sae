import torch
import triton
import triton.language as tl


@triton.jit
def embedding_bag_k(
    out_ptr,  # [B, dim]
    indices_ptr,  # [B, bag_size]
    weight_ptr,  # [n_keys**2, dim]
    per_sample_weights,  # [B, bag_size]
    dim: tl.constexpr,
    features: tl.constexpr,
    bag_size: tl.constexpr,
):
    out_idx = tl.program_id(axis=0).to(tl.int64)
    out_value = tl.zeros([dim], dtype=tl.float32)
    for bag in range(0, bag_size):
        my_index = tl.load(indices_ptr + out_idx * bag_size + bag).to(tl.int64)
        my_scaling = tl.load(per_sample_weights + out_idx * bag_size + bag)
        # my_weight = tl.load(weight_ptr + tl.arange(0, dim) + my_index * dim)
        my_weight = tl.load(weight_ptr + tl.arange(0, dim) * features + my_index)
        out_value = out_value + my_weight.to(tl.float32) * my_scaling
    tl.store(out_ptr + out_idx * dim + tl.arange(0, dim), out_value)


def embedding_bag_triton(
    indices: torch.Tensor, weight: torch.Tensor, per_sample_weights: torch.Tensor
) -> torch.Tensor:
    trt_out = torch.empty(
        [indices.shape[0], weight.shape[1]], dtype=weight.dtype, device=weight.device
    )
    weight = weight.mT
    grid = (indices.shape[0],)

    embedding_bag_k[grid](
        trt_out,
        indices,
        weight,
        per_sample_weights,
        dim=weight.shape[0],
        features=weight.shape[1],
        bag_size=indices.shape[1],
        num_warps=1,
        num_stages=1,
    )
    return trt_out


D = 4096
W = D * 32
K = 48
N = 16384

dev = "cuda:6"
indices = torch.randint(0, W, (N, K), device=dev, dtype=torch.int32)
weights = torch.randn((N, K), device=dev, dtype=torch.bfloat16)
weight = torch.randn((D, W), device=dev, dtype=torch.bfloat16)

# from xfog import xformers_embedding_bag
# y1 = xformers_embedding_bag(indices, weight, weights)
y1 = embedding_bag_triton(
    indices, weight.mT, weights)
y2 = torch.nn.functional.embedding_bag(
    indices, weight.mT, per_sample_weights=weights, mode="sum")
print(torch.allclose(y1, y2))