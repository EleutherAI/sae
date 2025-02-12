#%%
%env CUDA_VISIBLE_DEVICES=1
import torch
from math import log2
import torch
torch.set_grad_enabled(False)
B = 16384
N = 1024
M = N * 64
K = 512

vals = torch.randn(B, N, dtype=torch.bfloat16, device='cuda')
matrix = torch.randn(N, M, dtype=torch.bfloat16, device='cuda')
#%%

import torch.utils.benchmark

def timing(name, fn, *args):
    # fn = torch.compile(fn)
    fn(*args)
    timer = torch.utils.benchmark.Timer(
        stmt="fn(*args)",
        globals={"fn": fn, "args": args},
        setup="fn(*args)",
    ).blocked_autorange()
    print(name, timer.mean)

import triton
import triton.language as tl
def get_autotune_config():
    configs = [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]
    configs = [
        triton.Config(cfg.kwargs | {"BLOCK_SIZE_N": (cfg.kwargs["BLOCK_SIZE_N"] // 32) * 16 * 128},
                      num_warps=cfg.num_warps, num_stages=cfg.num_stages)
        for cfg in configs
    ]
    return configs

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K, T: tl.constexpr,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    BLOCK_SIZE_T: tl.constexpr = BLOCK_SIZE_N // T
    N_T: tl.constexpr = N // T
    
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_T), dtype=tl.float16)
    for t in range(T):
        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_T), dtype=tl.float32)
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetic` section for details
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            # We accumulate along the K dimension.
            accumulator = tl.dot(a, b, accumulator)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        c_acc = accumulator.to(tl.float16)
        c = tl.maximum(c, c_acc)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_T)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N_T)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b, T=128):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N // T), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        a, b, c,
        M=M, N=N, K=K, T=T,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        # BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64,
        # GROUP_SIZE_M=8
    )
    return c

timing("matmul", matmul, vals, matrix)
timing("dot", torch.matmul, vals, matrix)
# matmul(vals, matrix) - vals @ matrix
#%%
from matplotlib import pyplot as plt
import numpy as np
def p():
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    GROUP_SIZE_M = 8
    N = 16384
    M = 8192
    pid = np.arange(N * M // (BLOCK_SIZE_M * BLOCK_SIZE_N))
    num_pid_m = M // BLOCK_SIZE_M
    num_pid_n = N // BLOCK_SIZE_N
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = np.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    im = np.zeros((M // BLOCK_SIZE_M, N // BLOCK_SIZE_N))
    im[pid_m, pid_n] = pid
    plt.imshow(im)
    plt.xlabel("pid_n")
    plt.ylabel("pid_m")
    plt.colorbar()
    plt.show()
p()
#%%
@torch.compile(mode="max-autotune")
def naive(vals, matrix):
    y = vals @ matrix
    T = M//K
    BLOCK_SIZE_N = matmul_kernel.best_config.kwargs["BLOCK_SIZE_N"]
    return y.view(*y.shape[:-1], -1, T, BLOCK_SIZE_N // T).max(dim=-2)#.values#.view(*y.shape[:-1], -1)

@torch.compile
def lower_bound(vals, matrix):
    return vals @ matrix[:, :K]

# @torch.compile
def triton_impl(vals, matrix):
    return matmul(vals, matrix, T=M//K)

@torch.compile(mode="max-autotune")
def just_matmul(vals, matrix):
    return vals @ matrix

# Time each function
timing("naive", naive, vals, matrix)
timing("triton_impl", triton_impl, vals, matrix)
timing("lower_bound", lower_bound, vals, matrix)
timing("just_matmul", just_matmul, vals, matrix)
#%%
triton_impl(vals, matrix);
triton_impl(vals, matrix) - naive(vals, matrix)
#%%

# %%