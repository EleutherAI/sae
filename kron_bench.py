#%%
%env CUDA_VISIBLE_DEVICES=6
import torch
from math import log2
import torch
torch.set_grad_enabled(False)

B = 2**15
N = 1024
M = N * 64
N_b = 4
M_b = 2
N_n = N // N_b
M_n = M // M_b

vals = torch.randn(B, N, dtype=torch.bfloat16, device='cuda')
matrix_inner = torch.randn(M_b, N_b, dtype=torch.bfloat16, device='cuda')
matrix_outer = torch.randn(M_n, N_n, dtype=torch.bfloat16, device='cuda')
# %%
import torch.utils.benchmark
from opt_einsum import contract

def timing(name, fn, *args):
    # fn = torch.compile(fn)
    fn(*args)
    timer = torch.utils.benchmark.Timer(
        stmt="fn(*args)",
        globals={"fn": fn, "args": args},
        setup="fn(*args)",
    ).blocked_autorange()
    print(name, timer.mean)


@torch.compile(mode="max-autotune")
def naive(vals, matrix_inner, matrix_outer):
    return torch.einsum('bn,mn->bm', vals, torch.kron(matrix_outer, matrix_inner))

outside_kron = torch.kron(matrix_outer, matrix_inner)
@torch.compile(mode="max-autotune")
def cached(vals, matrix_inner, matrix_outer):
    return torch.einsum('bn,mn->bm', vals, outside_kron)

@torch.compile(mode="max-autotune")
def sequential(vals, matrix_inner, matrix_outer):
    vals_inner = vals.reshape(B, N_n, N_b)
    vals_inner = vals_inner @ matrix_inner.T
    vals_inner = vals_inner.reshape(B, N_n, M_b)
    vals_outer = torch.einsum('bnd,mn->bmd', vals_inner, matrix_outer)
    return vals_outer.reshape(B, M)

@torch.compile(mode="max-autotune")
def simulate_lora(vals, matrix_inner, matrix_outer):
    vals_inner = vals.reshape(B, N_n, N_b)
    vals_inner = vals_inner @ matrix_inner.T
    vals_outer = vals_inner.reshape(B * M_b, N_n)
    vals_outer = vals_outer @ matrix_outer.T
    return vals_outer

@torch.compile(mode="max-autotune")
def real_lora(vals, matrix_inner, matrix_outer):
    vals_inner = vals.reshape(B, N_n, N_b)
    vals_inner = vals_inner @ matrix_inner.T
    vals_outer = vals_inner.reshape(B * M_b, N_n)
    vals_outer = torch.einsum('bnd,mn->bdm', vals_inner, matrix_outer)
    vals_outer = vals_outer.reshape(B, M)
    return vals_outer

# timing("naive", naive, vals, matrix_inner, matrix_outer)
timing("cached", cached, vals, matrix_inner, matrix_outer)
# timing("sequential", sequential, vals, matrix_inner, matrix_outer)
# timing("simulate_lora", simulate_lora, vals, matrix_inner, matrix_outer)
timing("real_lora", real_lora, vals, matrix_inner, matrix_outer)
# %%
# real_lora(vals, matrix_inner, matrix_outer).dim_order()
torch.compiler.cudagraph_mark_step_begin()
real_lora(vals, matrix_inner, matrix_outer) - cached(vals, matrix_inner, matrix_outer)