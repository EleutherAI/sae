#%%
from dataclasses import replace
from sae.sae import Sae, SaeConfig
import torch
d_in = 2048
cfg = SaeConfig(k=64)
device = "cuda:6"
dtype = torch.float32
sae_regular = Sae(
    d_in,
    cfg,
    device,
    dtype
)
sae_pkm = Sae(
    d_in,
    replace(cfg, encoder_pkm=True),
    device,
    dtype
)
#%%
fake_data = torch.randn(16384, d_in, device=device, dtype=dtype)
#%%
import torch.utils.benchmark
from tqdm.auto import tqdm, trange
torch.set_float32_matmul_precision("high")
@torch.compile
def regular_forward(encoder, x):
    return encoder(x).topk(cfg.k)
@torch.compile
def pkm_forward(encoder, x):
    return encoder(x).topk(cfg.k)
def pkm_forward_topk(encoder, x):
    return encoder.topk(x, cfg.k)

regular_forward(sae_regular.encoder, fake_data)
pkm_forward(sae_pkm.encoder, fake_data)
pkm_forward_topk(sae_pkm.encoder, fake_data)

def benchmark(fn):
    fn()
    print(torch.utils.benchmark.Timer(
        stmt="fn()",
        globals={"fn": fn},
        setup="fn()",
    ).blocked_autorange())

with torch.inference_mode():
    benchmark(lambda: regular_forward(sae_regular.encoder, fake_data))
    # benchmark(lambda: pkm_forward(sae_pkm.encoder, fake_data))
    benchmark(lambda: pkm_forward_topk(sae_pkm.encoder, fake_data))
#%%
