import torch

from sae.xformers import xformers_embedding_bag
from sae.utils import eager_decode, triton_decode


def test_decode():
    batch = 2
    d_in = 64
    d_sae = 128
    k = 10

    # Fake data
    latents = torch.rand(batch, d_sae, device="cuda")
    W_dec = torch.randn(d_sae, d_in, device="cuda")

    top_vals, top_idx = latents.topk(k)
    eager_res = eager_decode(top_idx, top_vals, W_dec.mT)
    triton_res = triton_decode(top_idx, top_vals, W_dec.mT)
    xformer_res = xformers_embedding_bag(top_idx, W_dec, top_vals)

    torch.testing.assert_close(eager_res, triton_res)
    torch.testing.assert_close(eager_res, xformer_res)
