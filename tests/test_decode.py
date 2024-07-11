from sae.utils import eager_decode, triton_decode
import torch


def test_decode():
    batch = 2
    d_in = 50
    d_sae = 100
    k = 10

    # Fake data
    latents = torch.rand(batch, d_sae, device="cuda")
    W_dec = torch.randn(d_sae, d_in, device="cuda")

    top_vals, top_idx = latents.topk(k)
    eager_res = eager_decode(top_idx, top_vals, W_dec.mT)
    triton_res = triton_decode(top_idx, top_vals, W_dec.mT)

    torch.testing.assert_allclose(eager_res, triton_res)