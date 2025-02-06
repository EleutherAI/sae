import pytest
import torch

from sparsify.utils import eager_decode, triton_decode


@pytest.mark.parametrize("d_in", [48, 64])  # Power of 2 and not
def test_decode(d_in: int):
    batch = 2
    d_sae = 128
    k = 10

    # Fake data
    latents = torch.rand(batch, d_sae, device="cuda")
    W_dec = torch.randn(d_sae, d_in, device="cuda")

    top_vals, top_idx = latents.topk(k)
    eager_res = eager_decode(top_idx, top_vals, W_dec.mT)
    triton_res = triton_decode(top_idx, top_vals, W_dec.mT)

    torch.testing.assert_close(eager_res, triton_res)
