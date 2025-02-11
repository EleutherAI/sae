#%%
%env CUDA_VISIBLE_DEVICES=5
device = "cuda:0"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "EleutherAI/pythia-160m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device
)
#%%
from datasets import load_dataset
ds = load_dataset("ogethercomputer/RedPajama-Data-1T-Sample", split="train")
# %%
%load_ext autoreload
%autoreload 2
from functools import partial
from collections import OrderedDict
from tqdm.auto import tqdm
from more_itertools import chunked
from itertools import islice
import numpy as np
from itda import ITDAConfig, ITDA
torch.set_grad_enabled(False)
bs = 16
msl = 128
take_n = 40_000
for m in model.modules():
    m._forward_hooks = OrderedDict()
mlp_in_out_cache = []
layer = 9
model.gpt_neox.layers[layer].mlp.register_forward_hook(lambda m, i, o: mlp_in_out_cache.append((i[0].half(), o.half())))
tokenizer.pad_token = tokenizer.eos_token
d_model = model.config.hidden_size
add_error = False
subtract_mean = False
transcode = True
itda_config = ITDAConfig(
    d_model=d_model,
    target_l0=32,
    loss_threshold=
        (67.0 if not add_error else 60.0)
        if not transcode else 120.0
    ,
    add_error=add_error,
    subtract_mean=subtract_mean,
)
itda = ITDA(itda_config).half().to(device)
losses = []
dictionary_sizes = []
try:
    for batch_idx, batch in enumerate(chunked((bar := tqdm(islice(ds, take_n), total=take_n)), bs)):
        batch = tokenizer.batch_encode_plus(
            [x["text"] for x in batch],
            max_length=msl, padding=True, truncation=True,
            return_tensors="pt"
        )["input_ids"].cuda()
        model(batch, output_hidden_states=False, return_dict=False)
        if batch_idx + 30 > take_n // bs:
            continue
        x, y = mlp_in_out_cache[-1]
        x, y = x.view(-1, d_model), y.view(-1, d_model)
        loss = itda.step(x if transcode else y, y).losses.mean().item()
        bar.set_postfix(
            loss=loss,
            dictionary_size=itda.dictionary_size
        )
        losses.append(loss)
        dictionary_sizes.append(itda.dictionary_size)
except KeyboardInterrupt:
    pass
#%%
from matplotlib import pyplot as plt
plt.loglog(np.arange(100, len(losses)), losses[100:])
plt.xlim(100, len(losses))
plt.show()
plt.plot(dictionary_sizes)
plt.show()
#%%
x = np.arange(len(dictionary_sizes))
y = np.array(dictionary_sizes)
plt.plot(x, y, label="data")
plt.plot(x, (x ** (1/3))*1.8e3, label="model")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.show()
# %%
outputs = torch.cat([o.view(-1, d_model) for i, o in mlp_in_out_cache[-29:]], dim=0)
inputs = torch.cat([i.view(-1, d_model) for i, o in mlp_in_out_cache[-29:]], dim=0)
recon = itda(inputs if transcode else outputs, outputs).y_reconstructed
l2_loss = (recon - outputs).pow(2).sum(-1).mean()
total_variance = (outputs - outputs.mean(0)).pow(2).sum(-1).mean()
fvu = l2_loss / total_variance
fvu
# %%
outputs.shape
#%%
dictionary_in.shape