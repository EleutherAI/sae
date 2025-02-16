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
tokenizer.pad_token = tokenizer.eos_token
#%%
from datasets import load_dataset
from functools import partial
from collections import OrderedDict, defaultdict
from tqdm.auto import tqdm
from more_itertools import chunked
from itertools import islice
import numpy as np
import torch
class ActivationLoader(object):
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.ds = load_dataset("HuggingFaceFW/fineweb", split="train", name="sample-10BT")
    
    @torch.inference_mode()
    def __call__(self, take_n, skip_n=0, bs=16, msl=128,
                 hooks=None):
        for m in model.modules():
            m._forward_hooks = OrderedDict()
        cache = defaultdict(list)
        if hooks is not None:
            for layer_idx, module_name in hooks:
                layer = self.model.gpt_neox.layers[layer_idx]
                module = layer.mlp if module_name == "mlp" else 1/0
                module.register_forward_hook(
                    lambda m, i, o:
                        cache[(layer_idx, module_name)]
                        .append((i[0].half(), o.half())))
        for batch_idx, batch in enumerate(chunked(islice(self.ds, skip_n, skip_n+take_n), bs)):
            batch = self.tokenizer.batch_encode_plus(
                [x["text"] for x in batch],
                max_length=msl, padding=True, truncation=True,
                return_tensors="pt"
            )["input_ids"].cuda()
            self.model(batch, output_hidden_states=False, return_dict=False)
            yield {k: v[0] for k, v in cache.items()}
            cache.clear()
        for m in model.modules():
            m._forward_hooks = OrderedDict()
        return cache
# %%
%load_ext autoreload
%autoreload 2
from itda import ITDAConfig, ITDA
import torch
torch.set_grad_enabled(False)
mlp_in_out_cache = []
layer = 9
loader = ActivationLoader(model, tokenizer)
d_model = model.config.hidden_size
add_error = False
subtract_mean = True
transcode = True
itda_config = ITDAConfig(
    d_model=d_model,
    target_l0=32,
    loss_threshold=0.3,
    add_error=add_error,
    subtract_mean=subtract_mean,
)
itda = ITDA(itda_config).half().to(device)
losses = []
dictionary_sizes = []
take_n = 10_000
bs = 16
try:
    for batch_idx, batch in (bar := tqdm(enumerate(loader(take_n=take_n, bs=bs, msl=msl, hooks=[(layer, "mlp")])), total=take_n//bs)):
        x, y = batch[(layer, "mlp")]
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
skip = 10
plt.loglog(np.arange(skip, len(losses)), losses[skip:])
plt.xlim(skip, len(losses))
plt.show()
plt.loglog(dictionary_sizes)
plt.show()
# %%
msl = 128
inputs, outputs = [], []
for batch_idx, batch in enumerate(loader(take_n=100, skip_n=take_n, bs=bs, msl=msl, hooks=[(layer, "mlp")])):
    x, y = batch[(layer, "mlp")]
    x, y = x.view(-1, d_model), y.view(-1, d_model)
    inputs.append(x)
    outputs.append(y)
inputs, outputs = torch.cat(inputs), torch.cat(outputs)
recon = itda(inputs if transcode else outputs, outputs).y_reconstructed
l2_loss = (recon - outputs).pow(2).sum(-1).mean()
total_variance = (outputs - outputs.mean(0)).pow(2).sum(-1).mean()
fvu = l2_loss / total_variance
fvu
# %%
outputs.shape
#%%
dictionary_in.shape

# %%
