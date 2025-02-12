#%%
%env CUDA_VISIBLE_DEVICES=1
from transformers import AutoTokenizer, AutoModelForCausalLM

model_names = dict(
    monet_vd_s = "MonetLLM/monet-vd-850M-100BT-hf",
    monet_vd_l = "MonetLLM/monet-vd-1.4B-100BT-hf",
    monet_hd_s = "MonetLLM/monet-hd-850M-100BT-hf",
    monet_hd_l = "MonetLLM/monet-hd-1.4B-100BT-hf",
    llama_1_4b = "meta-llama/Llama-3.2-1B",
    llama_1_1b = "TinyLlama/TinyLlama-1.1B-Chat-v0.1"
    # pythia_410m = "EleutherAI/pythia-410m"
)
tokenizers = {name: AutoTokenizer.from_pretrained(model) for name, model in model_names.items()}
#%%
sum([x.numel() for x in models["monet_vd_s"].parameters()])
#%%
import torch
torch.set_grad_enabled(False)
models = {name: AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, torch_dtype=torch.bfloat16,).cuda() for name, model in model_names.items()}
#%%
from itertools import product
import torch.utils.benchmark

def time_compiled(model, input_ids):
    model = torch.compile(model)
    model(input_ids)
    timer = torch.utils.benchmark.Timer(
        stmt="model(input_ids)",
        globals={"model": model, "input_ids": input_ids},
        setup="model(input_ids)",
    ).blocked_autorange()
    return timer.mean

model_results = []
for (batch_size, sequence_length) in product((1, 4, 16), (64, 128, 256, 1024)):
    for model_name, model in models.items():
        input_ids = torch.zeros(batch_size, sequence_length, dtype=torch.long, device="cuda")
        time = time_compiled(model, input_ids)
        model_results.append((model_name, batch_size, sequence_length, time))
#%%
import pandas as pd
df = pd.DataFrame(model_results, columns=['model', 'batch_size', 'sequence_length', 'time'])
print(df.to_string())