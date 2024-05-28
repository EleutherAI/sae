import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae import SaeConfig, SaeTrainer
from sae.data import chunk_and_tokenize

dataset = Dataset.from_json("/mnt/ssd-1/igor/data/pile/val.jsonl")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokenized = chunk_and_tokenize(dataset, tokenizer)


gpt = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    device_map={"": "cuda"},
    torch_dtype=torch.bfloat16,
)

cfg = SaeConfig(gpt.config.hidden_size, batch_size=128)
trainer = SaeTrainer(cfg, tokenized, gpt)

trainer.fit()
