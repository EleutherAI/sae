## Introduction
This library trains _k_-sparse autoencoders (SAEs) on the residual stream activations of HuggingFace language models, roughly following the recipe detailed in [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093v1) (Gao et al. 2024).

This is a lean, simple library with few configuration options. Unlike most other SAE libraries (e.g. [SAELens](https://github.com/jbloomAus/SAELens), it does not cache activations on disk, but rather computes them on-the-fly. This allows us to scale to very large models and datasets with zero storage overhead, but has the downside that trying different hyperparameters for the same model and dataset will be slower than if we cached activations (since activations will be re-computed). We may add caching as an option in the future.

Unlike other libraries, we also train an SAE for _every_ layer of the network at once, rather than choosing a single layer to focus on. We will likely add the option to skip layers in the near future.

Following Gao et al., we use a TopK activation function which directly enforces a desired level of sparsity in the activations. This is in contrast to other libraries which use an L1 penalty in the loss function. We believe TopK is a Pareto improvement over the L1 approach, and hence do not plan on supporting it.

## Usage

To train SAEs from the command line, you can use the following command:

```bash
python -m sae EleutherAI/pythia-160m togethercomputer/RedPajama-Data-1T-Sample
```

The CLI supports all of the config options provided by the `TrainConfig` class. You can see them by running `python -m sae --help`.

Programmatic usage is simple. Here is an example:

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae import SaeConfig, SaeTrainer, TrainConfig
from sae.data import chunk_and_tokenize

MODEL = "EleutherAI/pythia-160m"
dataset = load_dataset(
    "togethercomputer/RedPajama-Data-1T-Sample",
    split="train",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenized = chunk_and_tokenize(dataset, tokenizer)


gpt = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map={"": "cuda"},
    torch_dtype=torch.bfloat16,
)

cfg = TrainConfig(
    SaeConfig(gpt.config.hidden_size), batch_size=16
)
trainer = SaeTrainer(cfg, tokenized, gpt)

trainer.fit()
```

## TODO

There are several features that we'd like to add in the near future:
- [ ] Implement AuxK loss for preventing dead latents (HIGH PRIORITY)
- [ ] Support for skipping layers
- [ ] Support for caching activations
- [ ] Evaluate SAEs with KL divergence when grafted into the model

If you'd like to help out with any of these, please feel free to open a PR! You can collaborate with us in the sparse-autoencoders channel of the EleutherAI Discord.