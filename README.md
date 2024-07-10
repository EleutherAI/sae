## Introduction
This library trains _k_-sparse autoencoders (SAEs) on the residual stream activations of HuggingFace language models, roughly following the recipe detailed in [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093v1) (Gao et al. 2024).

This is a lean, simple library with few configuration options. Unlike most other SAE libraries (e.g. [SAELens](https://github.com/jbloomAus/SAELens), it does not cache activations on disk, but rather computes them on-the-fly. This allows us to scale to very large models and datasets with zero storage overhead, but has the downside that trying different hyperparameters for the same model and dataset will be slower than if we cached activations (since activations will be re-computed). We may add caching as an option in the future.

Unlike other libraries, we also train an SAE for _every_ layer of the network at once, rather than choosing a single layer to focus on. We will likely add the option to skip layers in the near future.

Following Gao et al., we use a TopK activation function which directly enforces a desired level of sparsity in the activations. This is in contrast to other libraries which use an L1 penalty in the loss function. We believe TopK is a Pareto improvement over the L1 approach, and hence do not plan on supporting it.

## Loading pretrained SAEs

To load a pretrained SAE from the HuggingFace Hub, you can use the `Sae.load_from_hub` method as follows:

```python
from sae import Sae

sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", layer=10)
```

This will load the SAE for residual stream layer 10 of Llama 3 8B, which was trained with an expansion factor of 32. You can also load the SAEs for all layers at once using `Sae.load_many_from_hub`:

```python
saes = Sae.load_many_from_hub("EleutherAI/sae-llama-3-8b-32x")
saes["layer_10"]
```

The dictionary returned by `load_many_from_hub` is guaranteed to be [naturally sorted](https://en.wikipedia.org/wiki/Natural_sort_order) by the name of the hook point. For the common case where the hook points are named `layer_0`, `layer_1`, ..., `layer_n`, this means that the SAEs will be sorted by layer number. We can then gather the SAE activations for a model forward pass as follows:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
inputs = tokenizer("Hello, world!", return_tensors="pt")

with torch.inference_mode():
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    outputs = model(**inputs, output_hidden_states=True)

    latent_acts = []
    for sae, hidden_state in zip(saes.values(), outputs.hidden_states):
        latent_acts.append(sae.encode(hidden_state))

# Do stuff with the latent activations
```

## Training SAEs

To train SAEs from the command line, you can use the following command:

```bash
python -m sae EleutherAI/pythia-160m togethercomputer/RedPajama-Data-1T-Sample --attn_implementation=eager
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

## Distributed training

We support distributed training via PyTorch's `torchrun` command. By default we use the Distributed Data Parallel method, which means that the weights of each SAE are replicated on every GPU.

```bash
torchrun --nproc_per_node gpu -m sae meta-llama/Meta-Llama-3-8B --batch_size 1 --layers 16 24 --k 192 --grad_acc_steps 8 --ctx_len 2048
```

This is simple, but very memory inefficient. If you want to train SAEs for many layers of a model, we recommend using the `--distribute_layers` flag, which allocates the SAEs for different layers to different GPUs. Currently, we require that the number of GPUs evenly divides the number of layers you're training SAEs for.

```bash
torchrun --nproc_per_node gpu -m sae meta-llama/Meta-Llama-3-8B --distribute_layers --batch_size 1 --layer_stride 2 --grad_acc_steps 8 --ctx_len 2048 --k 192 --load_in_8bit --micro_acc_steps 2
```

The above command trains an SAE for every _even_ layer of Llama 3 8B, using all available GPUs. It accumulates gradients over 8 minibatches, and splits each minibatch into 2 microbatches before feeding them into the SAE encoder, thus saving a lot of memory. It also loads the model in 8-bit precision using `bitsandbytes`. This command requires no more than 48GB of memory per GPU on an 8 GPU node.

## TODO

There are several features that we'd like to add in the near future:
- [x] Distributed Data Parallel (HIGH PRIORITY)
- [x] Implement AuxK loss for preventing dead latents (HIGH PRIORITY)
- [x] Sharding / tensor parallelism for the SAEs (and model too?)
- [x] Support for skipping layers
- [ ] Support for caching activations
- [ ] Evaluate SAEs with KL divergence when grafted into the model

If you'd like to help out with any of these, please feel free to open a PR! You can collaborate with us in the sparse-autoencoders channel of the EleutherAI Discord.
