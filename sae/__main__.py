from dataclasses import dataclass

from datasets import load_dataset
from simple_parsing import field, parse
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import chunk_and_tokenize
from .trainer import SaeTrainer, TrainConfig


@dataclass
class RunConfig(TrainConfig):
    model: str = field(
        default="EleutherAI/pythia-160m",
        positional=True,
    )
    """Name of the model to train."""

    dataset: str = field(
        default="togethercomputer/RedPajama-Data-1T-Sample",
        positional=True,
    )
    """Path to the dataset to use for training."""

    split: str = "train"
    """Dataset split to use for training."""

    ctx_len: int = 2048
    """Context length to use for training."""


def run():
    args = parse(RunConfig)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": "cuda"},
        torch_dtype="auto",
    )

    print(f"Training on '{args.dataset}' (split '{args.split}')")
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        # TODO: Maybe set this to False by default? But RPJ requires it.
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized = chunk_and_tokenize(dataset, tokenizer, max_seq_len=args.ctx_len)

    trainer = SaeTrainer(args, tokenized, model)
    trainer.fit()


if __name__ == "__main__":
    run()