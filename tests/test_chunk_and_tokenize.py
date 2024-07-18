import pytest
from transformers import GPT2TokenizerFast
from datasets import Dataset
from sae.data import chunk_and_tokenize

@pytest.fixture
def setup_data():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    data = Dataset.from_dict({
        "text": ["This is a very short sentence.",] * 2000 + \
        ["This is a sentence which is just a little bit longer, you better have a way of dealing with it homeslice."] * 3

    })
    return tokenizer, data

def test_chunk_and_tokenize(setup_data):
    tokenizer, data = setup_data

    # Perform chunking and tokenization
    max_seq_len = 10  # Setting a small max_seq_len for testing overflow
    tokenized_data = chunk_and_tokenize(
        data,
        tokenizer,
        max_seq_len=max_seq_len,
        num_proc=2,
        batch_size=32,
    )

    # Verify the output
    input_ids = tokenized_data["input_ids"]
    input_id_lengths = [len(ids) for ids in input_ids] 

    assert all([l == max_seq_len for l in input_id_lengths]), f"All input_ids should have max_seq_len, got {input_id_lengths}"
    assert len(input_ids[-1]) <= max_seq_len, "Last input_ids should be <= max_seq_len"
    assert len(input_ids) >= 1610, f"Expected at least 1610 input_ids, got {len(input_ids)}"
    
