from ownGPT import own_tokenize
import jax.numpy as jnp
from pathlib import Path
import pytest

# TODO: add config for file paths etc.


@pytest.fixture
def test_text_file(test_path):
    test_file = test_path / Path("test.txt")
    assert test_file.exists()
    return test_file


def test_train_BPE_tokenizer(test_text_file, test_path):
    save_tokens = test_path / Path("test_tokens.json")
    bpe_tokenizer = own_tokenize.train_BPE_tokenizer(
        [str(test_text_file)], str(save_tokens)
    )
    encode = bpe_tokenizer.encode("Hello, my lord!", "How is it going?")
    print(encode.tokens)
    print(jnp.array(encode.ids))
    print(bpe_tokenizer.decode([6, 7, 8]))
    # remove file
    # save_tokens.unlink()
