from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint
import pytest

from ownGPT import model, own_tokenize, train
from ownGPT.config import Config


@dataclass
class Cfg:
    # parameters for models/checkpoint
    d_attn: int = 128
    d_e: int = 64  # 768 #256
    d_mlp: int = 64  # 768  # TODO: which choice?
    d_v: int = 128
    d_out: int = 128 # 256  # output dim
    # vocab_size: int = 50304
    l_x: int = 12  # 16
    l_z: int = 16
    l_max: int = 8  # 1024
    num_layers: int = 4
    attn_heads: int = 4


@pytest.fixture
def config():
    config = model.DTransformerConfig(
        vocab_size=0, # set it later correctly
        l_max=16,
        d_e=64,
        num_layers=4,
        attn_heads=4,
    )
    return config

@pytest.fixture
def save_path(test_path):
    return test_path / Path("models/test")


def test_NaiveDataLoader():
    train_set = Config.data_path / "tokenize" / "test.txt"
    seq_length = 7
    batch_size = 12
    loader = train.NaiveDataLoader(
        data_path=train_set,
        seq_length=seq_length
    )
    tokenizer = loader.get_tokenizer()
    key = jax.random.PRNGKey(23)
    minibatch = loader.minibatch(batch_size=batch_size, key=key, tokenizer=tokenizer)
    assert minibatch.shape == (batch_size, seq_length + 1)
    print(minibatch)


def test_training(test_path, config, save_path):
    # delete checkpoint if it exists for testing purposes
    # TODO: fix!!
    #try:
    #    save_path.unlink()
    #except FileNotFoundError:
    #    print(f"{save_path.name} not found. Continue.")
    train_set = Config.data_path / "tokenize" / "test.txt"
    train.train_dtransformer(train_set=train_set, config=config, batch_size=32, epochs=48, save_path=save_path)


def test_trained_model(test_path, config, save_path):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(save_path)
    model_params = restored["model"]["params"]  # because we saved it like that
    params = {"params": model_params}
    print(params)
    # print(model_params.keys())

    train_set = Config.data_path / "tokenize" / "test.txt"
    tokenizer = own_tokenize.train_BPE_tokenizer([str(train_set)])
    vocab_size = tokenizer.get_vocab_size()

    # this form is determined by the checkpoint
    # TODO: could read from checkpoint?
    config.vocab_size = vocab_size
    dtransfomer = model.DTransformer(config)

    inputs = ["Here we just", "How are you?", "Kindergarten is"]
    for x in inputs:
        print(f"Input: {x} \nEncoded: {tokenizer.encode(x).ids}")
        returned = dtransfomer.infer(
            tokenizer=tokenizer, x=x, l_gen=16, variables=params, temperature=.1
        )
        print(f"Returned: {returned}")
