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
    d_attn: int = 124
    d_e: int = 512  # 768 #256
    d_mlp: int = 512  # 768  # TODO: which choice?
    d_v: int = 128
    d_out: int = 512 # 256  # output dim
    vocab_size: int = 50304
    l_x: int = 12  # 16
    l_z: int = 16
    l_max: int = 64  # 1024


@pytest.fixture
def cfg_train():
    return Cfg()


def test_NaiveDataLoader():
    train_set = Config.data_path / "tokenize" / "test.txt"
    loader = train.NaiveDataLoader(
        data_path=train_set,
        seq_length=5
    )
    key = jax.random.PRNGKey(23)
    minibatch = loader.minibatch(batch_size=4, key=key)
    print(minibatch)


def test_trained_model(test_path, cfg_train):
    ckpt = test_path / Path("models")
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(ckpt)
    model_params = restored["model"]["params"]  # because we saved it like that
    params = {"params": model_params}
    print(params)
    # print(model_params.keys())

    train_set = Config.data_path / "tokenize" / "test.txt"
    tokenizer = own_tokenize.train_BPE_tokenizer([str(train_set)])
    vocab_size = tokenizer.get_vocab_size()

    # this form is determined by the checkpoint
    # TODO: could read from checkpoint?
    dtransfomer = model.DTransformer(
        vocab_size=vocab_size,
        l_max=cfg_train.l_max,
        d_e=cfg_train.d_e,
        d_mlp=cfg_train.d_mlp,
        d_v=cfg_train.d_v,
        num_layers=4,
        attn_heads=8,
    )

    inputs = ["Here we just", "How are you?"]
    for x in inputs:
        print(f"Input: {x} \nEncoded: {tokenizer.encode(x).ids}")
        returned = dtransfomer.infer(
            tokenizer=tokenizer, x=x, l_gen=50, variables=params, temperature=.1
        )
        print(f"Returned: {returned}")
