import logging
from pathlib import Path

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass

from ownGPT import model, own_tokenize

# only write new logs
logging.basicConfig(level=logging.DEBUG, filename="test_model.log", filemode="w")

# TODO: unclear: how to write sensible tests for the outputs? Here we mainly write smoke tests...

@dataclass
class Cfg:
    # nothing magical about these numbers (just make them distinct)
    l_x: int = 6
    d_x: int = 3
    l_z: int = 6
    d_z: int = 5
    l_max: int = 16
    d_attn: int = 8
    d_v: int = 12
    d_e: int = 10
    d_mlp: int = 8
    d_out: int = 10
    attn_heads: int = 4


@pytest.fixture
def cfg():
    return Cfg()


def test_attention(cfg):
    key1, key2, key3 = jax.random.split(key=jax.random.PRNGKey(23), num=3)
    att = model.Attention(d_attn=cfg.d_attn, d_v=cfg.d_v)
    variables = att.init(
        key1,
        jnp.ones((cfg.l_x, cfg.d_x)),
        jnp.ones((cfg.l_z, cfg.d_z)),
    )
    output = att.apply(
        variables,
        jax.random.uniform(key2, shape=(cfg.l_x, cfg.d_x)),
        jax.random.uniform(key3, shape=(cfg.l_z, cfg.d_z)),
    )
    assert output.shape == (cfg.l_x, cfg.d_v)
    print(output)
    # print(variables)


def test_multi_head_attention(cfg):
    mattn = model.MHAttention(
        d_attn=cfg.d_attn,
        d_v=cfg.d_v,
        d_out=cfg.d_out,
        attn_heads=cfg.attn_heads,
        l_x=cfg.l_x,
    )
    variables = mattn.init(
        jax.random.PRNGKey(0),
        jnp.ones((cfg.l_x, cfg.d_x)),
        jnp.ones((cfg.l_z, cfg.d_z)),
    )
    print(variables)
    output = mattn.apply(
        variables,
        jax.random.uniform(key=jax.random.PRNGKey(0), shape=(cfg.l_x, cfg.d_x)),
        jax.random.uniform(key=jax.random.PRNGKey(1), shape=(cfg.l_z, cfg.d_z)),
    )
    print(output)
    assert output.shape == (cfg.l_x, cfg.d_out)


def test_linear_normalization():
    ln = model.LayerNormalization()
    variables = ln.init(
        jax.random.PRNGKey(0),
        jnp.ones(
            10,
        ),
    )
    returned = ln.apply(
        variables,
        jax.random.uniform(key=jax.random.PRNGKey(1), shape=(10,)),
    )
    assert returned.shape == (10,)
    print(returned)


def test_DTransformerActivation(cfg):
    dta = model.DTransformerActivationLayer(d_mlp=cfg.d_mlp, d_e=cfg.d_e)
    key, subkey = jax.random.split(jax.random.PRNGKey(0))
    variables = dta.init(key, jnp.ones((cfg.l_x, cfg.d_e)))
    returned = dta.apply(
        variables, jax.random.uniform(key=subkey, shape=(cfg.l_x, cfg.d_e))
    )
    assert returned.shape == (cfg.l_x, cfg.d_e)
    print(returned)


def test_DTransformerBlock(cfg):
    dtb = model.DTransformerBlock(
        d_e=cfg.d_e,
        d_mlp=cfg.d_mlp,
        d_attn=cfg.d_attn,
        d_v=cfg.d_v,
        d_out=cfg.d_e,
        attn_heads=cfg.attn_heads,
        l_x=cfg.l_x,
    )
    key, subkey = jax.random.split(jax.random.PRNGKey(2))
    variables_dtb = dtb.init(key, jnp.ones((cfg.l_x, cfg.d_e)))
    returned = dtb.apply(variables_dtb, jax.random.uniform(subkey, (cfg.l_x, cfg.d_e)))
    assert returned.shape == (cfg.l_x, cfg.d_e)
    print(returned)


def test_TransformerEmbedding(test_path, cfg):
    te = model.DTransformerEmbedding(
        d_e=cfg.d_e,
        l_max=cfg.l_x,
        vocab_size=185,  # TODO: read from test_tokens.json?
    )
    tokens_file = test_path / Path("test_tokens.json")
    # TODO: fix Path vs string...
    tokenizer = own_tokenize.load_BPE_tokenizer(str(tokens_file))
    x = "Hi there!"
    x_ids = jnp.array(tokenizer.encode(x).ids, dtype=int)
    print(x_ids)
    print(type(x_ids))
    variables_te = te.init(jax.random.PRNGKey(0), jnp.ones(len(x_ids)))
    returned = te.apply(variables_te, x_ids)
    print(returned)


@pytest.fixture
def tokenizer(test_path):
    tokens_file = test_path / Path("test_tokens.json")
    # TODO: fix Path vs string...
    return own_tokenize.load_BPE_tokenizer(str(tokens_file))


@pytest.fixture
def dtransf(cfg):
    dtransformer = model.DTransformer(
        vocab_size=185,
        l_max=cfg.l_max,
        d_e=cfg.d_e,
        d_mlp=cfg.d_mlp,
        d_v=cfg.d_v,
        num_layers=4,
        attn_heads=6,
    )
    key, subkey = jax.random.split(jax.random.PRNGKey(5))
    variables = dtransformer.init(key, jax.random.normal(subkey, shape=(cfg.l_max,)))
    return (dtransformer, variables)


def test_DTransformer(cfg, tokenizer, dtransf):
    x = "Hey there!"
    y = "Then it went on"
    x_ids = jnp.array(tokenizer.encode(x).ids)
    y_ids = jnp.array(tokenizer.encode(y).ids)
    l_x = len(x_ids)
    l_y = len(y_ids)

    # need padding
    x_ids = jnp.pad(np.array(x_ids), (0, cfg.l_max - l_x), "constant")
    y_ids = jnp.pad(np.array(y_ids), (0, cfg.l_max - l_y), "constant")
    # print(x_ids)
    # print(y_ids)

    dtransformer, variables = dtransf
    # NOTE: apply wraps the call (but a direct call throws an error...)
    returned_x = dtransformer.apply(variables, x_ids)
    returned_y = dtransformer.apply(variables, y_ids)
    print(returned_x)
    print(returned_y)


def test_infer(tokenizer, dtransf):
    x = "He went on"
    y = "Going nowhere"

    dtransformer, variables = dtransf

    returned_x = dtransformer.infer(
        tokenizer=tokenizer,
        x=x,
        l_gen=4,
        variables=variables,
        temperature=1e-8,
    )
    returned_y = dtransformer.infer(
        tokenizer=tokenizer,
        x=y,
        l_gen=4,
        variables=variables,
        temperature=1e-8,
    )
    print(f"returned: {returned_x}")
    print(f"returned: {returned_y}")





