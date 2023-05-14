import jax
import jax.numpy as jnp
import numpy as np
from ownGPT import config, own_tokenize, model
from pathlib import Path

# TODO: unify some tests?

def test_shapes_attention():
    att = model.Attention(d_attn=config.d_attn, d_v=config.d_v)
    variables = att.init(jax.random.PRNGKey(0),  jnp.ones((config.l_x, config.d_x)), jnp.ones((config.l_z, config.d_z)))
    output_shape = att.apply(variables,  jnp.ones((config.l_x, config.d_x)), jnp.ones((config.l_z, config.d_z))).shape
    assert output_shape == (config.l_x, config.d_v)


def test_shapes_multi_head_attention():
    mattn = model.MHAttention(d_attn=config.d_attn, d_v=config.d_v, d_out=config.d_out, attn_heads=config.attn_heads, l_x=config.l_x)
    variables = mattn.init(jax.random.PRNGKey(0), jnp.ones((config.l_x, config.d_x)), jnp.ones((config.l_z, config.d_z)))
    output_shape = mattn.apply(variables, jnp.ones((config.l_x, config.d_x)), jnp.ones((config.l_z, config.d_z))).shape
    assert output_shape == (config.l_x, config.d_out)


def test_linear_normalization():
    # TODO: currently just some sort of a smoke test...
    ln = model.LayerNormalization()
    variables = ln.init(jax.random.PRNGKey(0), jnp.ones(10,))
    returned = ln.apply(variables, jnp.ones(10,))
    assert returned.shape == (10,)


def test_DTransformerActivation():
    dta = model.DTransformerActivationLayer(d_mlp=config.d_mlp, d_e=config.d_e)
    variables = dta.init(jax.random.PRNGKey(0), jnp.ones((config.l_x, config.d_e)))
    returned = dta.apply(variables, jnp.ones((config.l_x, config.d_e)))
    assert returned.shape == (config.l_x, config.d_e)


def test_DTransformerBlock():
    #mattn = model.MHAttention(d_attn=config.d_attn, d_v=config.d_v, d_out=config.d_out, attn_heads=config.attn_heads, l_x=config.l_x)
    #variables_mattn = mattn.init(jax.random.PRNGKey(0), jnp.ones((config.l_x, config.d_x)), jnp.ones((config.l_x, config.d_x)))
    # output shape: (l_x, d_out)
    #dta = model.DTransformerActivationLayer(d_mlp=config.d_mlp, d_e=config.d_out)
    #variables_dta = dta.init(jax.random.PRNGKey(0), jnp.ones((config.l_x, config.d_out)))
    # output shape: (l_x, d_out)
    #ln1 = model.LayerNormalization()
    #variables_ln1 = ln1.init(jax.random.PRNGKey(0), jnp.ones(config.d_out,))
    #ln2 = model.LayerNormalization()
    #variables_ln1 = ln2.init(jax.random.PRNGKey(0), jnp.ones(config.d_out,))

    dtb  = model.DTransformerBlock(
            d_e=config.d_e, 
            d_mlp=config.d_mlp,
            d_attn=config.d_attn,
            d_v=config.d_v,
            d_out=config.d_e,
            attn_heads=config.attn_heads,
            l_x=config.l_x
    )
    variables_dtb = dtb.init(jax.random.PRNGKey(0), jnp.ones((config.l_x, config.d_e))) 
    returned = dtb.apply(variables_dtb, jnp.ones((config.l_x, config.d_e)))
    assert returned.shape == (config.l_x, config.d_e)


def test_TransformerEmbedding(test_path):
    te = model.DTransformerEmbedding(
        d_e = config.d_e,
        l_max = config.l_x, 
        vocab_size=185 # TODO: read from test_tokens.json?
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


def test_DTransformer(test_path):
    te = model.DTransformerEmbedding(
        d_e = config.d_e,
        l_max = config.l_max, 
        vocab_size=185 # TODO: read from test_tokens.json?
    )
    tokens_file = test_path / Path("test_tokens.json")
    # TODO: fix Path vs string...
    tokenizer = own_tokenize.load_BPE_tokenizer(str(tokens_file))
    x = "Hi there! Yay"
    x_ids = tokenizer.encode(x).ids
    l_x = len(x_ids)
    # TODO: is padding correct here?
    x_ids = jnp.pad(np.array(x_ids), (0, 0), "constant") #config.l_max - l_x), "constant")
    print(x_ids)
    print(type(x_ids))
    #variables_te = te.init(jax.random.PRNGKey(0), jnp.ones(len(x_ids)))
    #X = te.apply(variables_te, x_ids)
    #print(f"shape of X: {X.shape}")
    #variables_dtb = dtb.init(jax.random.PRNGKey(0), jnp.ones((config.l_x, config.d_e))) #jnp.ones((config.l_x, config.d_e))) 
    dtransfomer = model.DTransformer(
        vocab_size=185,
        l_gen=3,
        l_max=config.l_max,
        l_x=l_x,
        d_e=config.d_e,
        d_mlp=config.d_mlp,
        d_v=config.d_v,
        num_layers=1,
        attn_heads=6,        
    )
    variables_dtransformer = dtransfomer.init(jax.random.PRNGKey(0), x_ids) #jnp.ones((config.l_x, config.d_e)))
    returned = dtransfomer.apply(variables_dtransformer, x_ids) # NOTE: apply wraps the call (but a direct call throws an error...)
    print(returned)

    # TODO: split tests
    returned = dtransfomer.infer(tokenizer=tokenizer, x=x, variables=variables_dtransformer)
    print(f"returned: {returned}")


