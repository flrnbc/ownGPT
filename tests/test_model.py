import jax
import jax.numpy as jnp
from ownGPT import config, model

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
    mattn = model.MHAttention(d_attn=config.d_attn, d_v=config.d_v, d_out=config.d_out, attn_heads=config.attn_heads, l_x=config.l_x)
    #variables_mattn = mattn.init(jax.random.PRNGKey(0), jnp.ones((config.l_x, config.d_x)), jnp.ones((config.l_x, config.d_x)))
    # output shape: (l_x, d_out)
    dta = model.DTransformerActivationLayer(d_mlp=config.d_mlp, d_e=config.d_out)
    #variables_dta = dta.init(jax.random.PRNGKey(0), jnp.ones((config.l_x, config.d_out)))
    # output shape: (l_x, d_out)
    ln1 = model.LayerNormalization()
    #variables_ln1 = ln1.init(jax.random.PRNGKey(0), jnp.ones(config.d_out,))
    ln2 = model.LayerNormalization()
    #variables_ln1 = ln2.init(jax.random.PRNGKey(0), jnp.ones(config.d_out,))

    dtb  = model.DTransformerBlock(
            d_e=config.d_e, 
            d_mlp=config.d_mlp,
            mhattention=mattn,
            act_layer=dta,
            layer_norm1=ln1,
            layer_norm2=ln2
    )
    variables_dtb = dtb.init(jax.random.PRNGKey(0), jnp.ones((config.l_x, config.d_x))) 
    returned = dtb.apply(variables_dtb, jnp.ones((config.l_x, config.d_x)))
    assert returned.shape == (config.l_x, config.d_x)