import jax
import jax.numpy as jnp
from ownGPT import config
from ownGPT import model

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