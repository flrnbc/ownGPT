"""
TODO: 
- dependencies

"""
import config
from flax import linen as nn 
import jax
import jax.numpy as jnp
import math


class Attention(nn.Module):
    @nn.compact
    def __call__(self, x, z):
        q = nn.Dense(features = config.l_x)(x)
        k = nn.Dense(features = config.l_z)(z)
        v = nn.Dense(features = config.l_z)(z) # TODO: are these different operators?
        s = q @ jnp.transpose(k)
        # TODO: masking
        w = jax.nn.softmax(s/math.sqrt(config.d_attn)) @ v 
        return w

att = Attention()
print(att.tabulate(jax.random.PRNGKey(0), jnp.ones((64, 64)), jnp.ones((64, 64))))



 