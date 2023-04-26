"""
TODO: 
- dependencies

"""
import config
from flax import linen as nn 
import jax
import jax.numpy as jnp
import math

# TODO: type annotation?!
# TODO: refactor: make classes independent of config (pass as parameters)

class Attention(nn.Module):
    @nn.compact
    def __call__(self, x, z):
        # assume x: (l_x, d_x), z: (l_z, d_z)
        # TODO: add assert?

        # query
        # operator: (d_x, d_attn) (features = number of cols)
        # q: (l_x, d_attn) (left mult with x apparently) TODO: check!
        q = nn.Dense(features = config.d_attn)(x) # TODO: with bias?

        # key
        # operator: (d_z, d_attn)
        # k: (l_z, d_attn)
        k = nn.Dense(features = config.d_attn)(z)

        # value
        # operator: (d_z, d_mid) (or d_out)
        # v: (l_z, d_out)
        v = nn.Dense(features = config.d_mid)(z) # TODO: are these different operators? and are they "persistent" for training?
        
        # score
        s = q @ jnp.transpose(k)

        # TODO: masking

        # attention: (l_x, d_mid)
        y = jax.nn.softmax(s/math.sqrt(config.d_attn)) @ v 
        return y

att = Attention()
variables2 = att.init(jax.random.PRNGKey(0),  jnp.ones((config.l_x, config.d_x)), jnp.ones((config.l_z, config.d_z)))
print(att.apply(variables2,  jnp.ones((config.l_x, config.d_x)), jnp.ones((config.l_z, config.d_z))).shape)
#print(att.tabulate(jax.random.PRNGKey(0), jnp.ones((64, 64)), jnp.ones((64, 64))))

class MHAttention(nn.Module):
    def setup(self):
        self.attns = [Attention() for idx in range(config.attn_heads)]

    def __call__(self, x, z):
        y = jnp.zeros((config.attn_heads*config.d_mid, config.l_x))
        for idx, attn in enumerate(self.attns):
            y = y.at[: , idx: idx+config.d_mid].set(attn(x, z))
        w = nn.Dense(features=config.attn_heads*config.d_mid)(y) # TODO: check: mult from right?
        return w

mattn = MHAttention()
#variables = mattn.init(jax.random.PRNGKey(0), jnp.ones((config.l_x, config.d_x)), jnp.ones((config.l_z, config.d_z)))
#print(mattn.apply(variables, jnp.ones(config.l_x, config.d_x), jnp.ones(config.l_z, config.d_z)).shape)
 