"""
TODO: 
- dependencies

"""
from ownGPT import config
from flax import linen as nn 
import jax
import jax.numpy as jnp
import math

# TODO: type annotation?!
# TODO: refactor: make classes independent of config (pass as parameters)

class Attention(nn.Module):
    # embedding dimension of key and query ("attention dimension")
    d_attn: int
    # dimension of value (d_mid or d_out in Phuong/Hutter)
    d_v: int

    @nn.compact
    def __call__(self, x, z):
        # query
        # l_x: typically token length
        # input x: (l_x, d_x)
        # operator: (d_x, d_attn) (features = number of cols)
        # output q: (l_x, d_attn) (left mult with input x)
        q = nn.Dense(features = self.d_attn)(x) # TODO: with bias?

        # key
        # l_z: typically token length
        # input z (l_z, d_z)
        # operator (d_z, d_attn)
        # k (l_z, d_attn) (left mult with input z)
        k = nn.Dense(features = self.d_attn)(z)

        # value
        # input z (l_z, d_z)
        # operator (d_z, d_v)
        # v (l_z, d_v)
        v = nn.Dense(features = self.d_v)(z) # TODO: are these different operators? and are they "persistent" for training?
        
        # scores (l_x, l_z)
        s = q @ jnp.transpose(k)

        # TODO: masking

        # attention (l_x, d_v)
        y = jax.nn.softmax(s/math.sqrt(config.d_attn)) @ v 
        return y


class MHAttention(nn.Module):
    # embedding dimension of key and query ("attention dimension")
    d_attn: int
    # dimension of value (d_mid or d_out in Phuong/Hutter)
    d_v: int
    # output dimension
    d_out: int
    # number of heads
    attn_heads: int
    # token length
    # TODO: necessary?
    l_x: int

    def setup(self):
        self.attns = [Attention(d_attn=self.d_attn, d_v=self.d_v) for idx in range(self.attn_heads)]
        self.w_out = nn.Dense(features=self.d_out)

    def __call__(self, x, z):
        y = jnp.zeros((self.l_x, self.attn_heads*self.d_v))
        for idx, attn in enumerate(self.attns):
            # TODO: how to do this better?
            y = y.at[: , idx: idx+config.d_v].set(attn(x, z))
        w = self.w_out(y) #nn.Dense(features=config.attn_heads*config.d_mid)(y) # TODO: check: mult from right?
        return w

 