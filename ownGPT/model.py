"""
TODO: 
- dependencies

"""
from flax import linen as nn 
import jax
import jax.numpy as jnp
from jax.nn import gelu
import math
from typing import Callable

# TODO: type annotation?!
# TODO: refactor: make classes independent of config (pass as parameters)

class Attention(nn.Module):
    d_attn: int     # embedding dimension of key and query ("attention dimension")
    d_v: int        # dimension of value (d_mid or d_out in Phuong/Hutter)

    @nn.compact
    def __call__(self, x, z):
        """ Compute attention for x, z.
        Think of x as the current token and z the context token.

        Input: 
        TODO: types?!?
            - x, z: in the following assumed to be of shape (l_x, d_x), (l_z, d_z)
            
        Output:
            - y: attention of shape (l_x, d_v)
        """ 
        # QUERY
        # l_x: typically token length
        # input x: (l_x, d_x)
        # operator: (d_x, d_attn) (features = number of cols)
        # output q: (l_x, d_attn) (left mult with input x)
        q = nn.Dense(features = self.d_attn)(x) # NOTE: contains bias by default

        # KEY
        # l_z: typically token length
        # input z (l_z, d_z)
        # operator (d_z, d_attn)
        # k (l_z, d_attn) (left mult with input z)
        k = nn.Dense(features = self.d_attn)(z)

        # VALUE
        # input z (l_z, d_z)
        # operator (d_z, d_v)
        # v (l_z, d_v)
        v = nn.Dense(features = self.d_v)(z) # TODO: are these different operators? and are they "persistent" for training?
        
        # scores (l_x, l_z)
        s = q @ jnp.transpose(k)

        # TODO: masking

        # attention (l_x, d_v)
        y = jax.nn.softmax(s/math.sqrt(self.d_attn)) @ v 
        return y


class MHAttention(nn.Module):
    d_attn: int         # embedding dimension of key and query ("attention dimension") 
    d_v: int            # dimension of value (d_mid or d_out in Phuong/Hutter)
    d_out: int          # output dimension
    attn_heads: int     # number of heads
    l_x: int            # token length TODO: necessary?

    def setup(self):
        self.attns = [Attention(d_attn=self.d_attn, d_v=self.d_v) for idx in range(self.attn_heads)]
        self.w_out = nn.Dense(features=self.d_out)

    def __call__(self, x, z):
        y = jnp.zeros((self.l_x, self.attn_heads*self.d_v))
        for idx, attn in enumerate(self.attns):
            # TODO: how to do this better?
            y = y.at[: , idx: idx+self.d_v].set(attn(x, z))
        w = self.w_out(y) #nn.Dense(features=config.attn_heads*config.d_mid)(y) # TODO: check: mult from right?
        return w


class DTransformerActivationLayer(nn.Module):
    d_mlp: int                          # "MLP dimension", see below (MLP: multi-layer perceptrons)
    d_e: int                            # embedding dimension
    act_fn: Callable=jax.nn.gelu        # activation function (TODO: typing not very useful here...)

    def setup(self):
        # just for convenience
        self.mlp1 = nn.Dense(features=self.d_mlp)
        self.mlp2 = nn.Dense(features=self.d_e)

    def __call__(self, x):
        """
        Input: 
            - x of shape, say (l, d_e)

        Output:
            - 'activated' x of shape (l, d_e)
        """
        return x + self.mlp2(self.act_fn(self.mlp1(x)))


class LayerNormalization(nn.Module):
    offset: bool=True                   # decides if we include a learnable offset/bias
    translation_by_mean: bool=True      # decides if we translate the input by its (broadcasted) mean
    # NOTE: if both are false, one speaks of root mean square layer normalization
    # TODO: what about row-wise layer normalization?

    @nn.compact
    def __call__(self, e):
        mean = 1
        if self.translation_by_mean:
            mean = jnp.mean(e)
        scale = jnp.std(e)
        if scale == 0:
            scale = 1
        
        # learnable parameters
        # (found it more transparent to spell this out)
        # TODO: is normal initialization a good choice here?
        gamma = self.param('gamma', nn.initializers.normal(), e.shape)
        if self.offset:
            beta = self.param('beta', nn.initializers.normal(), e.shape)

        return gamma*((e-mean)/scale) + beta


class DTransformerBlock(nn.Module):
    # TODO: how to make this cleaner? need so many attributes? many create MHAttention from the other params?
    d_e: int            # word embedding dimension
    d_mlp: int          # output dimension of "activation layers"
    mhattention: MHAttention
    act_layer: DTransformerActivationLayer 
    layer_norm1: LayerNormalization 
    layer_norm2: LayerNormalization
 
    @nn.compact
    def __call__(self, x):
        """
        Input: 
            - x: typically a 2D arrays whose rows are the word + positional embeddings
                of the tokens
        TODO: check for dimension?
        """
        num_rows = x.shape[0]
        num_cols = x.shape[1]
        x_normalized = jnp.zeros(x.shape)
        for i in range(num_rows):
            # TODO: can't we do that in matrix form?
            # set i-th row to layer normalized i-th row of x
            x_normalized = x_normalized.at[i:].set(self.layer_norm1(x[i:]))
            # use x_normalized for queries, keys and values        
        x = x.at[:].add(x + self.mhattention(x_normalized, x_normalized))

        for i in range(num_rows):
           x_normalized = x_normalized.at[i:].set(self.layer_norm2(x[i:]))
        x = x.at[:].add(self.act_layer(x_normalized)) 

        return x 

        
class DTransformer(nn.Module):
    l_max: int          # max sequence length
    d_e: int            # word embedding dimension
    d_mlp: int          # output dimension of "activation layers"
    num_layers: int     # number of layers
    attn_heads: int     # number of attention heads in each layer

    def setup(self):
        self.activation_layer = DTransformerActivationLayer(d_mlp=self.d_mlp, d_e=self.d_e)



