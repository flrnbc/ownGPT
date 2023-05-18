"""
TODO: 
- dependencies
- jit where possible (e.g. for loops)
- use logging

"""
from flax import linen as nn 
import jax
import jax.numpy as jnp
from jax.nn import gelu
import math
import numpy as np
from typing import Callable
from ownGPT import own_tokenize, config
from tokenizers import Tokenizer
from pathlib import Path

# TODO: type annotation?!
# TODO: refactor: make classes independent of config (pass as parameters)

class Attention(nn.Module):
    d_attn: int     # embedding dimension of key and query ("attention dimension")
    d_v: int        # dimension of value (d_mid or d_out in Phuong/Hutter)

    @nn.compact
    def __call__(self, x: jnp.array, z: jnp.array):
        """ Compute attention for x, z.
        For concreteness assume that x, z are of shape (l_x, d_x), (l_z, d_z) where
        l_x, d_x is the token length, embedding dimension respectively and similarly
        for z.

        Args: 
            x: current (embedded) token
            z: (embedded) context token

        Returns:
            attn: attention of shape (l_x, d_v)
        """ 
        # operator of shape (d_x, d_attn) (features = number of cols)
        # query of shape (l_x, d_attn)
        # NOTE: always left multiply with input
        query = nn.Dense(features = self.d_attn)(x) # NOTE: contains bias by default

        # operator of shape (d_z, d_attn)
        # key of shape (l_z, d_attn)
        key = nn.Dense(features = self.d_attn)(z)

        # operator of shape (d_z, d_v)
        # value (l_z, d_v)
        value = nn.Dense(features = self.d_v)(z) # TODO: are these different operators? and are they "persistent" for training?
        
        # scores of shape (l_x, l_z)
        scores = query @ jnp.transpose(key)

        # TODO: masking

        attn = jax.nn.softmax(scores/math.sqrt(self.d_attn)) @ value 
        return attn


class MHAttention(nn.Module):
    d_attn: int         # embedding dimension of key and query ("attention dimension") 
    d_v: int            # dimension of value (d_mid or d_out in Phuong/Hutter)
    d_out: int          # output dimension
    attn_heads: int     # number of heads
    l_x: int            # token length 
                        # TODO: here necessary?

    def setup(self):
        self.attns = [Attention(d_attn=self.d_attn, d_v=self.d_v) for _ in range(self.attn_heads)]
        self.w_out = nn.Dense(features=self.d_out)

    def __call__(self, x, z):
        y = jnp.zeros((self.l_x, self.attn_heads*self.d_v))
        for idx, attn in enumerate(self.attns):
            # TODO: how to do this better? E.g. using jit?!?
            y = y.at[: , idx: idx+self.d_v].set(attn(x, z))
        w = self.w_out(y) #nn.Dense(features=config.attn_heads*config.d_mid)(y)
        return w


class DTransformerActivationLayer(nn.Module):
    d_mlp: int                          # "MLP dimension", see below (MLP: multi-layer perceptrons)
    d_e: int                            # embedding dimension
    act_fn: Callable=jax.nn.gelu        # activation function

    def setup(self):
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

    #def setup(self):
    #    self.gamma = self.param('gamma', nn.initializers.normal(), e.shape)

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
    d_attn: int         # embedding dimension of key and query ("attention dimension") 
    d_v: int            # dimension of value (d_mid or d_out in Phuong/Hutter)
    d_out: int          # output dimension; 
                        # TODO: typically the embedding dimension?!
    attn_heads: int     # number of heads
    l_x: int            # token length TODO: necessary?

    def setup(self):
        self.act_layer=DTransformerActivationLayer(
            d_mlp=self.d_mlp,
            d_e=self.d_e
        )
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()
        # TODO: is there some sort of "Makefile auto-completion"?
        self.mhattention = MHAttention(
            d_attn=self.d_attn, 
            d_v=self.d_v, 
            d_out=self.d_out, 
            attn_heads=self.attn_heads, 
            l_x=self.l_x
        )

    def __call__(self, x):
        """
        Input: 
            - x: typically a 2D arrays whose rows are the word + positional embeddings
                of the tokens
        TODO: check for dimension?
        """
        num_rows = x.shape[0]
        #num_cols = x.shape[1]
        x_normalized = jnp.zeros(x.shape)
        # TODO: extract this part or do in matrix form?
        for i in range(num_rows):
            # set i-th row to layer normalized i-th row of x
            x_normalized = x_normalized.at[i,:].set(self.layer_norm1(x[i,:]))
        # use x_normalized for queries, keys and values
        x_mhattn = self.mhattention(x_normalized, x_normalized)        
        x = x.at[:,:].add(x + x_mhattn)

        for i in range(num_rows):
           x_normalized = x_normalized.at[i,:].set(self.layer_norm2(x[i,:]))
        x = x.at[:,:].add(self.act_layer(x_normalized)) 

        return x 


class DTransformerEmbedding(nn.Module):
    """
    Class to preprocess strings before feeding them into a Transformer model.
    """
    d_e: int            # word embedding dimension
    l_max : int
    vocab_size: int     # TODO: can read from tokens_file?!?

    def setup(self):
        self.word_embed     = nn.Embed(num_embeddings=self.vocab_size, features=self.d_e)
        # TODO: add assert l_max >= length(x)?
        self.pos_embed      = nn.Embed(num_embeddings=self.l_max, features=self.d_e)

    def __call__(self, x: jnp.array):
        # TODO: actually assume that x is a jnp.array with int entries
        X = jnp.zeros(shape=(x.size, self.d_e))
        x_int = x.astype(int)   # just to be sure 
                                # TODO: need int32?

        # note that we embed all possible positions even though len(x) might be less
        x_enum = jnp.array([i for i in range(self.l_max)], dtype=int)
        x_pembed = self.pos_embed(x_enum)
        x_wembed = self.word_embed(x_int)

        # NOTE: the following line does not work because the shapes might be differnt (unless len(x)=l_max) 
        # X = X.at[:,:].add(x_wembed + x_pembed)

        for i in range(x.size):
        #    print(type(x))
        #    print(type(x[i]))
            X = X.at[i,:].set(x_wembed[i,:] + x_pembed[i,:])
        
        return X


class TransformerUnembedding(nn.Module):
    """Turn result of Transformer blocks into probabilities"""
    vocab_size: int

    def setup(self):
        self.lin_layer = nn.Dense(features=self.vocab_size)

    def __call__(self, x: jnp.array):
        return jax.nn.softmax(self.lin_layer(x))


class DTransformer(nn.Module):
    vocab_size: int     # vocabulary size
                        # TODO: determined by tokenizer?
    l_gen: int          # max generated sequence
    l_max: int          # max sequence length
    d_e: int            # word embedding dimension
    d_v: int
    d_mlp: int          # output dimension of "activation layers"
    num_layers: int     # number of layers
    attn_heads: int     # number of attention heads in each layer
    #dtransformer_block: DTransformerBlock

    def setup(self):
        # TODO: add assert l_max >= l_x
        self.dtransformer_embed = DTransformerEmbedding(
            d_e=self.d_e, 
            l_max=self.l_max, 
            vocab_size=self.vocab_size
        )
        self.layers = [ 
            DTransformerBlock(
                d_e=self.d_e,
                d_mlp=self.d_mlp,
                d_attn=self.d_e,
                d_v=self.d_v,
                d_out=self.d_e, # that's at least typical...
                attn_heads=self.attn_heads,
                l_x=self.l_max)
            for _ in range(self.num_layers) 
        ]
        self.final_layer_norm = LayerNormalization()
        self.unembed   = TransformerUnembedding(vocab_size=self.vocab_size)


    def __call__(self, X: jnp.array):
        #X = X.at[:,:].set(self.dtransformer_embed(X))
        # TODO: clean up!
        Y = self.dtransformer_embed(X)
        for _, layer  in enumerate(self.layers):
            Y = Y.at[:,:].set(layer(Y))

        num_rows = Y.shape[0]
        Y_normalized = jnp.zeros(Y.shape)
        for i in range(num_rows):
            # TODO: refactor to matrix form?!
            Y_normalized = Y_normalized.at[i,:].set(self.final_layer_norm(Y[i,:]))

        return self.unembed(Y_normalized)

    
    def infer(self, tokenizer: Tokenizer, x: str, variables):
        x_ids = jnp.array(tokenizer.encode(x).ids, dtype=int)
        # TODO: in Hutter/Phuong this call is in the following loop. But why?!
        len_x_ids = len(x_ids)
        l_max = self.l_max
        if self.l_max < len_x_ids:
            x_ids = x_ids[len_x_ids-l_max:]
        else:
            x_ids = jnp.pad(x_ids, (0, self.l_max-len_x_ids)) 
        #x_ids = jnp.pad(x_ids_unpadded, (0, self.l_max-len_x_ids))
        #y = jnp.zeros(shape=(len_x_ids + self.l_gen,))
        #y = y.at[:len_x_ids].set(x_ids)
        #variables = self.init(jax.random.PRNGKey(0), x_ids)
        #P = jnp.zeros((len_x_ids, self.vocab_size)).astype("float64")

        key = jax.random.PRNGKey(23)
        for i in range(self.l_gen):
            # TODO: totally unclear: don't we "forget" our learnt parameters if we repeat the init step?
            #variables = self.init(jax.random.PRNGKey(0), x_ids)
            P = self.apply(variables, x_ids) 
            # create new random key (otherwise always get same sample)
            key, _ = jax.random.split(key) 
            new_token = jax.random.choice(
                    key=key,
                    a=self.vocab_size, 
                    p=P[len_x_ids+i,:]) #P[len_x_ids+i,:])
            try:
                x_ids = x_ids.at[len_x_ids+i].set(new_token)
            except:
                # TODO: this part does not yet seem to work as intended
                # TODO: also: save new tokens somewhere else as well to avoid filling
                # x_ids if l_gen > l_max?
                x_ids = jnp.append(x_ids, new_token)
                x_ids = x_ids[1:]
            #x_ids = x_ids.at[len_x_ids+i:].set(new_token)
            print(x_ids)
            # create new random key (otherwise always get same sample)
            key, _ = jax.random.split(key)

        return tokenizer.decode(x_ids[len_x_ids:len_x_ids+self.l_gen])

