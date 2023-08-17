"""
TODO: 
- dependencies
- fix imports
- vmap/jit where possible (e.g. for loops)
- use logging
- type annotation
- consider batch dimension
- deduce the lengths as much as possible

"""
import logging
import math
from pathlib import Path
from typing import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tokenizers import Tokenizer


class Attention(nn.Module):
    d_attn: int  # "attention dimension"
    d_v: int  # dimension of value (d_mid or d_out in Phuong/Hutter)
    unidirectional: bool = True  # decides if it attention is unidirectional or not

    @nn.compact
    def __call__(self, x: jnp.ndarray, z: jnp.ndarray):
        """Compute attention for x, z.
        For concreteness assume that x, z are of shape (l_x, d_x), (l_z, d_z) where
        l_x, d_x is the token length, embedding dimension respectively and similarly
        for z. Typically, d_x = d_z = d_e (cf. below).

        Args:
            x: current token (embedded)
            z: context token (embedded)

        Returns:
            attn: attention of shape (l_x, d_v)
        """
        # NOTE: operators always left multiply with input
        # features = number of cols = output dimension
        l_x = x.shape[0]
        l_z = z.shape[0]
        query = nn.Dense(features=self.d_attn)(x)  # (l_x, d_attn)
        key = nn.Dense(features=self.d_attn)(z)  # (l_z, d_attn)
        value = nn.Dense(features=self.d_v)(z)  # (l_z, d_v)
        scores = query @ jnp.transpose(key)
        assert scores.shape == (l_x, l_z)

        # masking
        if self.unidirectional:
            mask = jnp.tril(jnp.ones(scores.shape))
            scores = jnp.where(mask, scores, -jnp.inf)

        attn = jax.nn.softmax(scores / math.sqrt(self.d_attn)) @ value
        assert attn.shape == (l_x, self.d_v)
        return attn


class MHAttention(nn.Module):
    d_attn: int  # "attention dimension"
    d_v: int  # dimension of value (d_mid or d_out in Phuong/Hutter)
    d_out: int  # output dimension
    attn_heads: int  # number of heads

    def setup(self):
        # TODO: ordering of heads unimportant?
        self.heads = [
            Attention(d_attn=self.d_attn, d_v=self.d_v, unidirectional=True)
            for _ in range(self.attn_heads)
        ]
        self.w_out = nn.Dense(features=self.d_out)

    def __call__(self, x, z):
        l_x = x.shape[-2]
        y = jnp.concatenate(
            [head(x, z) for head in self.heads], axis=-1
        )  # TODO: too inefficient?
        assert y.shape == (l_x, self.attn_heads * self.d_v)
        out = self.w_out(y)
        assert out.shape == (l_x, self.d_out)
        return self.w_out(y)


class DTransformerActivationLayer(nn.Module):
    d_mlp: int  # "MLP dimension", see below (MLP: multi-layer perceptrons)
    d_e: int  # embedding dimension
    act_fn: Callable = jax.nn.gelu  # activation function

    def setup(self):
        self.mlp1 = nn.Dense(features=self.d_mlp)
        self.mlp2 = nn.Dense(features=self.d_e)

    def __call__(self, x):
        return x + self.mlp2(self.act_fn(self.mlp1(x)))


class LayerNormalization(nn.Module):
    """Alternatively, use nn.LayerNorm."""

    offset: bool = True  # decides if we include a learnable offset/bias

    @nn.compact
    def __call__(self, e):
        @jax.jit
        def normalization(e):
            # normalize along last axis
            mean = jnp.mean(e, axis=-1, keepdims=True)
            scale = jnp.std(e, axis=-1, keepdims=True)
            # TODO: do not treat scale == 0 here to make jit work
            return (e - mean) / scale

        # learnable parameters
        # TODO: is normal initialization a good choice here?
        gamma = self.param("gamma", nn.initializers.normal(), e.shape)
        if self.offset:
            beta = self.param("beta", nn.initializers.normal(), e.shape)
        return gamma * (normalization(e)) + beta


class DTransformerBlock(nn.Module):
    """See Algorithm 10 (DTransformer) in Hutter/Phuong"""

    d_e: int  # word embedding dimension
    d_mlp: int  # output dimension of "activation layers"
    d_attn: int  # embedding dimension of key and query ("attention dimension")
    d_v: int  # dimension of value (d_mid or d_out in Phuong/Hutter)
    d_out: int  # output dimension
    attn_heads: int  # number of heads
    bias_normalization: bool = True

    def setup(self):
        self.act_layer = DTransformerActivationLayer(d_mlp=self.d_mlp, d_e=self.d_e)
        self.layer_norm1 = (
            LayerNormalization()
        )  # nn.LayerNorm(epsilon=1e-6, use_bias=self.bias_normalization) #
        self.layer_norm2 = (
            LayerNormalization()
        )  # nn.LayerNorm(epsilon=1e-6, use_bias=self.bias_normalization)
        self.mhattention = MHAttention(
            d_attn=self.d_attn,
            d_v=self.d_v,
            d_out=self.d_out,
            attn_heads=self.attn_heads,
        )

    def __call__(self, x):
        # assume that we work with batches
        assert len(x.shape) == 3, "Require batches"
        batch_size, l_x, d_x = x.shape

        x_normalized = jax.vmap(self.layer_norm1)(x)
        # use vmap to deal with batches (along first axis)
        x = x + jax.vmap(self.mhattention)(x_normalized, x_normalized)
        assert x.shape == (batch_size, l_x, self.d_out)
        return x + self.act_layer(jax.vmap(self.layer_norm2)(x))


class DTransformerEmbedding(nn.Module):
    """
    Class to preprocess strings before feeding them into a DTransformer model
    (Algorithm 10 in [PH])
    """

    d_e: int  # word/positional embedding dimension
    l_max: int  # maximal token length
    vocab_size: int

    def setup(self):
        self.word_embed = nn.Embed(num_embeddings=self.vocab_size, features=self.d_e)
        self.pos_embed = nn.Embed(num_embeddings=self.l_max, features=self.d_e)

    def __call__(self, x: jnp.ndarray):
        assert (
            self.l_max >= x.shape[1]
        ), f"Provided token of length {x.shape[1]} exceeds maximal token length {self.l_max}."
        x = x.astype(int)
        batch_size, l_x = x.shape
        x_wembed = self.word_embed(x)
        positions = jnp.arange(0, l_x)
        x_pembed = self.pos_embed(positions)

        embeddings = x_wembed + x_pembed
        assert (batch_size, l_x, self.d_e) == embeddings.shape
        return embeddings


@dataclass
class DTransformerConfig:
    d_e: int  # embedding dimension
    # d_mlp: int  # output dimension of activation layers, typically d_mlp = d_e
    vocab_size: int
    l_max: int  # maximal context/token length
    num_layers: int
    attn_heads: int  # number of attention heads
    bias_normalization: bool=True

    """The following condition is required in many GPT implementations and is baked into the 
    implementation of Attention. However, it is made clear from Algorithm 4 [PH] (Attention) 
    that this is not necessary. We still impose it here to save one parameter and to match
    the dimensions to the common implementations.
    
    Note that we have d_e = d_v*attn_heads and d_v is often referred to as head_size.
    """
    def __post_init__(self):
        assert (
            self.d_e % self.attn_heads == 0
        ), "Embedding dimension has to be divisible by number of attention heads."

    def d_v(self):
        return int(self.d_e / self.attn_heads)


class DTransformer(nn.Module):
    config: DTransformerConfig

    def setup(self):
        self.dtransformer_embed = DTransformerEmbedding(
            d_e=self.config.d_e, l_max=self.config.l_max, vocab_size=self.config.vocab_size
        )
        self.layers = [
            DTransformerBlock(
                d_e=self.config.d_e,
                d_mlp=self.config.d_e, # common choice
                d_attn=self.config.d_e, # common choice
                d_v=self.config.d_v(), 
                d_out=self.config.d_e,
                attn_heads=self.config.attn_heads,
            )
            for _ in range(self.config.num_layers)
        ]
        self.final_layer_norm = (
            LayerNormalization()
        )  # nn.LayerNorm(epsilon=1e-6, use_bias=self.bias_normalization)
        self.unembed_lin_layer = nn.Dense(features=self.config.vocab_size, use_bias=False)

    def __call__(self, x: jnp.ndarray):
        batch_size, l_x = x.shape
        x = self.dtransformer_embed(x)
        assert x.shape == (batch_size, l_x, self.config.d_e)
        for _, layer in enumerate(self.layers):
            x = layer(x)
        x = jax.vmap(self.final_layer_norm)(x)
        # unembedding (Algorithm 7 in [PH])
        out = jax.nn.softmax(self.unembed_lin_layer(x))
        assert out.shape == (batch_size, l_x, self.config.vocab_size)
        return out

    def infer(
        self,
        tokenizer: Tokenizer,
        x: str,
        l_gen: int,
        variables,
        temperature: float = 1.0,
    ):
        x_ids = jnp.array(tokenizer.encode(x).ids, dtype=int)
        vocab_size = tokenizer.get_vocab_size()
        len_x_ids = len(x_ids)

        # pad/slice x_ids so that it is of length l_max
        # TODO: really needed?
        l_max = self.config.l_max
        if l_max < len_x_ids:
            y_ids = x_ids[len_x_ids - l_max :]
            start_idx = l_max
        else:
            y_ids = jnp.pad(x_ids, (0, l_max - len_x_ids))
            start_idx = len_x_ids
        assert y_ids.size == l_max
        y_ids = jnp.reshape(y_ids, newshape=(1, l_max))

        # the following array is mostly needed if we cannot
        # store all generated tokens in x_ids
        generated_tokens = jnp.zeros(l_gen, dtype=int)

        key = jax.random.PRNGKey(23)
        # P = self.apply(variables, y_ids)

        for i in range(l_gen):
            P = self.apply(variables, y_ids)
            # dealing with indices
            if start_idx + i >= l_max - 1:
                p_idx = l_max - 1
            else:
                p_idx = start_idx + i

            # create new random key (otherwise always get same sample)
            batch_size, l, v = P.shape
            assert (batch_size, l, v) == (1, l_max, self.config.vocab_size)
            P = jnp.reshape(P, newshape=P.shape[1:])
            p = jax.nn.softmax(P[p_idx, :] / temperature)
            # print(p)
            key, _ = jax.random.split(key)
            new_token = jax.random.choice(key=key, a=self.config.vocab_size, p=p)
            # print(f"max index: {jnp.argmax(P[p_idx, :])}, p_idx: {p_idx}")

            if p_idx >= l_max - 1:
                # TODO: fix that...
                z_ids = jnp.append(y_ids, new_token)
                y_ids = y_ids.at[:].set(z_ids[1:])
            else:
                y_ids = y_ids.at[p_idx].set(new_token)

            # print(y_ids)
            generated_tokens = generated_tokens.at[i].set(new_token)

        return tokenizer.decode(generated_tokens)
