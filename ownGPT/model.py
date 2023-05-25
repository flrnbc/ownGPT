"""
TODO: 
- dependencies
- vmap/jit where possible (e.g. for loops)
- use logging
- type annotation

"""
import logging
import math
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from flax import linen as nn
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from jax.nn import gelu
from tokenizers import Tokenizer

from ownGPT import config, own_tokenize


class Attention(nn.Module):
    d_attn: int  # embedding dimension of key and query ("attention dimension")
    d_v: int  # dimension of value (d_mid or d_out in Phuong/Hutter)
    unidirectional: bool = True  # decides if it attention is unidirectional or not

    @nn.compact
    def __call__(self, x: jnp.ndarray, z: jnp.ndarray):
        """Compute attention for x, z.
        For concreteness assume that x, z are of shape (l_x, d_x), (l_z, d_z) where
        l_x, d_x is the token length, embedding dimension respectively and similarly
        for z.

        Args:
            x: current token (embedded)
            z: context token (embedded)

        Returns:
            attn: attention of shape (l_x, d_v)
        """
        # operator of shape (d_x, d_attn) (features = number of cols)
        # query of shape (l_x, d_attn) NOTE: always left multiply with input
        query = nn.Dense(features=self.d_attn)(x)  # contains bias by default

        # operator of shape (d_z, d_attn)
        # key of shape (l_z, d_attn)
        key = nn.Dense(features=self.d_attn)(z)

        # operator of shape (d_z, d_v)
        # value of shape (l_z, d_v)
        value = nn.Dense(features=self.d_v)(z)

        # scores of shape (l_x, l_z)
        scores = query @ jnp.transpose(key)

        # masking
        # TODO: how to vectorize?
        if self.unidirectional:
            rows = scores.shape[0]
            cols = scores.shape[1]
            for row in range(rows):
                for col in range(cols):
                    # this is a bit subtle compared to Hutter/Phuong because of transposed arrays
                    if row < col:
                        scores = scores.at[row, col].set(-jnp.inf)

        attn = jax.nn.softmax(scores / math.sqrt(self.d_attn)) @ value
        return attn


class MHAttention(nn.Module):
    d_attn: int  # embedding dimension of key and query ("attention dimension")
    d_v: int  # dimension of value (d_mid or d_out in Phuong/Hutter)
    d_out: int  # output dimension
    attn_heads: int  # number of heads
    l_x: int  # token length

    def setup(self):
        self.attns = [
            Attention(d_attn=self.d_attn, d_v=self.d_v, unidirectional=True)
            for _ in range(self.attn_heads)
        ]
        self.w_out = nn.Dense(features=self.d_out)

    def __call__(self, x, z):
        y = jnp.zeros((self.l_x, self.attn_heads * self.d_v))
        for idx, attn in enumerate(self.attns):
            # TODO: how to do this better? E.g. using jit?!?
            attention = attn(x, z)
            y = y.at[:, idx : idx + self.d_v].set(attention)
        # TODO: bias correct (cf. Hutter/Phuong)?
        w = self.w_out(y)  # nn.Dense(features=config.attn_heads*config.d_mid)(y)
        return w


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
    offset: bool = True  # decides if we include a learnable offset/bias

    # TODO: row-wise layer normalization?

    @nn.compact
    def __call__(self, e):
        @jax.jit
        def norm(e):
            mean = jnp.mean(e)
            scale = jnp.std(e)
            # if scale == 0: # TODO: do not treat scale == 0 here because to make jit work
            #    scale = 1
            return (e - mean) / scale

        # learnable parameters
        # TODO: is normal initialization a good choice here?
        gamma = self.param("gamma", nn.initializers.normal(), e.shape)
        if self.offset:
            beta = self.param("beta", nn.initializers.normal(), e.shape)
        return gamma * (norm(e)) + beta


class DTransformerBlock(nn.Module):
    """See Algorithm 10 (DTransformer) in Hutter/Phuong"""
    d_e: int  # word embedding dimension
    d_mlp: int  # output dimension of "activation layers"
    d_attn: int  # embedding dimension of key and query ("attention dimension")
    d_v: int  # dimension of value (d_mid or d_out in Phuong/Hutter)
    d_out: int  # output dimension;
    attn_heads: int  # number of heads
    l_x: int  # token length

    def setup(self):
        self.act_layer = DTransformerActivationLayer(d_mlp=self.d_mlp, d_e=self.d_e)
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()
        # TODO: is there some sort of "Makefile auto-completion"?
        self.mhattention = MHAttention(
            d_attn=self.d_attn,
            d_v=self.d_v,
            d_out=self.d_out,
            attn_heads=self.attn_heads,
            l_x=self.l_x,
        )

    def __call__(self, x):
        """
        Input:
            - x: typically a 2D arrays whose rows are the word + positional embeddings
                of the tokens
        TODO: check for dimension?
        """
        num_rows = x.shape[0]
        # num_cols = x.shape[1]
        x_normalized = jnp.zeros(x.shape)
        # TODO: extract this part or do in matrix form? jit?
        for i in range(num_rows):
            x_normalized = x_normalized.at[i, :].set(self.layer_norm1(x[i, :]))
        x_mhattn = self.mhattention(x_normalized, x_normalized)
        x = x + x_mhattn

        # TODO: here seems to be an issue...
        for i in range(num_rows):
            x_normalized = x_normalized.at[i, :].set(self.layer_norm2(x[i, :]))
        x = x + self.act_layer(x_normalized)

        return x


class DTransformerEmbedding(nn.Module):
    """
    Class to preprocess strings before feeding them into a Transformer model.
    """

    d_e: int  # word embedding dimension
    l_max: int
    vocab_size: int  # TODO: can read from tokens_file?!?

    def setup(self):
        self.word_embed = nn.Embed(num_embeddings=self.vocab_size, features=self.d_e)
        # TODO: add assert l_max >= length(x)?
        self.pos_embed = nn.Embed(num_embeddings=self.l_max, features=self.d_e)

    def __call__(self, x: jnp.ndarray):
        # TODO: actually assume that x is a jnp.ndarray with int entries
        X = jnp.zeros(shape=(x.size, self.d_e))
        x_int = x.astype(int)  # just to be sure
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
            X = X.at[i, :].set(x_wembed[i, :] + x_pembed[i, :])

        return X


class TransformerUnembedding(nn.Module):
    """Turn result of Transformer blocks into probabilities"""

    vocab_size: int

    def setup(self):
        self.lin_layer = nn.Dense(features=self.vocab_size)

    def __call__(self, x: jnp.ndarray):
        return jax.nn.softmax(self.lin_layer(x))


class DTransformer(nn.Module):
    vocab_size: int  # vocabulary size
    # TODO: determined by tokenizer?
    l_max: int  # max sequence length
    d_e: int  # word embedding dimension
    d_v: int
    d_mlp: int  # output dimension of "activation layers"
    num_layers: int  # number of layers
    attn_heads: int  # number of attention heads in each layer
    # dtransformer_block: DTransformerBlock

    def setup(self):
        self.dtransformer_embed = DTransformerEmbedding(
            d_e=self.d_e, l_max=self.l_max, vocab_size=self.vocab_size
        )
        self.layers = [
            DTransformerBlock(
                d_e=self.d_e,
                d_mlp=self.d_mlp,
                d_attn=self.d_e,
                d_v=self.d_v,
                d_out=self.d_e,  # that's at least typical...
                attn_heads=self.attn_heads,
                l_x=self.l_max,
            )
            for _ in range(self.num_layers)
        ]
        self.final_layer_norm = LayerNormalization()
        self.unembed = TransformerUnembedding(vocab_size=self.vocab_size)

    def __call__(self, X: jnp.ndarray):
        # X = X.at[:,:].set(self.dtransformer_embed(X))
        # TODO: clean up!
        Y = X
        Y = self.dtransformer_embed(Y)
        for _, layer in enumerate(self.layers):
            Y = layer(Y)

        num_rows = Y.shape[0]
        # Y_normalized = jnp.zeros(Y.shape)
        for i in range(num_rows):
            # TODO: refactor to matrix form?!
            Y = Y.at[i, :].set(self.final_layer_norm(Y[i, :]))

        return self.unembed(Y)

    def infer(
        self,
        tokenizer: Tokenizer,
        x: str,
        l_gen: int,
        variables,
        temperature: float = 1.0,
    ):
        # TODO: add variables before calling?
        x_ids = jnp.array(tokenizer.encode(x).ids, dtype=int)
        len_x_ids = len(x_ids)

        # pad/slice x_ids so that it is of length l_max
        # TODO: y_ids really needed?
        l_max = self.l_max
        if self.l_max < len_x_ids:
            y_ids = x_ids[len_x_ids - l_max :]
            start_idx = l_max
        else:
            y_ids = jnp.pad(x_ids, (0, self.l_max - len_x_ids))
            start_idx = len_x_ids
        assert y_ids.size == l_max

        # the following array is mostly needed if we cannot
        # store all generated tokens in x_ids
        generated_tokens = jnp.zeros(l_gen, dtype=int)

        key = jax.random.PRNGKey(23)
        P = self.apply(variables, y_ids)

        for i in range(l_gen):
            P = P.at[:, :].set(self.apply(variables, y_ids))

            # dealing with indices
            if start_idx + i >= l_max - 1:
                p_idx = l_max - 1
            else:
                p_idx = start_idx + i

            # create new random key (otherwise always get same sample)
            key, _ = jax.random.split(key)
            p = jax.nn.softmax(P[p_idx, :] / temperature)
            print(p)
            new_token = jax.random.choice(key=key, a=self.vocab_size, p=p)
            print(f"max index: {jnp.argmax(P[p_idx, :])}, p_idx: {p_idx}")

            if p_idx >= l_max - 1:
                # TODO: fix that...
                z_ids = jnp.append(y_ids, new_token)
                y_ids = y_ids.at[:].set(z_ids[1:])
            else:
                y_ids = y_ids.at[p_idx].set(new_token)
            # except:
            # TODO: this part does not yet seem to work as intended
            # TODO: also: save new tokens somewhere else as well to avoid filling
            # x_ids if l_gen > l_max?
            # TODO: needs to be improved...
            # z_ids = jnp.append(y_ids, new_token)
            # y_ids = y_ids.at[:].set(z_ids[1:])
            # x_ids = x_ids.at[len_x_ids+i:].set(new_token)

            print(y_ids)
            generated_tokens = generated_tokens.at[i].set(new_token)

            # create new random key (otherwise always get same sample)
            key, _ = jax.random.split(key)

        return tokenizer.decode(generated_tokens)


def cross_entropy(prob_distr: jnp.ndarray, sample: jnp.ndarray):
    # TODO: vectorize?
    loss = 0.0
    cols = sample.size  # TODO: add check that it's a row vector
    for i in range(cols - 1):
        loss -= jnp.log(prob_distr[i, sample[i + 1]])
    return loss


# the following is the spelt out version of TrainState
# def update_step(tx, apply_fn, sample, opt_state, params, state):
#     def loss(params):
#         P, updated_state = apply_fn({'params': params, **state},
#                                    sample, mutable=list(state.keys()))
#         l = cross_entropy(prob_distr=P, sample=sample)
#         return l, updated_state

#     (l, state), grads = jax.value_and_grad(loss, has_aux=True)(params)
#     updates, opt_state = tx.update(grads, opt_state)
#     params = optax.apply_updates(params, updates)
#     return opt_state, params, state


def train_step(state: TrainState, sample: jnp.ndarray):
    # TODO: no optimizer?

    def loss_fn(params):  # sample is implicit
        P = state.apply_fn({"params": params}, sample)
        loss = cross_entropy(prob_distr=P, sample=sample)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)  # , has_aux=True)
    loss, grads = grad_fn(state.params)
    # print(f"Loss: {loss}")
    new_state = state.apply_gradients(grads=grads)

    return new_state, loss  # TODO: does the state store the new params?!?


# NOTE: one key point of jax seems to be the 'decoupling' of the state and the model
# for example, we only train the state and the model only goes into the state via the
# apply_fn!


def train(
    state,
    train_data,
    l_max,
    # optimizer,
    epochs: int,
):
    assert len(train_data) > l_max, "Training data not long enough."  # TODO: necessary?
    sample_size = len(train_data) - l_max

    # init_vars = model.init(jax.random.PRNGKey(42), jnp.ones(config.l_max))
    # tx = optax.adamw(learning_rate=learning_rate) # TODO: add more parameters?!?
    # state = TrainState.create(apply_fn=model.apply, params=init_vars["params"], tx=optimizer)

    for _ in range(epochs):
        for i in range(sample_size):
            state, loss = train_step(state=state, sample=train_data[i : l_max + i])
            if i % 100 == 0:
                print(f"step {i}, loss: {loss}")

    return state


if __name__ == "__main__":
    train_set = config.data_path / "Tolstoy_WarAndPeace_orig.txt"
    tokenizer = own_tokenize.train_BPE_tokenizer([str(train_set)])
    vocab_size = tokenizer.get_vocab_size()

    # TODO: how to improve?
    train_data = jnp.array(
        own_tokenize.encode_file(tokenizer, train_set, limit=500).ids
    )
    optimizer = optax.adam(learning_rate=1e-2)

    model = DTransformer(
        vocab_size=vocab_size,
        l_max=config.l_max,
        d_e=config.d_e,
        d_mlp=config.d_mlp,
        d_v=config.d_v,
        num_layers=6,
        attn_heads=8,
    )

    init_vars = model.init(jax.random.PRNGKey(42), jnp.ones(config.l_max))
    state = TrainState.create(
        apply_fn=model.apply, params=init_vars["params"], tx=optimizer
    )

    # train
    epochs = 5
    train_meta_data = {"epochs": epochs, "train set": "Tolstoy_WarAndPeace_orig.txt"}
    new_state = train(
        state=state, train_data=train_data, l_max=config.l_max, epochs=epochs
    )
    ckpt = {"model": new_state, "train_meta_data": train_meta_data}

    # store
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)  # mainly for speedup
    orbax_checkpointer.save("../models/test", ckpt, save_args=save_args)
    print("Model saved")
