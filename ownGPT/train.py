from pathlib import Path
from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from flax import linen as nn
from flax.training import orbax_utils
from flax.training.train_state import TrainState

#from own_tokenize import Tokenizer
from own_tokenize import train_BPE_tokenizer, encode_file
from config import Config
from model import DTransformer


""" NOTE: the following is essentially the spelt out version of TrainState
def update_step(tx, apply_fn, sample, opt_state, params, state):
    def loss(params):
        P, updated_state = apply_fn({'params': params, **state},
                                   sample, mutable=list(state.keys()))
        l = cross_entropy(prob_distr=P, sample=sample)
        return l, updated_state

    (l, state), grads = jax.value_and_grad(loss, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params, state """

def train_step(state: TrainState, minibatch: jnp.ndarray, vocab_size, l_max): # TODO: read vocab_size from model/state?
    """
    Train model using minibatches (2D arrays where each row is a token sequence of length l_max + 1 (predict the last token)).
    """
    batch_size = minibatch.shape[0]
    # NOTE: implicitly checks if minibatch is 2D as well
    assert minibatch.shape == (batch_size, l_max + 1) 
    x = minibatch[:, :-1]
    y = minibatch[:, 1:]

    def compute_loss(params):
        P = state.apply_fn({"params": params}, x)
        P_one_hot = jax.nn.one_hot(
            y, num_classes=vocab_size
        )
        assert P.shape == P_one_hot.shape
        cross_entropy = optax.softmax_cross_entropy(
            logits=P, labels=P_one_hot
        )
        # return P as well e.g. for logging purposes
        return jnp.mean(cross_entropy), P

    # has_aux = True to consider only first argument of output; otherwise crashes 
    # because loss function returns a tuple
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True) # TODO: no optimizer?
    (loss, P), grads = grad_fn(state.params)
    # print(f"Loss: {loss}")
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss  
    
# TODO: does the state store the new params?!?
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
    assert len(train_data) > l_max, "Training data not long enough."
    # TODO: build batch
    sample_size = len(train_data) - l_max

    # init_vars = model.init(jax.random.PRNGKey(42), jnp.ones(config.l_max))
    # tx = optax.adamw(learning_rate=learning_rate) # TODO: add more parameters?!?
    # state = TrainState.create(apply_fn=model.apply, params=init_vars["params"], tx=optimizer)

    for _ in range(epochs):
        for i in range(sample_size):
            # TODO: check training! How to vectorize?
            sample = train_data[i : l_max + i]
            state, loss = train_step(state=state, minibatch=sample)
            if i % 100 == 0:
                print(f"step {i}, loss: {loss}")

    return state

@dataclass
class NaiveDataLoader:
    """Naive because we load all the data into memory. Also only work with txt-file."""
    data_path: Path
    seq_length: int

    def get_tokenizer(self):
        tokenizer = train_BPE_tokenizer([str(self.data_path)])
        return tokenizer

    def minibatch(self, batch_size: int, tokenizer, key) -> jnp.array:
        """
        Args:
            batch_size: int
            key

        Returns:
            minibatch of batch_size. Each row of a minibatch is a 1D jnp.array
            of length seq_length + 1. It corresponds to a consecutive token sequence of the same
            length randomly selected from data_path (after encoding).
        """
        full_train_data = jnp.array(
            encode_file(tokenizer, self.data_path).ids
        )
        stop = full_train_data.size - (self.seq_length + 1)
        assert stop >= 0, "Train data not long enough."
        starts_of_batch_rows = jax.random.randint(key=key, shape=(batch_size,), minval=0, maxval=stop+1) # values are in [minval, maxval) so need stop+1
        # TODO: could not make the following work with e.g. vmap...
        minibatch = jnp.array([full_train_data[start:start+self.seq_length+1] for start in starts_of_batch_rows])
        return minibatch


# TODO: use model as input
def train_dtransformer(
    train_set: Path, batch_size: int, epochs: int, save_path, limit: int=None
):
    #train_data = jnp.array(
    #    encode_file(tokenizer, train_set, limit=limit).ids
    #)

    loader = NaiveDataLoader(data_path=train_set, seq_length=Config.l_max)
    tokenizer = loader.get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    dt = DTransformer(
        vocab_size=vocab_size,
        l_max=Config.l_max,
        d_e=Config.d_e,
        d_mlp=Config.d_mlp,
        d_v=Config.d_v,
        num_layers=Config.num_layers,
        attn_heads=Config.attn_heads,
    )
    init_vars = dt.init(jax.random.PRNGKey(42), jnp.ones((batch_size, Config.l_max)))

    # TODO: need to improve optimizer?
    optimizer = optax.adam(learning_rate=1e-2)
    state = TrainState.create(
        apply_fn=dt.apply, params=init_vars["params"], tx=optimizer
    )

    # training
    train_meta_data = {"epochs": epochs, "train set": str(train_set)}
    key = jax.random.PRNGKey(87)
    for idx in range(epochs):
        key, _ = jax.random.split(key)
        minibatch = loader.minibatch(batch_size=batch_size, tokenizer=tokenizer, key=key)
        state, loss = train_step(state=state, minibatch=minibatch, vocab_size=vocab_size, l_max=Config.l_max)
        print(f"epoch {idx}, loss: {loss}")

    # store
    ckpt = {"model": state, "train_meta_data": train_meta_data}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)  # mainly for speedup
    orbax_checkpointer.save(save_path, ckpt, save_args=save_args)
    print("Model saved to {save_path}")


if __name__ == "__main__":
    train_set = Path(Config.data_path) / Path("tokenize/test.txt") # Tolstoy_WarAndPeace.txt")
    save_path = Path(Config.models_path / "test2")
    train_dtransformer(train_set=train_set, batch_size=20, epochs=5, save_path=save_path)
# tokenizer = own_tokenize.train_BPE_tokenizer([str(train_set)])
# vocab_size = tokenizer.get_vocab_size()
