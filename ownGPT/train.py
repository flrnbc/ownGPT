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

# from own_tokenize import Tokenizer
from tokenizers import Tokenizer
from own_tokenize import train_BPE_tokenizer, encode_file
from config import Config
from model import DTransformer, DTransformerConfig


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


def train_step(
    state: TrainState, minibatch: jnp.ndarray, vocab_size, l_max
):  
    """
    Train model using minibatches (2D arrays where each row is a token sequence of length l_max + 1 (predict next token)).
    """
    batch_size = minibatch.shape[0]
    # NOTE: implicitly checks if minibatch is 2D as well
    assert minibatch.shape == (batch_size, l_max + 1)
    x = minibatch[:, :-1]
    y = minibatch[:, 1:]

    def compute_loss(params):
        P = state.apply_fn({"params": params}, x)
        P_one_hot = jax.nn.one_hot(y, num_classes=vocab_size)
        assert P.shape == P_one_hot.shape
        cross_entropy = optax.softmax_cross_entropy(logits=P, labels=P_one_hot)
        # return P as well e.g. for logging purposes
        return jnp.mean(cross_entropy), P

    # has_aux = True to consider only first argument of output 
    # otherwise crashes because loss function returns a tuple
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)  # TODO: no optimizer?
    (loss, P), grads = grad_fn(state.params)
    # print(f"Loss: {loss}")
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


# TODO: does the state store the new params?!?
# NOTE: one key point of jax seems to be the 'decoupling' of the state and the model
# for example, we only train the state and the model only goes into the state via the
# apply_fn!


def train(
    state: TrainState,
    train_data: Path,
    l_max: int,
    # optimizer,
    epochs: int,
):
    assert len(train_data) > l_max, "Training data not long enough."
    sample_size = len(train_data) - l_max

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
    """Naive because we load all the data into memory. Also only work with txt-files."""

    data_path: Path
    seq_length: int

    def minibatch(self, batch_size: int, tokenizer: Tokenizer, key: jax.random.PRNGKey) -> jnp.array:
        """Method to create a minibatch of batch_size.     
        Returns:
            minibatch of batch_size. Each row of a minibatch is a 1D jnp.array
            of length seq_length + 1. It corresponds to a consecutive token sequence
            of the same length randomly selected with 'key' from data_path after encoding 
            via 'tokenizer'.
        """
        full_train_data = jnp.array(encode_file(tokenizer, self.data_path).ids)
        stop = full_train_data.size - (self.seq_length + 1)
        assert stop >= 0, "Train data not long enough."
        starts_of_batch_rows = jax.random.randint(
            key=key, shape=(batch_size,), minval=0, maxval=stop + 1
        )  # values are in [minval, maxval) so need stop+1
        # TODO: could not make the following work with e.g. vmap...
        minibatch = jnp.array(
            [
                full_train_data[start : start + self.seq_length + 1]
                for start in starts_of_batch_rows
            ]
        )
        assert minibatch.shape == (batch_size, self.seq_length + 1)
        return minibatch

@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    seed: int 

def train_model(
    model: DTransformer, # TODO: general to other transformers from paper
    train_set: Path,
    # config: DTransformerConfig,
    tokenizer: Tokenizer,
    train_cfg: TrainConfig,
    save_path: Path,
    # limit: int = None,
):
    # train_data = jnp.array(
    #    encode_file(tokenizer, train_set, limit=limit).ids
    # )
    # TODO: check earlier if save_path already exists
    cfg = model.config
    l_max = cfg.l_max
    loader = NaiveDataLoader(data_path=train_set, seq_length=l_max)
    vocab_size = tokenizer.get_vocab_size()
    cfg.vocab_size = vocab_size

    # deal with PRNG keys and init
    key = jax.random.PRNGKey(train_cfg.seed)
    key, init_key, load_key = jax.random.split(key, num=3) # TODO: independent enough?
    init_vars = model.init(init_key, jnp.ones((train_cfg.batch_size, l_max)))

    # TODO: need to improve optimizer?
    optimizer = optax.adam(learning_rate=train_cfg.learning_rate)
    state = TrainState.create(
        apply_fn=model.apply, params=init_vars["params"], tx=optimizer
    )

    # training
    train_meta_data = {"epochs": train_cfg.epochs, "train set": str(train_set)}
    # NOTE: we use the same minibatch for each epoch, see e.g. Algorithm 13 in [PH]
    minibatch = loader.minibatch(batch_size=train_cfg.batch_size, tokenizer=tokenizer, key=load_key)
    for idx in range(train_cfg.epochs):
        state, loss = train_step(
            state=state, minibatch=minibatch, vocab_size=vocab_size, l_max=cfg.l_max
        )
        print(f"epoch {idx+1}, loss: {loss}")

    # store
    ckpt = {"model": state, "train_meta_data": train_meta_data}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)  # mainly for speedup
    orbax_checkpointer.save(save_path, ckpt, save_args=save_args)
    print(f"Model saved to {save_path.name}")


if __name__ == "__main__":
    train_set = Path(Config.data_path) / Path("tokenize/Tolstoy_WarAndPeace.txt")
    save_path = Path(Config.models_path / "test5")
    train_model(
        train_set=train_set,
        config=Config,
        batch_size=200,
        epochs=50,
        save_path=save_path,
    )
# tokenizer = own_tokenize.train_BPE_tokenizer([str(train_set)])
# vocab_size = tokenizer.get_vocab_size()
