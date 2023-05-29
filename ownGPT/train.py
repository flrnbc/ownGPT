import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint
from flax import linen as nn
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import own_tokenize
from own_tokenize import Tokenizer
import model
from pathlib import Path
import config


@jax.jit
def cross_entropy(prob_distr: jnp.ndarray, sample: jnp.ndarray):
    # TODO: vectorize instead of jit?
    loss = 0.0
    cols = sample.size
    for i in range(cols - 1):
        loss -= jnp.log(prob_distr[i, sample[i + 1]])
    return loss


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


def train_step(state: TrainState, sample: jnp.ndarray):
    def loss_fn(params):
        P = state.apply_fn({"params": params}, sample)
        loss = cross_entropy(prob_distr=P, sample=sample)
        return loss

    # TODO: no optimizer?
    grad_fn = jax.value_and_grad(loss_fn)
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
            # TODO: check training!
            state, loss = train_step(state=state, sample=train_data[i : l_max + i])
            if i % 100 == 0:
                print(f"step {i}, loss: {loss}")

    return state


def train_dtransformer(
    train_set: Path, epochs, save_path, limit: int=None
):
    tokenizer = own_tokenize.train_BPE_tokenizer([str(train_set)])
    train_data = jnp.array(
        own_tokenize.encode_file(tokenizer, train_set, limit=limit).ids
    )
    vocab_size = tokenizer.get_vocab_size()

    dt = model.DTransformer(
        vocab_size=vocab_size,
        l_max=config.l_max,
        d_e=config.d_e,
        d_mlp=config.d_mlp,
        d_v=config.d_v,
        num_layers=config.num_layers,
        attn_heads=config.attn_heads,
    )
    init_vars = dt.init(jax.random.PRNGKey(42), jnp.ones(config.l_max))

    # TODO: need to improve optimizer?
    optimizer = optax.adam(learning_rate=1e-2)
    state = TrainState.create(
        apply_fn=dt.apply, params=init_vars["params"], tx=optimizer
    )

    # training
    train_meta_data = {"epochs": epochs, "train set": str(train_set)}
    trained_state = train(
        state=state, train_data=train_data, l_max=config.l_max, epochs=epochs
    )

    # store
    ckpt = {"model": trained_state, "train_meta_data": train_meta_data}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)  # mainly for speedup
    orbax_checkpointer.save(save_path, ckpt, save_args=save_args)
    print("Model saved to {save_path}")


if __name__ == "__main__":
    train_set = Path(config.data_path / "Tolstoy_WarAndPeace_orig.txt")
    save_path = Path(config.models_path / "test")
    train_dtransformer(train_set=train_set, epochs=5, save_path=save_path, limit=200)
# tokenizer = own_tokenize.train_BPE_tokenizer([str(train_set)])
# vocab_size = tokenizer.get_vocab_size()
