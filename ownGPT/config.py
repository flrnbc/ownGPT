from pathlib import Path
import os

# various dimensions (for attention)
d_attn = 124
d_e = 512  # embedding dimension
d_mlp = 512  # TODO: which choice?
d_v = 128
d_out = 512  # TODO: which choice?
vocab_size = 50304  # from nanoGPT

# sequence lengths
l_max = 1024 # maximal sequence length
l_gen = 64 # maximal generated length

# further neural network parameters
num_layers = 8
attn_heads = 8

# data paths
cwd = Path(__file__).parents[1].resolve()
data_path = cwd / Path("data")
models_path = cwd / Path("models")
tokenize_path = cwd / Path("data/tokenize")
tokens_path = cwd / Path("data/tokens")
