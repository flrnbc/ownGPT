from pathlib import Path
import os

# various dimensions (for attention)
d_attn = 124
d_e = 2  # 768 #256
d_mlp = 768  # TODO: which choice?
d_x = 32  # current token
d_z = 64  # context token
d_v = 128
d_out = 256  # output dim
vocab_size = 50304 # from nanoGPT

# sequence lengths
# "features" which are in our case rows of the "attention matrices"
l_x = 12  # 16
l_z = 16
# maximal sequence length
l_max = 32  # 1024
# maximal generated length
l_gen = 64

# further neural network parameters
num_layers = 6
attn_heads = 8

# data paths
cwd = Path(__file__).parents[1].resolve()
data_path = cwd / Path("data")
models_path = cwd / Path("models")
tokenize_path = cwd / Path("data/tokenize")
tokens_path = cwd / Path("data/tokens")