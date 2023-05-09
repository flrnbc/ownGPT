from pathlib import Path
import os

# various dimensions (for attention)
d_attn = 124
d_e = 2 #768 #256
d_mlp = 768 # TODO: which choice?
d_x = 32 # current token
d_z = 64 # context token
d_v = 128
d_out = 256 # output dim


# number of attention heads
attn_heads = 6

# sequence lengths 
# "features" which are in our case rows of the "attention matrices"
l_x = 8 #16
l_z = 16
# maximal sequence length
l_max = 512 # 1024

# vocabulary size
vocab_size = 50304 # taken from nanoGPT

# data paths
cwd = Path(__file__).parents[1].resolve()
tokenize_path = cwd / Path("data/tokenize")
tokens_path = cwd / Path("data/tokens")
