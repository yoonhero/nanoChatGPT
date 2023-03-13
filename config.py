import torch
from dataclasses import dataclass

# Hyper Parameters
batch_size = 64 # how many independent sequences will we process in parallel?
max_iters = 1000
eval_interval = 10
save_interval = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Basic Configs
MODEL_PATH="./tmp/checkpoints/" # for colab /content/drive/MyDrive/tmp/checkpoints/
TXT_FILE_PATH="./dataset/" # for colab /content/drive/MyDrive/korean_murim_book.txt
load = False

# GPT config class
@dataclass 
class GPTConfig: 
    block_size:int # what is the maximum context length for predictions?
    n_embd: int
    n_heads: int
    n_layer: int
    vocab_size: int
    dropout: int = 0.2

S_GPT_CONFIG = GPTConfig(block_size=32, n_embd=32, n_heads=16, n_layer=1, dropout=0.2, vocab_size=50257)
LARGE_GPT_CONFIG = GPTConfig(block_size=128, n_embd=512, n_heads=16, n_layer=6, dropout=0.2, vocab_size=50257)