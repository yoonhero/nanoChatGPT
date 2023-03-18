import torch
from dataclasses import dataclass

# Hyper Parameters
batch_size = 64 # how many independent sequences will we process in parallel?
max_iters = 1000
eval_interval = 10
save_interval = 50
learning_rate = 6e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Basic Configs
MODEL_PATH="./tmp/checkpoints/" # for colab /content/drive/MyDrive/tmp/checkpoint/
TXT_FILE_PATH="./dataset/" # for colab /content/drive/MyDrive/korean_murim_book.txt

# GPT config class
@dataclass 
class GPTConfig: 
    block_size:int # what is the maximum context length for predictions?
    n_embd: int
    n_heads: int
    n_layer: int
    vocab_size: int
    dropout: int = 0.2
    

# GPT-3.5 Tokenizer = 100277
SMALL_GPT_CONFIG = GPTConfig(block_size=32, n_embd=32, n_heads=16, n_layer=2, vocab_size=10000)
LARGE_GPT_CONFIG = GPTConfig(block_size=128, n_embd=768, n_heads=12, n_layer=12, dropout=0.1, vocab_size=64512)
SUPER_LARGE_GPT_CONFIG = GPTConfig(block_size=512, n_embd=1024, n_heads=16, n_layer=16, vocab_size=64512)
KOGPT_CONFIG = GPTConfig(block_size=2048, n_embd=4096, n_heads=16, n_layer=28, vocab_size=64512)
ULTRA_SUPER_SUPER_LARGE_LARGE_CHATGPT_CONFIG = GPTConfig(block_size=8192, n_embd=1024, n_heads=64, n_layer=12, dropout=0.2, vocab_size=100000)

