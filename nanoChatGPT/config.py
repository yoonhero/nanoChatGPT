import torch
from dataclasses import dataclass

# Hyper Parameters
batch_size = 64 
max_epoch = 1000
eval_interval = 10
save_interval = 50
learning_rate = 5e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Basic Configs
TRAINING_OUTPUT_DIR="./tmp/checkpoints/" # for colab /content/drive/MyDrive/tmp/checkpoint/

# GPT config class
@dataclass 
class GPTConfig: 
    block_size:int 
    n_embd: int
    n_heads: int
    n_layer: int
    vocab_size: int
    dropout: int = 0.2
    

# parameters: 680048
SMALL_GPT_CONFIG = GPTConfig(block_size=32, n_embd=32, n_heads=16, n_layer=2, vocab_size=64512) 
# paramters: 67959936
LARGE_GPT_CONFIG = GPTConfig(block_size=128, n_embd=384, n_heads=12, n_layer=10, dropout=0.2, vocab_size=64512) 
# paramters: 338298880
SUPER_LARGE_GPT_CONFIG = GPTConfig(block_size=512, n_embd=1024, n_heads=16, n_layer=16, vocab_size=64512) 
KOGPT_CONFIG = GPTConfig(block_size=2048, n_embd=4096, n_heads=16, n_layer=28, vocab_size=64512)
ULTRA_SUPER_SUPER_LARGE_LARGE_CHATGPT_CONFIG = GPTConfig(block_size=8192, n_embd=1024, n_heads=64, n_layer=12, dropout=0.2, vocab_size=100000)

## LLAMA 7B model configuration
LLAMA_7B_CONFIG = GPTConfig(block_size=128, n_embd=384, n_heads=8, n_layer=10, vocab_size=64512)
## 0.3B
GPT_FINAL_CONFIG = GPTConfig(block_size=256, n_embd=384, n_heads=8, n_layer=10, vocab_size=480000)

