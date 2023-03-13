import torch

# Hyper Parameters
batch_size = 64 # how many independent sequences will we process in parallel?
max_iters = 1000
start_epoch = 0
eval_interval = 10
save_interval = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Basic Configs
PATH="./tmp/checkpoints/"
TXT_FILE_PATH="./dataset/"
load = False