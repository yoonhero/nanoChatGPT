import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tiktoken
from dataclasses import dataclass
import glob 
from torch.utils.data import Dataset, random_split, DataLoader

# Hyper Parameters
batch_size = 64 # how many independent sequences will we process in parallel?
max_iters = 1000
start_epoch = 0
eval_interval = 10
save_interval = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH="./tmp/checkpoints/"
TXT_FILE_PATH="./dataset/"
load = False
# -------------------

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)

# GPT config class
@dataclass 
class GPTConfig: 
    block_size:int # what is the maximum context length for predictions?
    n_embd: int
    n_heads: int
    n_layer: int
    dropout: int = 0.2
    vocab_size: int

S_GPT_CONFIG = GPTConfig(block_size=64, n_embd=128, n_heads=16, n_layer=2, dropout=0.2, vocab_size=enc.n_vocab)
LARGE_GPT_CONFIG = GPTConfig(block_size=128, n_embd=512, n_head=16, n_layer=6, dropout=0.2, vocab_size=enc.n_vocab)

class GPTDataset(Dataset):
    def __init__(self, txt_file, block_size):
        self.block_size = block_size
        
        with open(txt_file, "r", encoding="cp949") as f:
            text = f.read()
        self.encoded_texts = encode(text)
        self.length = (len(self.encoded_texts)-block_size) // block_size
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.encoded_texts[index*self.block_size:(index+1)*self.block_size+1]
        y = self.encoded_texts[index*self.block_size+1:(index+1)(self.block_size)+1]
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        x, y = x.to(device), y.to(device)
        return x, y

# Train and test splits
# data = torch.tensor(encode(text), dtype=torch.long)
# # data loading
# def get_batch(split):
#     # generate a small batch of data of inputs x and targets y
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([data[i:i+block_size] for i in ix])
#     y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#     x, y = x.to(device), y.to(device)
#     return x, y

dataset = GPTDataset(TXT_FILE_PATH, block_size=S_GPT_CONFIG.block_size)
total_size = len(dataset)
train_size = int(0.8*total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        # losses = torch.zeros(eval_iters)
        losses = []
        # for k in range(eval_iters):
        if split == "train":
            for X, Y in train_loader:
            # X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses.append(loss.item())
        elif split == "val":
            for X, Y in val_loader:
                _, loss = model(X, Y)
                losses.append(loss.item())      
        out[split] = loss.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout
        self.key = nn.Linear(self.n_embd, self.head_size, bias=False)
        self.query = nn.Linear(self.n_embd, self.head_size, bias=False)
        self.value = nn.Linear(self.n_embd, self.head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(self.block_size, self.block_size)))
        
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, C) @ (B, C, T) => (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregatoin of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, C)
        return out
    

class MultiHeadAttention(nn.Module):
    # Multiple heads of self-attention in parallel
    def __init__(self, num_heads, head_size, n_embd, dropout) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embd = n_embd
        self.dropout = dropout
        self.heads = nn.ModuleList([Head(self.head_size) for _ in range(self.num_heads)])
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.dropout = dropout
        self.n_embd = n_embd
        self.net = nn.Sequential(
            nn.Linear(self.n_embd, 4*self.n_embd),
            nn.ReLU(),
            nn.Linear(4*self.n_embd, self.n_embd),
            nn.Dropout(self.dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followd by computation"""

    def __init__(self, n_embd, n_heads):
        # n_embd: embedding dimension, n_heads: the number of the heads 
        super().__init__()
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_size = self.n_embd // self.n_heads
        self.sa = MultiHeadAttention(self.n_heads, self.head_size)
        self.ffwd = FeedForward(self.n_embd)
        self.ln1 = nn.LayerNorm(self.n_embd)
        self.ln2 = nn.LayerNorm(self.n_embd)
    
    def forward(self, x):
        x = x+self.sa(self.ln1(x))
        x = x+self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.config = config
        self.n_embd = self.config.n_embd
        self.n_heads = self.config.n_heads
        self.n_layer = self.config.n_layer
        self.dropout = self.config.dropout
        self.vocab_size = self.config.vocab_size

        # each toekn directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)
        self.positional_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        self.sa_heads = MultiHeadAttention(self.n_heads, self.n_embd//self.n_heads, self.n_embd, self.dropout)
        # feed forward layer is needed for think about the self attention score 
        # when we pass the self attention score straight forward to the last layer 
        # it's hard to think about the meaning of the score
        self.blocks = nn.Sequential(*[Block(self.n_embd, n_heads=self.n_heads) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integes
        # C is the Channel which represents the embedding table output size
        # when we pass the idx to the token embedding table 
        # we get a embedidng tensor by the idx and get by one by one
        token_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_emb + pos_emb
        x = self.sa_heads(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get the prediction
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sample index to the running sequnce
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = GPTLanguageModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.9)

def save_model(epoch, model, optimizer):
    model_state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }   
    torch.save(model_state_dict, PATH+f"{epoch}.tar")

def get_last_epoch() -> int:
    """Get the last epoch and TAR file"""
    files = glob.glob(f"{PATH}*")
    if len(files) == 0:
        return None
    
    epochs = [int(filename.split("/")[-1].split(".")[0]) for filename in files]
    return max(epochs)

if load: 
    last_epoch = get_last_epoch()
    model_state_dict = torch.load(PATH + f"{last_epoch}.tar")

    model.load_state_dict(model_state_dict["model"])
    optimizer.load_state_dict(model_state_dict["optimizer"])

    start_epoch = model_state_dict["epoch"]

for iter in range(start_epoch, start_epoch+max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss(model=model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    if iter % save_interval == 0:
        save_model(iter, model, optimizer)


    for idx, (x, y) in enumerate(train_loader):
        # sample a batch of data
        # xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # scheduler.step()


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
result = decode(model.generate(context, max_new_tokens=500)[0].tolist())

with open('result.txt', "w", encoding="cp949") as f:
    f.writelines(result)
    f.close()
