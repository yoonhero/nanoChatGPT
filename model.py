import torch
import torch.nn as nn
import torch.nn.functional as F
from config import device

class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layer = config.n_layer
        self.dropout = config.dropout
        self.block_size = config.block_size
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_size = self.n_embd // self.n_heads

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
    def __init__(self, config) -> None:
        super().__init__()
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.head_size = self.n_embd//self.n_heads
        self.dropout = config.dropout
        self.block_size = config.block_size
        self.heads = nn.ModuleList([Head(config) for _ in range(self.n_heads)])
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.dropout = config.dropout
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

    def __init__(self, config):
        # n_embd: embedding dimension, n_heads: the number of the heads 
        super().__init__()
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.head_size = self.n_embd // self.n_heads

        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(self.n_embd)
        self.ln2 = nn.LayerNorm(self.n_embd)
    
    def forward(self, x):
        x = x+self.sa(self.ln1(x))
        x = x+self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.n_layer = config.n_layer
        self.dropout = config.dropout
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size

        # each toekn directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)
        self.positional_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        self.sa_heads = MultiHeadAttention(config)
        # feed forward layer is needed for think about the self attention score 
        # when we pass the self attention score straight forward to the last layer 
        # it's hard to think about the meaning of the score
        self.blocks = nn.Sequential(*[Block(config) for _ in range(self.n_layer)])
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
            print(idx.shape, max_new_tokens)
            if idx.shape[1] > self.block_size:
                idx_cond = idx[:, -self.block_size:]
            else: 
                idx_cond = idx

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
