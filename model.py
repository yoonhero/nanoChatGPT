import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    
class CasualAttention(nn.Module):
    # Multiple heads of self-attention in parallel
    # for efficiency
    def __init__(self, config) -> None:
        super().__init__()
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        assert self.n_embd % self.n_heads == 0, "Please Check Embedding and Heads Number Config."
        self.head_size = self.n_embd//self.n_heads
        self.dropout = config.dropout
        self.block_size = config.block_size

        self.c_attn = nn.Linear(self.n_embd, self.n_embd*3, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_drop = nn.Dropout(self.dropout)

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        # calculate query and key and value parallel and split it
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, N_HEADS, HEAD_SIZE) -> (B, N_HEADS, T, HEAD_SIZE)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1))) # (B, N_HEADS, T, T)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) 
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

        y = self.resid_drop(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.net = nn.Sequential(
            nn.Linear(self.n_embd, 4*self.n_embd, bias=False),
            nn.ReLU(),
            # SwiGLU for better result => LLAMA
            # nn.SiLU(),
            nn.Linear(4*self.n_embd, self.n_embd, bias=False),
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
        self.block_size = config.block_size
        self.head_size = self.n_embd // self.n_heads

        # self.positional_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        # self.sa = MultiHeadAttention(config)
        self.sa = CasualAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(self.n_embd)
        self.ln2 = nn.LayerNorm(self.n_embd)
    
    def forward(self, x):
        B, T, C = x.shape
        # pos_emb = self.positional_embedding_table(torch.arange(T, device=device))
        x = x+self.sa(self.ln1(x))
        # x = x+pos_emb
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
        # self.sa_heads = MultiHeadAttention(config)
        self.dropout = nn.Dropout(self.dropout)
        # feed forward layer is needed for think about the self attention score 
        # when we pass the self attention score straight forward to the last layer 
        # it's hard to think about the meaning of the score
        self.blocks = nn.Sequential(*[Block(config) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size)

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print(f"Number of parameters: {self.get_num_params()}")

    def get_num_params(self):
        n_params = [p.nelement() for p in self.parameters()]
        return sum(n_params)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integes
        # C is the Channel which represents the embedding table output size
        # when we pass the idx to the token embedding table 
        # we get a embedidng tensor by the idx and get by one by one
        token_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.positional_embedding_table(torch.arange(T, device="cuda")) # (T, C)
        x = token_emb + pos_emb
        x = self.dropout(x)
        # x = self.sa_heads(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] / temperature # becomes (B, C)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sample index to the running sequnce
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

