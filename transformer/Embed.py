import torch 
import torch.nn as nn
from torch.autograd import Variable
import math

# Encoder

# Input Embedding
# Positional Encoding 
# Encoder Box Layer
    # Multi-Head Attention
    # Add & Norm
    # Feed Forward
    # Add & Norm

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant "pe" matrix with values dependent on pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos/10000**((i*2)/d_model))
                pe[pos, i+1] = math.cos(pos/10000**(((i+1)*2)/d_model))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # make embeddings relatively layer
        x = x*math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x+pe
        return self.dropout(x)


if __name__ == "__main__":
    sample_pos_encoding = PositionalEncoder(128, 50)
    import matplotlib.pyplot as plt
    plt.pcolormesh(sample_pos_encoding.pe.numpy(), cmap='RdBu')
    plt.xlabel('Depth')
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()