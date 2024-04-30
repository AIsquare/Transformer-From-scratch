import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

#Input Emebeddings

class InputEmbedding(nn.Module):
    def __init__(self,embed_dim, vocab_size):
        super().__init__()
        #Vocab Size
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        #Mapping vocabulary to embed_dim(vocab_size,embed_di)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self,x):

        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        embedded_input = self.embedding(x)
       # Shape: (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        scaled_embedded_input = embedded_input * torch.sqrt(torch.tensor(self.embed_dim))
        return scaled_embedded_input

#Positional Encoding

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim = 512, max_seq_len = 100, dropout = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)
        self.positional_encoding = self._precompute_positional_encoding(max_seq_len, embed_dim)

        def _precompute_positional_encoding(self, max_seq_len, embed_dim):
            with torch.no_grad():
                # positional encoding matrix of shape (max_seq_len, embed_dim)
                positional_encoding = torch.zeros(max_seq_len, embed_dim)
                position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
                # The positional encoding matrix
                division_term = torch.exp(
                    torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
                positional_encoding[:, 1::2] = torch.cos(position * division_term)
                # Shape (max_seq_len, embed_dim) -> (1, max_seq_len, embed_dim)
                positional_encoding = positional_encoding.unsqueeze(0)
            return positional_encoding

        def forward(self, x):
           x = x + self.positional_encoding[:, : x.size(1)].to(x.device)
           x = self.dropout(x)
        return x

class LayerNormalization(nn.Module):
    def __init__(self, embed_dim, eps= 1e-6):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.Tensor(embed_dim).uniform_()) # Initialize with values sa
        self.bias = nn.Parameter(torch.Tensor(embed_dim).normal_()) # Initialize with values s

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.gain + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, intermediate_size) # W1 and B1 in the formula
        self.fc2 = nn.Linear(intermediate_size, embed_dim) # W2 and B2 in the formula
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # (Batch, Seq_len, embed_dim) -> (Batch, Seq_len, intermediate_size) -> (Batch, Seq_len,embed_dim)
        x_intermediate = self.dropout(F.relu(self.fc1(x)))
        x_output = self.fc2(x_intermediate)
        return x_output

def generate_square_subsequent_mask(size: int, device: torch.device = "cpu"):
        mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=0)
        mask = mask.long()
    return mask.unsqueeze(0)


# noinspection PyUnreachableCode
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim = 512, num_heads = 8, attn_dropout = 0.1, ff_dropout=0.1,max_len=350):
        super().__init__()
        self.num_heads = num_heads
        assert embed_dim % self.num_heads == 0, "invalid heads and embedding dimension configuration"

        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(ff_dropout)
        # Create a buffer to store the mask with no grad
        # Shape: (1, max_len, max_len)
        self.register_buffer("mask",torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1))

    def forward(self, x, mask=None):
        # Shape: (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_heads, head_dim) -
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim
        q = self.query(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        attention = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(q.size(-1))
        # Apply mask if provided
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))

        # Shape: (batch_size, num_heads, seq_len, seq_len) -> (batch_size, num_heads, seq_len, he
        attention = self.attn_dropout(F.softmax(attention, dim=-1))

        # Shape: (batch_size, num_heads, seq_len, seq_len) * (batch_size, num_heads, seq_len, hea
        # -> (batch_size, num_heads, seq_len, head_dim)
        y = torch.einsum('bhij,bhjd->bhid', attention, v)

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # Shape: (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        return self.proj_dropout(self.proj(y))

class ResidualConnection(nn.Module):
    def __init__(self, embed_dim, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = LayerNormalization(embed_dim=embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        normalized_x = self.layer_norm(x)
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        sublayer_output = sublayer(normalized_x)
        # Add residual connection and apply dropout
        # (batch_size, seq_len, embed_dim) + (batch_size, seq_len, embed_dim) -> (batch_size, seq
        residual_output = x + self.dropout(sublayer_output)
        return residual_output

class ProjectionHead(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, vocab_size)
    def forward(self, x):
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, vocab_size)
        return self.fc(x)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            embed_dim = 512,
            num_heads = 8,
            ff_dim = 2048,
            attn_dropout = 0.1,
            ff_dropout = 0.1,
            dropout = 0.1,
            max_len = 512
    ):
        super().__init__()
        self.MultiHeadAttention = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        max_len=max_len,
        )
        self.feed_forward = FeedForwardBlock(
        embed_dim=embed_dim,
        intermediate_size=ff_dim,
        dropout=ff_dropout,
        )
        self.residual_connection1 = ResidualConnection(embed_dim=embed_dim, dropout=dropout)
        self.residual_connection2 = ResidualConnection(embed_dim=embed_dim, dropout=dropout)


    def forward(self, x, attention_mask=None):
        x_with_attention = self.residual_connection1(x, lambda x: self.MultiHeadAttention(x, mask
        x_with_ff = self.residual_connection2(x_with_attention, self.feed_forward)
        return x_with_ff

class GPT(nn.Module):
    def __init__(
            self,
        vocab_size: int,
        embed_dim: int = 512,
        max_len: int = 512,
        embed_dropout: float = 0.1,
        num_blocks: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1
    ):

        super().__init__()
        self.max_len = max_len
        self.token_embedding = InputEmbedding(embed_dim=embed_dim,vocab_size=vocab_size)
        self.positional_embedding = PositionalEncoding(embed_dim=embed_dim,max_seq_len=max_len,dropout=embed_dropout)
        self.blocks = nn.ModuleList([DecoderBlock(embed_dim=embed_dim,num_heads=num_heads,ff_dim=ff_dim,attn_dropout=attn_dropout,ff_dropout=ff_dropout,max_len=max_len,) for _ in range(num_blocks)])
        self.projection_head = ProjectionHead(embed_dim=embed_dim, vocab_size=vocab_size)
    def forward(self, input_ids, attention_mask = None):
        # Shape: (batch_size, seq_len) -> (seq_len)
        seq_len = input_ids.size(1)
        assert seq_len <= self.max_len, "Sequence longer than model capacity"
        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        x = self.token_embedding(input_ids) # (batch_size, seq_len, embed_dim)
        # Shape: (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        x = self.positional_embedding(x)
        # output of each block is the hidden state of the transformer
        # Shape: (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        for block in self.blocks:
        x = block(x, attention_mask=attention_mask)
        # Shape: (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, vocab_size)
        x = self.projection_head(x) # (batch_size, seq_len, vocab_size)
        return x