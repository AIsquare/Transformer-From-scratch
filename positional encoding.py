import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt

# Define sequences (assuming it's a list of tokenized sequences)
sequences = 'A quick brown fox jumps over the lazy dog!'
chars = sorted(list(set(sequences)))
print(''.join(chars))
vocab_size = len(chars)
print(vocab_size)

stoi  = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #encoder: take a string, output a list of integer
decode = lambda l: ''.join([itos[i] for i in l])
print(encode('hi Aamir'))
print(decode(encode('hi Aamir')))


# Convert the sequences to a tensor
tensor_sequences = torch.tensor(encode('hi Aamir')).long()
print(tensor_sequences)
# # Vocabulary size
vocab_size = len(stoi)
# Embedding dimensions
d_model = 4

# Create the embeddings using nn.Embedding module
lut = nn.Embedding(vocab_size, d_model)  # Look-up table (lut)

# Embed the sequence
embeddings = lut(tensor_sequences)

#print('Embedding',embeddings)

'''Encoding the position of each word in each sequence via positional encodings
'''
def gen_pe(max_length, d_model, n):

  # generate an empty matrix for the positional encodings (pe)
  pe = np.zeros(max_length*d_model).reshape(max_length, d_model)

  # for each position
  for k in np.arange(max_length):

    # for each dimension
    for i in np.arange(d_model//2):

      # calculate the internal value for sin and cos
      theta = k / (n ** ((2*i)/d_model))

      # even dims: sin
      pe[k, 2*i] = math.sin(theta)

      # odd dims: cos
      pe[k, 2*i+1] = math.cos(theta)

  return pe
# maximum sequence length
max_length = 12
n = 100
encodings = gen_pe(max_length, d_model, n)
#print(encodings)

seq_length = embeddings.shape[1]
print('seq len',seq_length)
print(encodings[:seq_length])

d_model = 4
n = 100

div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(n) / d_model))
print(div_term)

max_length = 12

# generate the positions into a column matrix
k = torch.arange(0, max_length).unsqueeze(1)

print(k)

# generate an empty tensor
pe = torch.zeros(max_length, d_model)

# set the odd values (columns 1 and 3)
pe[:, 0::2] = torch.sin(k * div_term)

# set the even values (columns 2 and 4)
pe[:, 1::2] = torch.cos(k * div_term)

# add a dimension for broadcasting across sequences: optional
pe = pe.unsqueeze(0)
#print(pe)


class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
    """
    Args:
      d_model:      dimension of embeddings
      dropout:      randomly zeroes-out some of the input
      max_length:   max sequence length
    """
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    # create tensor of 0s
    pe = torch.zeros(max_length, d_model)

    # create position column
    k = torch.arange(0, max_length).unsqueeze(1)

    # calc divisor for positional encoding
    div_term = torch.exp(
      torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )

    # calc sine on even indices
    pe[:, 0::2] = torch.sin(k * div_term)

    # calc cosine on odd indices
    pe[:, 1::2] = torch.cos(k * div_term)

    # add dimension
    pe = pe.unsqueeze(0)

    # buffers are saved in state_dict but not trained by the optimizer
    self.register_buffer("pe", pe)

  def forward(self, x: torch.Tensor):
    """
    Args:
      x:        embeddings (batch_size, seq_length, d_model)

    Returns:
                embeddings + positional encodings (batch_size, seq_length, d_model)
    """
    # add positional encoding to the embeddings
    x = x + self.pe[: x.size(1)].requires_grad_(False)

    # perform dropout
    return self.dropout(x)


d_model = 4
max_length = 12
dropout = 0.0

# create the positional encoding matrix
pe = PositionalEncoding(d_model, dropout, max_length)

# preview the values
print(pe.state_dict())
#print(pe(embeddings))
print(embeddings.shape)

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    This class is designed for generating positions encodings.
    The PositionalEncoding class will inherit the nn.Module class.

    """

    def __init__(self, d_model, max_sequence_length):

        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_position_encoding = torch.sin(position / denominator)
        odd_position_encoding = torch.cos(position / denominator)
        stacked = torch.stack([even_position_encoding, odd_position_encoding], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

positional_encoding = PositionalEncoding(d_model=6, max_sequence_length=10)
print(positional_encoding.forward())

def visualize_pe(positional_encoding, max_length, d_model):
    PE = positional_encoding.forward()
    plt.imshow(PE, aspect="auto")
    plt.title("Positional Encoding")
    plt.xlabel("Encoding Dimension")
    plt.ylabel("Position Index")

    # set the tick marks for the axes
    if d_model < 10:
        plt.xticks(torch.arange(0, d_model))
    if max_length < 20:
        plt.yticks(torch.arange(max_length - 1, -1, -1))

    plt.colorbar()
    plt.show()

# plot the encodings
max_length = 10
d_model = 6

visualize_pe(positional_encoding, max_length, d_model)