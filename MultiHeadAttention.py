import torch
import torch.nn as nn
import torch.nn.functional as f
import math


class Multi_head_attention(nn.Module):
  def __init__(self, embed_size, n_heads):
    super(Multi_head_attention, self).__init__()

    self.embed_size = embed_size
    self.n_heads = n_heads
    self.head_dim = embed_size // n_heads

    assert(self.head_dim * n_heads == embed_size)

    self.query = nn.Linear(embed_size, embed_size)
    self.key = nn.Linear(embed_size, embed_size)
    self.value = nn.Linear(embed_size, embed_size)

  def forward(self, x, value=None, query=None, key=None, mask=None):
    embed_size = self.embed_size 
    head_dim = self.head_dim
    n_heads = self.n_heads

    if None in [query, key, value]:
      query = self.query(x)
      key = self.key(x)
      value = self.value(x)
    
    else:
      query = self.query(query)
      key = self.key(key)
      value = self.value(value)

    query = query.reshape(x.size()[0], head_dim, n_heads)
    key = key.reshape(x.size()[0], head_dim, n_heads)
    value = value.reshape(x.size()[0], head_dim, n_heads)

    alingment = torch.matmul(torch.transpose(key, 1, 2), query)[:,0]
    scaled_alingment = torch.div(alingment, (embed_size ** .5))

    if mask is not None:
      scaled_alingment = scaled_alingment.masked_fill(
          mask == 0, float(1e-20))

    weights = f.softmax(scaled_alingment, 1) 
    weighted_values = torch.einsum('sij,sj->sij', [value,weights])
    weighted_values = weighted_values.reshape(
        weighted_values.size()[0], head_dim * n_heads)

    return weighted_values
