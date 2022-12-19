import torch
import torch.nn as nn
import torch.nn.functional as f

import math


class SelfAttention(nn.Module):
  def __init__(self, embed_size):
    super(SelfAttention, self).__init__()

    self.embed_size = embed_size
    self.embedding = nn.Sequential(
        nn.Linear(1, embed_size),
        nn.ReLU(),)
    
    self.query = nn.Linear(embed_size, embed_size)
    self.key = nn.Linear(embed_size, embed_size)
    self.value = nn.Linear(embed_size, embed_size)


  def forward(self, x):
    embed_size = self.embed_size
    x = self.embedding(x)
    
    query = self.query(x)
    key = self.key(x)
    value = self.value(x)

    alingment = torch.matmul(query, key.t())

    weights = f.softmax(torch.div(alingment[0],math.sqrt(embed_size)), dim=0)
    weights = weights.reshape(weights.size()[0], 1)
    weighted_values = torch.mul(weights, value)

    return weighted_values

attention = SelfAttention(50)
