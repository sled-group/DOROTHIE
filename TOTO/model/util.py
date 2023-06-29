import os
import torch
import math
from torch import nn


class PosEncoding(nn.Module):
    '''
    Transformer-style positional encoding with wavelets
    '''
    def __init__(self, dropout: float = 0.0, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len)
        position = torch.arange(0, max_len, 2)
        pe[ 0::2] = torch.sin(position )
        pe[ 1::2] = torch.cos(position)
        pe=pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self,x,idx):
        # print(idx)
        x = x + self.pe[idx,:]
        return self.dropout(x)