import torch
import torch.nn as nn
import torch.nn.functional as F 
from positional_encoding import PositionalEncoding


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_len, d_model, max_len=5000, dropout_rate=0.1):
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.embedding = nn.Embedding(vocab_len, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
    def forward(self, x):    
        #return self.dropout(self.embedding(x) + self.pe(x))
        x = self.embedding(x)
        x = self.pe(x)
        return self.dropout(x)
# d_model = 512
# max_len = 100
# vocab_len = 1000
# x = torch.tensor([1,2,3,4,5])
# x = F.pad(x,(0, max_len-x.size(0)), value=0)
# print(x.size())
# em = embedding(vocab_len, d_model, 0.1, max_len).forward(x)
# print(em)
