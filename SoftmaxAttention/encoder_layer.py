import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from multi_heads_attention import MultiHeadAttention
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.att = MultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm2 = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        mtha_out = self.att(x, mask)
        x = self.layer_norm1(x+self.dropout1(mtha_out))
        ff_out = self.feedforward(x)
        x = self.layer_norm2(x+self.dropout2(ff_out))
        return x