import torch
import torch.nn as nn
from encoder_layer import EncoderLayer
from embedding import EmbeddingLayer

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, max_len, dropout_rate, num_heads, d_ff, output_size):
        super(Transformer, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size, d_model, max_len, dropout_rate)
        self.encoders = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]
            )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(d_model, output_size)
    def forward(self, x, mask=None):
        #batch_size = x.size(0)
        x = self.embedding(x)
        for encoder in self.encoders:
            x = encoder(x, mask)
        x = x.mean(dim=1)
        x = self.fc(self.dropout(x))
        return x