import torch 
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model//num_heads
        assert d_model%num_heads==0, "Valid number of heads! d_model must be divisible by num_heads "
        #Learnable weight matrices for query, key, value
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2) #(batch_size, num_heads, seq_len, depth)
    
    def combine_heads(self, x, batch_size):
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.d_model)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        b = -torch.log(torch.tensor(seq_len, dtype=torch.float))
        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        dk = K.size(-1)
        qk = torch.matmul(Q, K.transpose(-2,-1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float))
        if mask is not None:
            qk = qk.masked_fill(mask==0, torch.float(-1e9))
        attention_weights = torch.sigmoid(qk+b)
        output = torch.matmul(attention_weights, V)
        output = self.combine_heads(output, batch_size)

        output = self.W_o(output)
        return output