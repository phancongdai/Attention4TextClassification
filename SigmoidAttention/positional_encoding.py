import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros((self.max_len, self.d_model), dtype=torch.float)
        positon = torch.arange(0, self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-(math.log(10000)/self.d_model)))
        pe[:,0::2] = torch.sin(positon*div_term)
        pe[:,1::2] = torch.cos(positon*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# def positional_encoding(d_model, max_len):
#     pe = torch.zeros((max_len, d_model), dtype=torch.float)
#     positon = torch.arange(0, max_len).unsqueeze(1)
#     div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-(math.log(10000)/d_model)))
#     pe[:,0::2] = torch.sin(positon*div_term)
#     pe[:,1::2] = torch.cos(positon*div_term)
#     pe = pe.unsqueeze(0)
#     return pe


    
# #Test positional encoding

# # Assuming the positional_encoding function is already defined above.

# def test_shape_output():
#     """Test if the output tensor has the correct shape"""
#     d_model = 512
#     max_len = 100
#     pe = positional_encoding(d_model, max_len)
#     #print(pe.shape)
#     assert pe.shape == (1, max_len, d_model), f"Expected shape (1, {max_len}, {d_model}), got {pe.shape}"

# def test_single_dimension():
#     """Test positional encoding with d_model=2 and max_len=1"""
#     d_model = 2
#     max_len = 1
#     pe = positional_encoding(d_model, max_len)
#     expected_pe = torch.tensor([[[0., 1.]]])
#     assert torch.allclose(pe, expected_pe), "Output does not match expected value for d_model=2, max_len=1"

# def test_small_sequence():
#     """Test positional encoding with d_model=4 and max_len=2"""
#     d_model = 4
#     max_len = 2
#     pe = positional_encoding(d_model, max_len)
#     position = torch.tensor([[0.], [1.]])
#     div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#     expected_pe = torch.zeros(max_len, d_model)
#     expected_pe[:, 0::2] = torch.sin(position * div_term)
#     expected_pe[:, 1::2] = torch.cos(position * div_term)
#     expected_pe = expected_pe.unsqueeze(0)
#     assert torch.allclose(pe, expected_pe), "Output does not match expected value for small sequence"

# def test_zero_position():
#     """Test if the first row (position 0) is all zeros"""
#     d_model = 6
#     max_len = 10
#     pe = positional_encoding(d_model, max_len)
#     assert torch.allclose(pe[0, 0, ::2], torch.zeros(d_model // 2)), "Sine values at position 0 are not zero"
#     assert torch.allclose(pe[0, 0, 1::2], torch.ones(d_model // 2)), "Cosine values at position 0 are not one"

# def test_even_dimension():
#     """Test if positional encoding is correct for an even dimension"""
#     d_model = 4
#     max_len = 5
#     pe = positional_encoding(d_model, max_len)
#     position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#     div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#     expected_pe = torch.zeros(max_len, d_model)
#     expected_pe[:, 0::2] = torch.sin(position * div_term)
#     expected_pe[:, 1::2] = torch.cos(position * div_term)
#     expected_pe = expected_pe.unsqueeze(0)
#     assert torch.allclose(pe, expected_pe), "Output does not match expected value for even dimension"


# def test_high_dimension():
#     """Test positional encoding with a high dimension"""
#     d_model = 128
#     max_len = 50
#     pe = positional_encoding(d_model, max_len)
#     assert pe.shape == (1, max_len, d_model), f"Expected shape (1, {max_len}, {d_model}), got {pe.shape}"

# def test_known_output_small():
#     """Test positional encoding for small known output"""
#     d_model = 2
#     max_len = 2
#     pe = positional_encoding(d_model, max_len)
#     expected_pe = torch.tensor([[[0., 1.], [0.8415, 0.5403]]])  # Manually calculated values for sine and cosine
#     #print(pe)
#     #print(expected_pe)
#     assert torch.allclose(pe, expected_pe, atol=1e-4), "Output does not match expected small known output"

# def test_known_output_large():
#     """Test positional encoding for larger known output"""
#     d_model = 4
#     max_len = 3
#     pe = positional_encoding(d_model, max_len)
#     expected_pe = torch.tensor([[[0.0000, 1.0000, 0.0000, 1.0000],
#                                  [0.8415, 0.5403, 0.0084, 0.9999],
#                                  [0.9093, -0.4161, 0.0168, 0.9998]]])  # Manually calculated values for sine and cosine
#     assert torch.allclose(pe, expected_pe, atol=1e-2), "Output does not match expected large known output"

# def test_large_sequence():
#     """Test if the function handles large sequences"""
#     d_model = 512
#     max_len = 10000
#     pe = positional_encoding(d_model, max_len)
#     assert pe.shape == (1, max_len, d_model), f"Expected shape (1, {max_len}, {d_model}), got {pe.shape}"

# def test_stability_with_large_inputs():
#     """Test if the function is stable with large inputs"""
#     d_model = 512
#     max_len = 10000
#     pe = positional_encoding(d_model, max_len)
#     assert not torch.isnan(pe).any(), "Output contains NaN values"
#     assert not torch.isinf(pe).any(), "Output contains Inf values"

# test_shape_output()
# test_single_dimension()
# test_small_sequence()
# test_zero_position()
# test_even_dimension()
# test_high_dimension()
# test_known_output_small()
# test_known_output_large()
# test_large_sequence()
# test_stability_with_large_inputs()

# print("All tests passed!")
