import torch
import torch.nn as nn
import math

# Adding positional Information using sinusoidal function
# class SinusoidalEmbeddingLayer(nn.Module):
#     def __init__(self, vocab_size, embed_size, max_length, device):
#         super(SinusoidalEmbeddingLayer, self).__init__()

#         self.embedding = nn.Embedding(vocab_size, embed_size)
        
#         # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
#         self.register_buffer("positional_embedding", self._get_positional_encoding(max_length, embed_size, device))
#         self.layer_norm = nn.LayerNorm(embed_size, eps=1e-12)
    
#     def _get_positional_encoding(self, max_length, embed_size, device):
#         pe = torch.zeros(max_length, embed_size, device=device)                              # Create a tensor of zeros of size (max_length, embed_size)
#         position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)              # Create a tensor of size (max_length, 1)
#         div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))    # Create a tensor of exp values of 0 to embed_size/2
        
#         pe[:, 0::2] = torch.sin(position * div_term)                                          # Apply sin function to even indices, start=0 , step=2
#         pe[:, 1::2] = torch.cos(position * div_term)                                          # Apply cos function to odd indices, start=1, step=2
#         pe = pe.unsqueeze(0)                                                                  # shape: (1, max_length, embed_size)
#         return pe

#     def forward(self, x):
#         x = x.long()
#         word_embedding = self.embedding(x)                                                  # Convert unique word tokens to word embeddings
        
#         positional_embeddings = self.positional_embedding[:, :x.size(-2), :].to(x.device)   # Get sinosudal indicies information as positional embeddings          Shape: (1, Seqlen, embed_size)
#         x = word_embedding + positional_embeddings                                          # Adds word embedding to positional embedding
#         x = self.layer_norm(x)                                                              # Apply layer normalization
#         return x
    
class SinusoidalEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size, max_length, device):
        super(SinusoidalEmbeddingLayer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        self.register_buffer("positional_embedding", self._get_positional_encoding(max_length, embed_size, device))
        self.layer_norm = nn.LayerNorm(embed_size, eps=1e-12)
    
    def _get_positional_encoding(self, max_length, embed_size, device):
        pe = torch.zeros(max_length, embed_size, device=device)                              # Create a tensor of zeros of size (max_length, embed_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)              # Create a tensor of size (max_length, 1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))    # Create a tensor of exp values of 0 to embed_size/2
        
        pe[:, 0::2] = torch.sin(position * div_term)                                          # Apply sin function to even indices, start=0 , step=2
        pe[:, 1::2] = torch.cos(position * div_term)                                          # Apply cos function to odd indices, start=1, step=2
        pe = pe.unsqueeze(0)                                                                  # shape: (1, max_length, embed_size)
        return pe

    def forward(self, x):
        x = x.long()
        word_embedding = self.embedding(x)                                                  # Convert unique word tokens to word embeddings
        
        positional_embeddings = self.positional_embedding[:, :x.size(1), :].to(x.device)   # Get sinosudal indicies information as positional embeddings          Shape: (1, Seqlen, embed_size)
        x = word_embedding + positional_embeddings                                          # Adds word embedding to positional embedding
        x = self.layer_norm(x)                                                              # Apply layer normalization
        return x