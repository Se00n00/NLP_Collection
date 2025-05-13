import torch
import torch.nn as nn

# Adding Positional Information using indicies Information, though lacks relative positional information
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size, max_length):
        super(EmbeddingLayer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size, eps=1e-12)
    
    def forward(self, x):
        word_embedding = self.embedding(x)                                                  # Convert unique word tokens to word embeddings
        
        positional_indices = torch.arange(x.size(-2), device=x.device).unsqueeze(0)         # Creates positional inidices tensor                             Shape: (1, Seqlen)
        positional_embeddings = self.position_embedding(positional_indices)                 # Convert positional indicies to positional embeddings          Shape: (1, Seqlen, embed_size)  
        
        x = word_embedding + positional_embeddings                                          # Adds word embedding to positional embedding
        x = self.layer_norm(x)                                                              # Apply layer normalization
        return x