import torch
import torch.nn as nn

from src.blocks.encoder import Encoder_Layer
from src.layers.PositionalEmbeddings.sinosuidal import SinusoidalEmbeddingLayer

class Transformer_Decoder(nn.Module):
    def __init__(self, config):
        super(Transformer_Decoder, self).__init__()
        
        self.embedding_layer = SinusoidalEmbeddingLayer(config.vocab_size, config.embed_dim, config.max_length, config.device)
        self.layers = nn.ModuleList(Encoder_Layer(config) for _ in range(config.num_layers))
    
    def forward(self, input, mask=None):
        x = self.embedding_layer(input)

        for layer in self.layers:
            x, attention_scores = layer(x, x, x, casual_masked=False, mask=mask)
        
        return x, attention_scores  # Return the output and attention scores
    


# Example Config
class Config:
    vocab_size = 10000
    embed_dim = 512
    max_length = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_layers = 6
    n_heads = 8
    ff_dim = 2048
    dropout = 0.1