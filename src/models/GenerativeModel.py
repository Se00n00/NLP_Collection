import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from src.blocks.decoder import Decoder_Layer
from src.layers.PositionalEmbeddings.sinosuidal import SinusoidalEmbeddingLayer

class GenerativeModel(nn.Module):
    def __init__(self, config):
        super(GenerativeModel, self).__init__()

        self.embedding_layer = SinusoidalEmbeddingLayer(config.vocab_size, config.embed_dim, config.max_length, config.device)
        
        self.decoder_layers = nn.ModuleList([Decoder_Layer(config) for _ in range(config.num_layers)])
        
        self.fc_out = nn.Linear(config.embed_dim, config.vocab_size)        # Final linear layer
    
    def forward(self, input):
        input = self.embedding_layer(input)                                    # Shape: (batch, seq_len, embed_dim)

        for decoder_layer in self.decoder_layers:
            input, attention_output = decoder_layer(input)

        output = self.fc_out(input)
        return output, attention_output