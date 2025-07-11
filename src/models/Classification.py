import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from src.blocks.encoder import Encoder_Layer
from src.blocks.decoder import Decoder_Layer
from src.layers.PositionalEmbeddings.sinosuidal import SinusoidalEmbeddingLayer

class ClassificationModel(nn.Module):
    def __init__(self, config):
        super(ClassificationModel, self).__init__()

        self.embedding_layer = SinusoidalEmbeddingLayer(config.vocab_size, config.embed_dim, config.max_length, config.device)
        
        self.encoder_layers = nn.ModuleList([Encoder_Layer(config) for _ in range(config.num_layers)])
        
        self.fc_out = nn.Linear(config.embed_dim, config.num_classes)        # Final linear layer
    
    def forward(self, input, mask=None):
        input = self.embedding_layer(input)                                    # Shape: (batch, seq_len, embed_dim)

        for encoder_layer in self.encoder_layers:
            input, attention_output = encoder_layer(input, mask=mask)

        output = self.fc_out(input)
        return output, attention_output
