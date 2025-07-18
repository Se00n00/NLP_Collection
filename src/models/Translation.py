import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from src.blocks.encoder import Encoder_Layer
from src.blocks.encoder_decoder import Encoder_Decoder_Layer
from src.layers.PositionalEmbeddings.sinosuidal import SinusoidalEmbeddingLayer

class TranslationModel(nn.Module):
    def __init__(self, config):
        super(TranslationModel, self).__init__()

        self.src_embedding_layer = SinusoidalEmbeddingLayer(config.src_vocab_size, config.embed_dim, config.max_length, config.device)
        self.tgt_embedding_layer = SinusoidalEmbeddingLayer(config.tgt_vocab_size, config.embed_dim, config.max_length, config.device)
        
        self.decoder_layers = nn.ModuleList([Encoder_Decoder_Layer(config) for _ in range(config.num_layers)])
        self.encoder_layers = nn.ModuleList([Encoder_Layer(config) for _ in range(config.num_layers)])

        self.fc_out = nn.Linear(config.embed_dim, config.tgt_vocab_size)        # Final linear layer
    
    def forward(self, source_input, target_input, Coupled=False):
        src = self.src_embedding_layer(source_input)                                    # Shape: (batch, seq_len, embed_dim)
        tgt = self.tgt_embedding_layer(target_input)                                    # Shape: (batch, seq_len, embed_dim)

        if Coupled:
            for encoder_layer, decoder_layer in zip(self.encoder_layers, self.decoder_layers):
                src, attention_output_1 = encoder_layer(src)                                                # Encoder Layer
                tgt, attention_output_2 = decoder_layer(tgt, src)   
        
        else:
            for encoder_layer in self.encoder_layers:
                src, attention_output_1 = encoder_layer(src)

            for decoder_layer in self.decoder_layers:
                tgt, attention_output_2 = decoder_layer(tgt, src)

        output = self.fc_out(tgt)
        return output, attention_output_1, attention_output_2