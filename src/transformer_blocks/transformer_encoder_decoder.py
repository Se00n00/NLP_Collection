import torch
import torch.nn as nn 

from src.blocks.encoder import Encoder_Layer
from src.blocks.encoder_decoder import Encoder_Decoder_Layer
from src.layers.PositionalEmbeddings.sinosuidal import SinusoidalEmbeddingLayer

class Transformer_Encoder_Decoder(nn.Module):
    def __init__(self, config):
        super(Transformer_Encoder_Decoder, self).__init__()
        
        self.encoder_embedding_layer = SinusoidalEmbeddingLayer(config.encoder_vocab_size, config.embed_dim, config.max_length, config.device)
        self.decoder_embedding_layer = SinusoidalEmbeddingLayer(config.decoder_vocab_size, config.embed_dim, config.max_length, config.device)
        
        self.encoder_layers = nn.ModuleList(Encoder_Layer(config) for _ in range(config.num_layers))
        self.decoder_layers = nn.ModuleList(Encoder_Decoder_Layer(config) for _ in range(config.num_layers))
        self.normalization = nn.LayerNorm(config.embed_dim, eps=1e-12)

    def forward(self, decoder_input, encoder_input, decoder_mask=None, encoder_mask=None, coupled=False):

        encoder_pe = self.encoder_embedding_layer(decoder_input)    # Encoder positional embeddings
        decoder_pe = self.decoder_embedding_layer(encoder_input)    # Decoder positional embeddings

        if coupled:
            for encoder_layer, decoder_layer in zip(self.encoder_layers, self.decoder_layers):
                encoder_pe, attention_scores1 = encoder_layer(encoder_pe)                                                # Encoder Layer
                decoder_pe, attention_scores2 = decoder_layer(decoder_pe, encoder_pe)  

        else:
            for encoder_layer in self.encoder_layers:
                encoder_pe, attention_scores1 = encoder_layer(encoder_pe)

            for decoder_layer in self.decoder_layers:
                decoder_pe, attention_scores2 = decoder_layer(decoder_pe, encoder_pe)
                                                        
        output = self.normalization(decoder_pe)
        return output, attention_scores2


# Example Config
class Config:
    decoder_vocab_size = 10000
    encoder_vocab_size = 10000
    embed_dim = 512
    max_length = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_layers = 6
    n_heads = 8
    ff_dim = 2048
    dropout = 0.1