import torch
import torch.nn as nn

from src.layers.Attention.multihead import Multi_Head_Attention
from src.layers.PostAttention.feedforward import FeedForward

class Encoder_Decoder_Layer(nn.Module):
    def __init__(self, config):
        super(Encoder_Decoder_Layer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(config.embed_dim, eps=1e-12)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim, eps=1e-12)
        self.layer_norm3 = nn.LayerNorm(config.embed_dim, eps=1e-12)
        self.masked_multi_head_attention = Multi_Head_Attention(config.embed_dim, config.n_heads)
        self.multi_head_attention = Multi_Head_Attention(config.embed_dim, config.n_heads)
        self.feed_forward = FeedForward(config.embed_dim, config.ff_dim, config.dropout)
    
    def forward(self, x, encoder_output, encoder_mask=None, decoder_mask=None):
        norm1 = self.layer_norm1(x)                                         # Apply layer1 normalization
        attention_output, attention_scores = self.masked_multi_head_attention(norm1, norm1, norm1, casual_masked=True, mask=decoder_mask)
        x = x + attention_output                                        # Residual connection
        
        norm2 = self.layer_norm2(x)                                         # Apply layer2 normalization
        attention_output, attention_scores = self.multi_head_attention(norm2, encoder_output, encoder_output, casual_masked=False, mask=encoder_mask)
        x = x + attention_output                                        # Residual connection
        
        norm3 = self.layer_norm3(x)                                         # Apply layer3 normalization
        x = x + self.feed_forward(norm3)                                    # Residual connection
        return x, attention_scores                                      # Return the output and attention scores