import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Multi_Head_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Multi_Head_Attention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections
        self.WQ = nn.Linear(embed_dim, embed_dim)
        self.WK = nn.Linear(embed_dim, embed_dim)
        self.WV = nn.Linear(embed_dim, embed_dim)
        self.WO = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, q, k, v, casual_masked=False, mask=None):
        batch_size = q.size(0)
        q_len, k_len, v_len = q.size(1), k.size(1), v.size(1)
        
        # Linear projections
        Q = self.WQ(q)  # [B, L_q, E]
        K = self.WK(k)  # [B, L_k, E]
        V = self.WV(v)  # [B, L_v, E]
        
        # Reshape for multi-head: [B, T, E] → [B, H, L, D]
        Q = Q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, v_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # Scaled dot-product attention [B, H, L, L]
        
        # Optional Assertion
        assert q_len == k_len, "Query and Key lengths must be equal for self-attention"

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            scores = scores.masked_fill(mask == 0, float('-inf'))


        if casual_masked:
            casual_mask = torch.triu(torch.ones(q_len, k_len, device=q.device), diagonal=1).bool() # Create causal mask (for decoder self-attention)
            casual_mask = casual_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, L_q, L_k]
            scores = scores.masked_fill(casual_mask, float('-inf'))
        
        
        
        attention_scores = F.softmax(scores, dim=-1)  # Attention Scores[B, H, T_q, T_k]
        attention_output = torch.matmul(attention_scores, V)  # [B, H, T_q, D]

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.embed_dim)# Concatenate heads: [B, H, T, D] → [B, T, E]
        output = self.WO(attention_output)
        
        return output, attention_scores