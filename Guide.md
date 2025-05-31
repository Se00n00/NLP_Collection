# Introduction to Guide
This Markdown File is the guide to understand and implement wide range of NLP task
This Guide would help you to implement those task on the basis of order of topices according to my arrangement so its better to stick to the sequential order.

```
  Feel free to open issues and pull requests to help improve this project and guide
```
### Things to Understand
```
  Tokenizer
    Model
      Trainer
        Inference
          Evaluation
            FineTuning & Benchmarking
```

# Implementing Transformer and Its Varients
[Skip to all Implementation at same place?](https://github.com/Se00n00/Paper_Implementation_and_Reviews/blob/main/Language/attention_is_all_you_need.ipynb)
### MHA: Multi-Head Attention
This is the core of transformer. [Guide to Implement: Attention in Transformer](https://www.deeplearning.ai/short-courses/attention-in-transformers-concepts-and-code-in-pytorch/)

Here is more simpler Implementation but complex to understand
```Python
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
```
### FFD: Point-Wise Feed Forward Network
This Layer apply Feed-forward to each token embedding sepeartly
```Python
  
class FeedForward(nn.Module):
    def __init__(self, model_dim, ff_dim, droupout=0.5):    # ff_dim is usally higher that model_dim
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, model_dim)
        self.relu = nn.ReLU()
        self.droupout = nn.Dropout(droupout)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        x = self.droupout(x)       # apply dropout to the output of the second linear layer to reduce overfitting
        return x
```
### Position Embeddings: Linear or Sinosuidal
```Python
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
```
