from transformers import PretrainedConfig

class ModelConfig(PretrainedConfig):
    def __init__(self, 
                 src_vocab_size=50257,
                 tgt_vocab_size=50257,
                 embed_dim=512, 
                 max_length=512, 
                 device="cpu",
                 num_layers=6,
                 n_heads=8,
                 ff_dim=1024,
                 dropout=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.device=device
        self.num_layers=num_layers
        self.n_heads=n_heads
        self.ff_dim=ff_dim
        self.dropout=dropout