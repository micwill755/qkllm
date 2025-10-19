from transformers.configuration_utils import PretrainedConfig

class DeepSeekV3Config(PretrainedConfig):
    model_type = "deepseek_v3"
    
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        num_hidden_layers=61,
        num_attention_heads=12,
        max_position_embeddings=1024,
        num_experts=8,
        num_experts_per_tok=2,
        intermediate_size=2048,
        latent_dim=64,
        rms_norm_eps=1e-5,
        rope_base=10000,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.intermediate_size = intermediate_size
        self.latent_dim = latent_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_base = rope_base
        super().__init__(**kwargs)