"""
Quick script to estimate model parameter count before training.
"""
from train import get_config

def estimate_params(config):
    """Rough parameter estimation"""
    H = config.hidden_size
    V = config.vocab_size
    L = config.num_hidden_layers
    E = config.num_local_experts
    FF = config.intermediate_size
    
    # Embeddings
    embed_params = V * H
    
    # Per layer
    # Attention: Q, K, V, O projections
    attn_params_per_layer = (
        H * (config.num_attention_heads * config.head_dim) +  # Q
        H * (config.num_key_value_heads * config.head_dim) +  # K  
        H * (config.num_key_value_heads * config.head_dim) +  # V
        (config.num_attention_heads * config.head_dim) * H    # O
    )
    
    # MoE per layer
    router_params_per_layer = H * E
    expert_params_per_layer = E * (H * 2*FF + 2*FF + FF * H + H)  # ffn_in + bias + ffn_out + bias
    
    # Layer norms
    norm_params_per_layer = 2 * H  # 2 norms per layer
    
    # Total per layer
    params_per_layer = attn_params_per_layer + router_params_per_layer + expert_params_per_layer + norm_params_per_layer
    
    # Final norm
    final_norm = H
    
    # LM head
    lm_head = H * V if not config.tie_word_embeddings else 0
    
    total = embed_params + L * params_per_layer + final_norm + lm_head
    
    return {
        "total_M": total / 1e6,
        "total_B": total / 1e9,
        "embed_M": embed_params / 1e6,
        "per_layer_M": params_per_layer / 1e6,
        "expert_params_M": (E * (H * 2*FF + 2*FF + FF * H + H)) / 1e6
    }

if __name__ == "__main__":
    configs = ['nano-gpt', 'gpt-oss-5b', 'gpt-oss-120b']
    
    for config_name in configs:
        print(f"\n=== {config_name} ===")
        config = get_config(config_name)
        params = estimate_params(config)
        print(f"Estimated total parameters: {params['total_M']:.1f}M ({params['total_B']:.2f}B)")
        print(f"  - Embeddings: {params['embed_M']:.1f}M")  
        print(f"  - Per layer: {params['per_layer_M']:.1f}M")
        print(f"  - Expert params per layer: {params['expert_params_M']:.1f}M")
        print(f"Config: H={config.hidden_size}, L={config.num_hidden_layers}, E={config.num_local_experts}, FF={config.intermediate_size}")
