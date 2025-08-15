#!/usr/bin/env python3
"""
Test script to check model memory usage before training
"""
import torch
from model import ModelConfig, Transformer

def test_model_size():
    # Test nano-gpt config
    config = ModelConfig(
        # Core dimensions - MUCH smaller
        vocab_size=50257,             # Standard GPT-2 vocab size
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=3072,       # 4 * hidden_size
        
        # Attention
        num_attention_heads=12,
        num_key_value_heads=12,       # No GQA, so n_kv_head == n_head
        head_dim=64,                  # hidden_size / num_attention_heads = 768 / 12 = 64
        
        # Context & Position - MUCH smaller  
        max_position_embeddings=1024, # Small context length
        sliding_window=128,
        
        # MoE - Disable MoE completely
        num_local_experts=1,          # A "1-expert MoE" is just a standard dense FFN
        experts_per_token=1,
        
        # For a non-SWA model, all layers are full attention
        layer_types=['full_attention'] * 12,
        
        # Other reasonable defaults
        enable_sink_token=False,
        attention_bias=True,
        tie_word_embeddings=False,
    )
    
    print("Creating model with nano-gpt config...")
    print(f"Config: {config}")
    
    try:
        model = Transformer(config)
        print("✅ Model created successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        
        # Try moving to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Moving model to {device}...")
        model = model.to(device)
        print(f"✅ Model successfully moved to {device}")
        
        # Test forward pass with small batch
        print("Testing forward pass...")
        batch_size, seq_len = 2, 256
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        
        with torch.no_grad():
            output = model(input_ids)
            print(f"✅ Forward pass successful! Output shape: {output[0].shape}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_model_size()
    if success:
        print("\n🎉 Model size test passed! You should be able to train this model.")
    else:
        print("\n💥 Model size test failed! The model is still too large.")
