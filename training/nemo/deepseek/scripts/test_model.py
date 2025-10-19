#!/usr/bin/env python3
"""
Simple test script for DeepSeek V3 NeMo model
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

import torch
from omegaconf import OmegaConf
import lightning as L
from training.nemo.deepseek.models import DeepSeekV3NeMoModel


def test_model():
    """Test the DeepSeek V3 model"""
    print("Testing DeepSeek V3 NeMo Model...")
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "deepseek_v3_base.yaml"
    cfg = OmegaConf.load(config_path)
    
    # Create smaller config for testing
    cfg.model.emb_dim = 64
    cfg.model.n_layers = 2
    cfg.model.num_heads = 4
    cfg.model.seq_len = 8
    cfg.model.num_experts = 4
    cfg.model.expert_dim = 128
    
    print(f"Model config:")
    print(f"  - Embedding dim: {cfg.model.emb_dim}")
    print(f"  - Layers: {cfg.model.n_layers}")
    print(f"  - Experts: {cfg.model.num_experts}")
    print(f"  - Sequence length: {cfg.model.seq_len}")
    
    try:
        # Create trainer (required for NeMo models)
        trainer = L.Trainer(
            devices=1,
            accelerator='cpu',
            max_epochs=1,
            logger=False,
            enable_checkpointing=False
        )
        
        # Create model
        model = DeepSeekV3NeMoModel(cfg, trainer=trainer)
        print("✅ Model created successfully")
        
        # Test forward pass
        batch_size = 2
        seq_len = cfg.model.seq_len
        vocab_size = cfg.model.vocab_size
        
        # Create test input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        print(f"Input shape: {input_ids.shape}")
        
        # Forward pass
        with torch.no_grad():
            if hasattr(model, 'model'):
                # NeMo model
                logits = model.model(input_ids)
            else:
                # Fallback model
                logits = model(input_ids)
        
        expected_shape = (batch_size, seq_len, vocab_size)
        print(f"Output shape: {logits.shape}")
        print(f"Expected shape: {expected_shape}")
        
        if logits.shape == expected_shape:
            print("✅ Forward pass successful!")
            print(f"Sample logits: {logits[0, 0, :5]}")
            return True
        else:
            print("❌ Shape mismatch!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n🎉 All tests passed! DeepSeek V3 NeMo model is working!")
    else:
        print("\n❌ Tests failed. Check the error messages above.")