#!/usr/bin/env python3
"""
DeepSeek V3 Training Script with NeMo
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

import torch
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

from training.nemo.deepseek.models import DeepSeekV3NeMoModel


class SimpleDataModule(L.LightningDataModule):
    """Simple data module for testing"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
    def setup(self, stage=None):
        # Create dummy data for testing
        vocab_size = self.cfg.model.vocab_size
        seq_len = self.cfg.model.seq_len
        batch_size = self.cfg.model.micro_batch_size
        
        # Generate random token sequences
        self.train_data = torch.randint(0, vocab_size, (1000, seq_len))
        self.val_data = torch.randint(0, vocab_size, (100, seq_len))
        
    def train_dataloader(self):
        dataset = torch.utils.data.TensorDataset(self.train_data)
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.cfg.model.micro_batch_size,
            shuffle=True,
            num_workers=2
        )
        
    def val_dataloader(self):
        dataset = torch.utils.data.TensorDataset(self.val_data)
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.cfg.model.micro_batch_size,
            shuffle=False,
            num_workers=2
        )


def create_trainer(cfg):
    """Create PyTorch Lightning trainer"""
    callbacks = []
    
    # Add learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    # Add checkpoint callback only if checkpointing is enabled
    if cfg.trainer.get('enable_checkpointing', True):
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='./checkpoints',
            filename='deepseek-v3-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min',
        )
        callbacks.append(checkpoint_callback)
    
    trainer = L.Trainer(
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        max_steps=cfg.trainer.max_steps,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        limit_val_batches=cfg.trainer.limit_val_batches,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        callbacks=callbacks,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        logger=cfg.trainer.logger,
    )
    
    return trainer


def train_with_nemo(cfg):
    """Train using NeMo framework"""
    logging.info("Training with NeMo framework")
    
    # Create trainer first (required by NeMo models)
    trainer = create_trainer(cfg)
    
    # Initialize model with trainer
    model = DeepSeekV3NeMoModel(cfg, trainer=trainer)
    
    # Create data module
    data_module = SimpleDataModule(cfg)
    
    # Setup experiment manager
    exp_manager(trainer, cfg.get("exp_manager", None))
    
    # Start training
    trainer.fit(model, data_module)


@hydra_runner(config_path="../configs", config_name="deepseek_v3_base")
def main(cfg: DictConfig) -> None:
    """Main training function"""
    print(f"Training DeepSeek V3 with config: {cfg.name}")
    print(f"Model parameters:")
    print(f"  - Embedding dim: {cfg.model.emb_dim}")
    print(f"  - Layers: {cfg.model.n_layers}")
    print(f"  - Experts: {cfg.model.num_experts}")
    print(f"  - Devices: {cfg.trainer.devices}")
    
    # Set random seed for reproducibility
    L.seed_everything(42)
    
    train_with_nemo(cfg)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--config":
        # Load specific config
        config_name = sys.argv[2] if len(sys.argv) > 2 else "deepseek_v3_base"
        config_path = Path(__file__).parent.parent / "configs" / f"{config_name}.yaml"
        cfg = OmegaConf.load(config_path)
        main(cfg)
    else:
        # Use hydra runner
        main()