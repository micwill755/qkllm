import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Optional
import lightning as L

from .components import DeepSeekBlock, RMSNorm


class DeepSeekV3Core(nn.Module):
    """Core DeepSeek V3 model architecture (pure PyTorch)"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Token embedding
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DeepSeekBlock(
                emb_dim=cfg.emb_dim,
                num_heads=cfg.num_heads,
                seq_len=cfg.seq_len,
                num_experts=cfg.num_experts,
                top_k=cfg.top_k,
                expert_dim=cfg.expert_dim
            ) for _ in range(cfg.n_layers)
        ])
        
        # Final layer norm and output projection
        self.final_rms_norm = RMSNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
        
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Token embedding
        x = self.tok_emb(input_ids)  # (batch, seq_len, emb_dim)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final normalization and output projection
        x = self.final_rms_norm(x)
        logits = self.out_head(x)  # (batch, seq_len, vocab_size)
        
        return logits


class DeepSeekV3NeMoModel(MegatronGPTModel):
    """DeepSeek V3 model integrated with NeMo framework"""
    
    def __init__(self, cfg: DictConfig, trainer=None):
        # Call parent constructor first
        super().__init__(cfg, trainer=trainer)
        
        # Initialize core model
        self.core_model = DeepSeekV3Core(cfg.model)
        
    def forward(self, tokens, text_position_ids, attention_mask, labels=None):
        """Forward pass compatible with NeMo training loop"""
        logits = self.core_model(tokens, attention_mask)
        
        if labels is not None:
            # Calculate loss for training
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss
        
        return logits
        
    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning"""
        tokens = batch['tokens']
        labels = batch.get('labels', tokens)
        attention_mask = batch.get('attention_mask', None)
        text_position_ids = batch.get('text_position_ids', None)
        
        loss = self.forward(tokens, text_position_ids, attention_mask, labels)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning"""
        tokens = batch['tokens']
        labels = batch.get('labels', tokens)
        attention_mask = batch.get('attention_mask', None)
        text_position_ids = batch.get('text_position_ids', None)
        
        loss = self.forward(tokens, text_position_ids, attention_mask, labels)
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
        
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay
        )
        return optimizer
        
    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """List available pretrained models"""
        return None
        
    def setup_optimizer_param_groups(self):
        """Setup parameter groups for optimizer"""
        return [{'params': self.parameters()}]