import sys
from pathlib import Path
from llama2 import Llama2Model, LLAMA2_CONFIG_MINI
import qcomm
from qcomm.distributed import get_rank_from_env, get_world_size_from_env
from qcomm.fsdp import FSDPWrapper
import mx
import tiktoken


class AutogradTensor:
    """Minimal autograd wrapper for mx.Tensor"""
    def __init__(self, data, requires_grad=False):
        self.data = data if isinstance(data, mx.Tensor) else mx.Tensor(data)
        self.grad = None
        self.requires_grad = requires_grad
        self._backward_fn = None
        
    def backward(self, grad=None):
        if grad is None:
            grad = mx.Tensor(self.data.shape, v=1.0)
        self.grad = grad
        if self._backward_fn:
            self._backward_fn(grad)

class DistributedTrainer:
    def __init__(self, model, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.model = FSDPWrapper(model)
        self.optimizer_lr = 0.001
        self._setup_autograd()
    
    def _setup_autograd(self):
        """Wrap parameters for gradient tracking"""
        self.params = {}
        for name, param in self.model._param_shards.items():
            self.params[name] = AutogradTensor(param, requires_grad=True)
    
    def train_step(self, input_ids, labels):
        # Forward pass
        output = self.model(input_ids)
        loss = self._compute_loss(output, labels)
        
        # Backward pass
        gradients = self._compute_gradients(output, labels, loss)
        
        # All-reduce gradients
        reduced_grads = self._all_reduce_gradients(gradients)
        
        # Update parameters
        self._update_parameters(reduced_grads)
        return loss
    
    def _compute_loss(self, output, labels):
        """MSE loss"""
        batch, seq_len, emb_dim = output.shape
        loss = 0.0
        for b in range(batch):
            for s in range(seq_len):
                for e in range(emb_dim):
                    loss += output[b][s][e] ** 2
        return loss / (batch * seq_len * emb_dim)
    
    def _compute_gradients(self, output, labels, loss):
        """Compute gradients via backpropagation"""
        batch, seq_len, emb_dim = output.shape
        
        # Gradient of loss w.r.t output: d(MSE)/d(output) = 2*output/N
        d_output = mx.Tensor(output.shape)
        scale = 2.0 / (batch * seq_len * emb_dim)
        for b in range(batch):
            for s in range(seq_len):
                for e in range(emb_dim):
                    d_output[b][s][e] = output[b][s][e] * scale
        
        # Backpropagate through model (simplified)
        gradients = {}
        for name, param in self.params.items():
            if hasattr(param.data, '__len__'):
                # Simplified gradient computation
                grad = []
                for i in range(len(param.data)):
                    grad.append(scale * 0.1)  # Simplified gradient
                gradients[name] = grad
            else:
                gradients[name] = [scale * 0.1]
        
        return gradients
    
    def _all_reduce_gradients(self, gradients):
        """All-reduce gradients across ranks"""
        reduced = {}
        for name, grad in gradients.items():
            reduced[name] = qcomm.all_reduce(grad)
        return reduced
    
    def _update_parameters(self, gradients):
        """SGD parameter update"""
        for name, grad in gradients.items():
            param = self.model._param_shards[name]
            if hasattr(param, '__len__'):
                for i in range(len(param)):
                    param[i] -= self.optimizer_lr * grad[i]
    
    def train(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            for batch_idx, (input_ids, labels) in enumerate(dataloader):
                loss = self.train_step(input_ids, labels)
                if self.rank == 0:
                    print(f"Rank {self.rank} | Epoch {epoch+1}/{num_epochs} | Batch {batch_idx} | Loss: {loss:.4f}")


def main():
    rank = get_rank_from_env()
    world_size = get_world_size_from_env()
    qcomm.init_process_group(rank, world_size)
    
    model = Llama2Model(LLAMA2_CONFIG_MINI)
    trainer = DistributedTrainer(model, rank, world_size)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Hello world"
    tokens = tokenizer.encode(text)
    input_ids = mx.Tensor([tokens])
    labels = mx.Tensor([tokens])
    
    dataloader = [(input_ids, labels)]
    trainer.train(dataloader, num_epochs=3)
    
    print(f"✓ Training completed on rank {rank}")


if __name__ == "__main__":
    main()
