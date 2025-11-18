import random
import math

class DiffusionTrainer:
    def __init__(self, model, forward_diffusion, noise_schedule, learning_rate=0.001):
        self.model = model
        self.forward_diffusion = forward_diffusion
        self.noise_schedule = noise_schedule
        self.learning_rate = learning_rate
    
    def cross_entropy_loss(self, logits, targets):
        """Compute cross-entropy loss"""
        total_loss = 0.0
        
        for pos_logits, target in zip(logits, targets):
            # Softmax
            max_logit = max(pos_logits)
            exp_logits = [math.exp(l - max_logit) for l in pos_logits]
            sum_exp = sum(exp_logits)
            log_prob = pos_logits[target] - max_logit - math.log(sum_exp)
            total_loss -= log_prob
        
        return total_loss / len(targets)
    
    def train_step(self, clean_tokens):
        """Single training step"""
        # Sample random timestep
        t = random.randint(0, self.noise_schedule.num_steps - 1)
        
        # Add noise
        noisy_tokens = self.forward_diffusion.get_noisy_sample(clean_tokens, t)
        
        # Predict original tokens
        predicted_logits = self.model.forward(noisy_tokens, t)
        
        # Compute loss
        loss = self.cross_entropy_loss(predicted_logits, clean_tokens)
        
        return loss
    
    def train(self, dataset, num_epochs=10, batch_size=32):
        """Training loop"""
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            # Shuffle dataset
            random.shuffle(dataset)
            
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                
                batch_loss = 0.0
                for tokens in batch:
                    loss = self.train_step(tokens)
                    batch_loss += loss
                
                avg_batch_loss = batch_loss / len(batch)
                total_loss += avg_batch_loss
                num_batches += 1
                
                # In practice, update model parameters here
                # For minimal implementation, we skip backprop
            
            avg_epoch_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        return self.model
