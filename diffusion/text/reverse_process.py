import random
import math

class ReverseDiffusion:
    def __init__(self, model, noise_schedule):
        self.model = model
        self.noise_schedule = noise_schedule
    
    def sample_from_logits(self, logits, temperature=1.0):
        """Sample token from logits using temperature-scaled softmax"""
        # Apply temperature
        scaled_logits = [l / temperature for l in logits]
        
        # Softmax
        max_logit = max(scaled_logits)
        exp_logits = [math.exp(l - max_logit) for l in scaled_logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]
        
        # Sample
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                return i
        return len(probs) - 1
    
    def denoise_step(self, noisy_tokens, timestep, temperature=1.0):
        """Single denoising step"""
        logits = self.model.forward(noisy_tokens, timestep)
        
        # Sample new tokens for each position
        denoised_tokens = []
        for pos_logits in logits:
            token = self.sample_from_logits(pos_logits, temperature)
            denoised_tokens.append(token)
        
        return denoised_tokens
    
    def generate(self, seq_len, vocab_size, num_steps=None, temperature=1.0):
        """Generate text by iterative denoising from pure noise"""
        if num_steps is None:
            num_steps = self.noise_schedule.num_steps
        
        # Start with random tokens
        tokens = [random.randint(0, vocab_size - 1) for _ in range(seq_len)]
        
        # Denoise from T -> 0
        for t in reversed(range(num_steps)):
            tokens = self.denoise_step(tokens, t, temperature)
            
            # Add small noise for non-final steps (helps exploration)
            if t > 0:
                beta_t = self.noise_schedule.get_beta(t - 1)
                for i in range(len(tokens)):
                    if random.random() < beta_t * 0.5:  # Reduced noise
                        tokens[i] = random.randint(0, vocab_size - 1)
        
        return tokens
