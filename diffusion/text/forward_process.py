import random

class ForwardDiffusion:
    def __init__(self, noise_schedule, vocab_size):
        self.noise_schedule = noise_schedule
        self.vocab_size = vocab_size
    
    def add_noise(self, tokens, timestep):
        """Add noise to tokens at given timestep using mask-and-replace"""
        beta_t = self.noise_schedule.get_beta(timestep)
        noisy_tokens = []
        
        for token in tokens:
            if random.random() < beta_t:
                noisy_tokens.append(random.randint(0, self.vocab_size - 1))
            else:
                noisy_tokens.append(token)
        
        return noisy_tokens
    
    def get_noisy_sample(self, tokens, timestep):
        """Get noisy version of tokens at timestep t"""
        alpha_bar_t = self.noise_schedule.get_alpha_bar(timestep)
        noisy_tokens = []
        
        for token in tokens:
            # With probability (1 - alpha_bar_t), replace with random token
            if random.random() > alpha_bar_t:
                noisy_tokens.append(random.randint(0, self.vocab_size - 1))
            else:
                noisy_tokens.append(token)
        
        return noisy_tokens
