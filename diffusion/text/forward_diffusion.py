from noise import Noise

import tiktoken
import random

class ForwardDiffusion:
    def __init__(self, noise_type="uniform", num_steps=100):
        self.num_steps = num_steps
        self.encoder = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.encoder.n_vocab
        self.noise_type = noise_type
        self.mask_token = self.vocab_size
        
        # Noise schedule
        self.betas = self._create_schedule()
        self.alpha_bars = self._compute_alpha_bars()
    
    def _create_schedule(self):
        """Linear beta schedule"""
        beta_start, beta_end = 0.0001, 0.02
        # β_t = start + (end - start) * t / T - β increases linearly
        return [beta_start + (beta_end - beta_start) * t / self.num_steps 
                for t in range(self.num_steps)]

    def _compute_alpha_bars(self):
        """Cumulative product of (1 - beta)"""
        alpha_bars = []
        alpha_bar = 1.0
        for beta in self.betas:
            alpha_bar *= (1 - beta)
            alpha_bars.append(alpha_bar)
        return alpha_bars
    
    def _add_noise_uniform(self, tokens, alpha_bar):  # ← Uses alpha_bar, not beta
        noisy_tokens = []
        for token in tokens:
            if random.random() > alpha_bar:  # ← Cumulative noise
                noisy_tokens.append(random.randint(0, self.vocab_size - 1))
            else:
                noisy_tokens.append(token)
        return noisy_tokens

    def _add_noise_absorbing(self, tokens, alpha_bar):
        """Add absorbing noise - replace with MASK token"""
        noisy_tokens = []
        for token in tokens:
            if random.random() > alpha_bar:  # Token gets masked
                noisy_tokens.append(self.mask_token)
            else:  # Token survives
                noisy_tokens.append(token)
        return noisy_tokens

    def get_noisy_at_t(self, text, t):
        """Get noisy version at timestep t"""
        tokens = self.encoder.encode(text)
        alpha_bar = self.alpha_bars[t]
        
        # Choose noise type
        if self.noise_type == "uniform":
            noisy_tokens = self._add_noise_uniform(tokens, alpha_bar)
        elif self.noise_type == "absorbing":
            noisy_tokens = self._add_noise_absorbing(tokens, alpha_bar)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        return self._decode(noisy_tokens)
    
    def _decode(self, tokens):
        """Decode tokens, handling mask tokens for absorbing noise"""
        if self.noise_type == "absorbing":
            decoded_parts = []
            for t in tokens:
                if t == self.mask_token:
                    decoded_parts.append("[MASK]")
                else:
                    decoded_parts.append(self.encoder.decode([t]))
            return "".join(decoded_parts)
        else:
            return self.encoder.decode(tokens)
    
    def visualize(self, text, timesteps=None):
        """Show text at different noise levels"""
        if timesteps is None:
            timesteps = [0, 25, 50, 75, 99]
        
        print(f"Original: {text}\n")
        for t in timesteps[1:]:
            noisy = self.get_noisy_at_t(text, t)
            print(f"t={t:3d} (α_bar={self.alpha_bars[t]:.3f}): {noisy}")

if __name__ == '__main__':
    # Test uniform noise
    forward_uniform = ForwardDiffusion(noise_type="uniform", num_steps=100)
    forward_uniform.visualize("Diffusion models are the future")

    print("\n" + "="*50 + "\n")

    # Test absorbing noise
    forward_absorbing = ForwardDiffusion(noise_type="absorbing", num_steps=100)
    forward_absorbing.visualize("Diffusion models are the future")
        