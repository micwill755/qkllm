class NoiseSchedule:
    def __init__(self, num_steps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_steps = num_steps
        self.betas = [beta_start + (beta_end - beta_start) * t / num_steps 
                      for t in range(num_steps)]
        
        # Precompute cumulative products for efficiency
        self.alphas = [1.0 - beta for beta in self.betas]
        self.alpha_bars = []
        alpha_bar = 1.0
        for alpha in self.alphas:
            alpha_bar *= alpha
            self.alpha_bars.append(alpha_bar)
    
    def get_beta(self, t):
        return self.betas[t]
    
    def get_alpha_bar(self, t):
        return self.alpha_bars[t]
