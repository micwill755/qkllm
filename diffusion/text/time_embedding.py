'''

What is Time Embedding?

Purpose: Convert a timestep number (like t=50) into a vector that the model can understand.

Why we need it: The model needs to know "how noisy" the input is to make appropriate predictions.

'''

import math

class TimeEmbedding:
    def __init__(self, emb_dim):
        self.emb_dim = emb_dim
    
    def forward(self, timestep):
        """
        Convert timestep to sinusoidal embedding
        
        Args:
            timestep: int (0 to num_steps-1)
        
        Returns:
            List of floats, length emb_dim
        """
        half_dim = self.emb_dim // 2
        embedding = []
        
        '''
        What the Model Learns:
        - The model's weights learn to interpret the time embedding:
        - High freq dimensions changing fast → "I'm at a specific timestep, use precise denoising"
        - Low freq dimensions barely changing → "I'm in early/middle/late stage globally"
        Combination → "I'm at t=50, use medium-confidence predictions"

        The time embedding itself never changes - only the model's interpretation of it does through training.
        
        '''
        for i in range(half_dim):
            # frequency represents how fast the encoding changes as position/timestep changes
            freq = 1.0 / (10000 ** (2 * i / self.emb_dim))
            
            # Compute angle
            angle = timestep * freq
            
            # Add sin and cos
            embedding.append(math.sin(angle))
            embedding.append(math.cos(angle))
        
        # Handle odd embedding dimensions
        return embedding[:self.emb_dim]
    
    def __call__(self, timestep):
        return self.forward(timestep)

class TimestepEmbedder:
    def __init__(self):
        pass