import math
import mx

class TimeEmbedding:
    def __init__(self, emb_dim):
        self.emb_dim = emb_dim
    
    def forward(self, timestep):
        half_dim = self.emb_dim // 2
        batch_size = timestep.shape[0]
        result = mx.zeros([batch_size, self.emb_dim])
        
        for b in range(batch_size):
            # Extract scalar using .item() method
            t_slice = timestep[b] if len(timestep.shape) == 1 else timestep[b][0]
            t_val = t_slice._c_tensor.item()  # Call C method
            
            for i in range(half_dim):
                freq = 1.0 / (10000 ** (2 * i / self.emb_dim))
                angle = t_val * freq
                result[b][2*i] = math.sin(angle)
                if 2*i + 1 < self.emb_dim:
                    result[b][2*i + 1] = math.cos(angle)
        
        return result

    
    def __call__(self, timestep):
        return self.forward(timestep)
