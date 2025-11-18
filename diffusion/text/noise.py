import tiktoken
import random
import math

# 1. simple noise - mask/replace

'''

Input: "Diffusion models are the future"
       ↓
[Tokenize]
       ↓
[42, 156, 8, 2, 891]
       ↓
[For each token]
       ↓
    Is random() < beta_t?
    ├─ YES → Replace with random token ID
    └─ NO  → Keep original token ID
       ↓
[7234, 156, 445, 2, 891]
       ↓
[Detokenize]
       ↓
"quantum models jumping the future"

'''

class Noise:
    def __init__(self, encoder, vocab_size):
        self.encoder = encoder
        self.vocab_size = vocab_size
        
    def add_noise_uniform(self, tokens, beta):
        """Add uniform noise with probability beta"""
        for i, t in enumerate(tokens):
            if random.random() < beta:
                random_token = random.randint(0, self.vocab_size - 1)
                tokens[i] = random_token

    # 2. absorbing noise - instead of replacing tokens with random tokens
    # we replace them with a special [MASK] token.

    def add_noise_absorbing(self, tokens, mask_token, beta):
        """Add absorbing noise - replace with MASK token"""
        for i, t in enumerate(tokens):
            if random.random() < beta:
                tokens[i] = mask_token

    def decode_with_mask(self, tokens):
        """Decode tokens, replacing mask_token with [MASK] string"""
        # Replace mask tokens with a valid token temporarily
        mask_token = vocab_size
        decoded_parts = []
        
        for t in tokens:
            if t == mask_token:
                decoded_parts.append("[MASK]")
            else:
                decoded_parts.append(self.encoder.decode([t]))
        
        return "".join(decoded_parts)

if __name__ == '__main__':
    encoder = tiktoken.get_encoding("gpt2")
    vocab_size = encoder.n_vocab
    mask_token = vocab_size

    noise = Noise(encoder, vocab_size)
    
    s = 'Diffusion models are the future'
    
    print(f"Original text: {s}")
    print(f"Original tokens: {encoder.encode(s)}")
    print(f"Vocab size: {vocab_size}\n")
    
    # β = 1 - e^(-σ)
    # β is the probability of replacement (discrete-time)
    beta_t = 0.3
    # inversely σ = -ln(1 - β)
    # σ is the rate of replacement (continuous-time)
    sigma = -math.log(1 - beta_t)
    move_chance = 1 - math.exp(-sigma)
    
    print(f"Beta: {beta_t:.3f}")
    print(f"Sigma: {sigma:.3f}")
    print(f"Move chance from sigma: {move_chance:.3f}\n")
    
    # Test 1: Uniform noise with beta
    print("=== UNIFORM NOISE (using beta) ===")
    tokens = encoder.encode(s)
    noise.add_noise_uniform(tokens, beta_t)
    print(f"Tokens: {tokens}")
    print(f"Text: {encoder.decode(tokens)}\n")
    
    # Test 2: Uniform noise with sigma
    print("=== UNIFORM NOISE (using sigma) ===")
    tokens = encoder.encode(s)
    noise.add_noise_uniform(tokens, move_chance)
    print(f"Tokens: {tokens}")
    print(f"Text: {encoder.decode(tokens)}\n")
    
    # Test 3: Absorbing noise
    print("=== ABSORBING NOISE ===")
    tokens = encoder.encode(s)
    noise.add_noise_absorbing(tokens, mask_token, move_chance)
    print(f"Tokens: {tokens}")
    print(f"Text: {noise.decode_with_mask(tokens)}")