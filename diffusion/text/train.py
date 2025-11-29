# train.py
import mx
import mx.optimizers as optim
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import numpy as np

from model import DiffusionModel

# Hyperparameters
vocab_size = 10000
emb_dim = 256
n_heads = 8
n_layers = 6
batch_size = 16
seq_len = 128
num_epochs = 3
learning_rate = 3e-4
T = 1000  # diffusion timesteps

# Load text file
print("Loading dataset...")
with open("input.txt", "r") as f:
    text = f.read()

# Split into chunks
def text_chunks(text, chunk_size=1000):
    for i in range(0, len(text), chunk_size):
        yield text[i:i+chunk_size]

# Train tokenizer on sample
print("Training tokenizer...")
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<unk>"])

def dataset_iterator():
    while True:
        for chunk in text_chunks(text, chunk_size=500):
            yield {"text": chunk}

tokenizer.train_from_iterator(text_chunks(text), trainer=trainer)

# Diffusion noise schedule (linear)
betas = np.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alphas_cumprod = np.cumprod(alphas)

def add_noise(x, t):
    """Add noise to embeddings at timestep t"""
    alpha_t = alphas_cumprod[t]
    noise = mx.randn(x.shape)
    return np.sqrt(alpha_t) * x + np.sqrt(1 - alpha_t) * noise, noise

def get_batch(dataset_iter, batch_size, seq_len):
    """Get a batch of tokenized sequences"""
    texts = []
    for _ in range(batch_size):
        try:
            texts.append(next(dataset_iter)["text"])
        except StopIteration:
            break
    
    # Tokenize and pad
    encodings = [tokenizer.encode(text).ids[:seq_len] for text in texts]
    tokens = np.zeros((len(encodings), seq_len), dtype=np.int32)
    
    for i, enc in enumerate(encodings):
        tokens[i, :len(enc)] = enc
    
    return mx.array(tokens)

# Initialize model and optimizer
model = DiffusionModel(vocab_size, emb_dim, n_heads, n_layers)
optimizer = optim.Adam(learning_rate=learning_rate)

# Training loop
print("Starting training...")
dataset_iter = dataset_iterator()

for epoch in range(num_epochs):
    for step in range(1000):  # 1000 steps per epoch
        # Get batch
        tokens = get_batch(dataset_iter, batch_size, seq_len)
        
        # Sample random timesteps
        t = np.random.randint(0, T, size=(batch_size,))
        timestep = mx.array(t).reshape([batch_size, 1])
        
        # Get token embeddings and add noise
        token_emb = model.token_embedding(tokens)
        noisy_emb, noise = add_noise(token_emb, t[0])  # Simplified: same t for batch
        
        # Predict noise
        pred = model.forward(tokens, timestep)
        
        # MSE loss between predicted and actual noise
        loss = mx.mean((pred - noise) ** 2)
        
        # Backward pass
        loss.backward()
        optimizer.update(model)

        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
    
    # Save checkpoint
    #mx.save(f"checkpoint_epoch_{epoch}.npz", model.parameters())

print("Training complete!")
