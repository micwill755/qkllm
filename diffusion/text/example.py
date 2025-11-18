import random
from noise_schedule import NoiseSchedule
from forward_process import ForwardDiffusion
from model import DiffusionTransformer
from reverse_process import ReverseDiffusion
from train import DiffusionTrainer

def main():
    # Hyperparameters
    vocab_size = 1000
    emb_dim = 128
    seq_len = 20
    num_steps = 100
    
    print("=== Text Diffusion Model Demo ===\n")
    
    # Initialize components
    noise_schedule = NoiseSchedule(num_steps=num_steps)
    forward_diffusion = ForwardDiffusion(noise_schedule, vocab_size)
    model = DiffusionTransformer(vocab_size, emb_dim)
    reverse_diffusion = ReverseDiffusion(model, noise_schedule)
    
    # Demo 1: Forward diffusion (adding noise)
    print("1. Forward Diffusion (Adding Noise)")
    clean_tokens = [random.randint(0, vocab_size - 1) for _ in range(seq_len)]
    print(f"Clean tokens: {clean_tokens[:10]}...")
    
    noisy_t50 = forward_diffusion.get_noisy_sample(clean_tokens, 50)
    print(f"Noisy (t=50): {noisy_t50[:10]}...")
    
    noisy_t90 = forward_diffusion.get_noisy_sample(clean_tokens, 90)
    print(f"Noisy (t=90): {noisy_t90[:10]}...\n")
    
    # Demo 2: Model prediction
    print("2. Model Prediction")
    logits = model.forward(noisy_t50, 50)
    print(f"Output shape: {len(logits)} positions x {len(logits[0])} vocab\n")
    
    # Demo 3: Generation
    print("3. Text Generation (from noise)")
    generated = reverse_diffusion.generate(seq_len, vocab_size, num_steps=20)
    print(f"Generated: {generated[:10]}...\n")
    
    # Demo 4: Training setup
    print("4. Training Setup")
    trainer = DiffusionTrainer(model, forward_diffusion, noise_schedule)
    
    # Create dummy dataset
    dataset = [[random.randint(0, vocab_size - 1) for _ in range(seq_len)] 
               for _ in range(100)]
    
    loss = trainer.train_step(dataset[0])
    print(f"Single step loss: {loss:.4f}")
    
    print("\n=== Demo Complete ===")
    print("\nKey Concepts:")
    print("- Forward: Gradually corrupt text with noise")
    print("- Model: Predicts original tokens from noisy input + timestep")
    print("- Reverse: Iteratively denoise to generate text")
    print("- Training: Learn to denoise at all timesteps")

if __name__ == "__main__":
    main()
