# Text Diffusion Model

Minimal implementation of a discrete text diffusion model.

## Architecture

```
noise_schedule.py      - Defines β_t schedule for noise levels
forward_process.py     - Adds noise to clean text
model.py              - Transformer with time conditioning
reverse_process.py    - Denoising and generation
train.py              - Training loop and loss computation
example.py            - Demo script
```

## How It Works

1. **Forward Process**: Gradually replace tokens with random ones
   - At timestep t, each token has probability β_t of being replaced
   - By timestep T, text is mostly random noise

2. **Model**: Transformer that predicts original tokens
   - Input: noisy tokens + timestep embedding
   - Output: logits over vocabulary for each position

3. **Reverse Process**: Iterative denoising
   - Start with random tokens
   - At each step, predict less noisy version
   - After T steps, generate coherent text

4. **Training**: Learn to denoise at all timesteps
   - Sample random timestep t
   - Add noise to clean text
   - Train model to predict original tokens

## Usage

```python
from noise_schedule import NoiseSchedule
from forward_process import ForwardDiffusion
from model import DiffusionTransformer
from reverse_process import ReverseDiffusion

# Setup
noise_schedule = NoiseSchedule(num_steps=1000)
forward_diffusion = ForwardDiffusion(noise_schedule, vocab_size=1000)
model = DiffusionTransformer(vocab_size=1000, emb_dim=128)
reverse_diffusion = ReverseDiffusion(model, noise_schedule)

# Generate
tokens = reverse_diffusion.generate(seq_len=20, vocab_size=1000)
```

## Run Demo

```bash
cd llm/llm/diffusion/text
python example.py
```

## Key Differences from Image Diffusion

- **Discrete tokens** instead of continuous pixels
- **Categorical noise** (random token replacement) instead of Gaussian
- **Mask-and-replace** strategy for adding noise
- Slower convergence due to discrete nature

## Extensions

- Add proper transformer blocks (attention, FFN)
- Implement gradient-based training
- Use better noise schedules (cosine, learned)
- Add conditioning (prompts, style control)
- Implement continuous diffusion in embedding space
