# SEDD Transformer Architecture

## Overview
Score Entropy Discrete Diffusion (SEDD) is a diffusion transformer designed for discrete text generation. It learns to denoise text by predicting clean tokens from noisy ones, conditioned on the noise level (timestep).

## Architecture Diagram

```
Input: Token Indices + Timestep σ
         │                │
         ▼                ▼
   ┌─────────────┐   ┌─────────────┐
   │ Vocab Embed │   │ Timestep    │
   │ (tokens →   │   │ Embedder    │
   │  vectors)   │   │ (σ → cond)  │
   └─────────────┘   └─────────────┘
         │                │
         ▼                │
   ┌─────────────┐        │
   │ Rotary Pos  │        │
   │ Embedding   │        │
   └─────────────┘        │
         │                │
         ▼                ▼
   ┌─────────────────────────────────┐
   │        DDiT Block 1             │
   │  ┌─────────────────────────┐    │
   │  │ AdaLN Modulation        │◄───┤ (condition c)
   │  │ (6 params from cond)    │    │
   │  └─────────────────────────┘    │
   │           │                     │
   │           ▼                     │
   │  ┌─────────────────────────┐    │
   │  │ LayerNorm + Modulate    │    │
   │  └─────────────────────────┘    │
   │           │                     │
   │           ▼                     │
   │  ┌─────────────────────────┐    │
   │  │ Multi-Head Attention    │    │
   │  │ • QKV projection        │    │
   │  │ • Rotary pos encoding   │    │
   │  │ • Flash attention       │    │
   │  └─────────────────────────┘    │
   │           │                     │
   │           ▼                     │
   │  ┌─────────────────────────┐    │
   │  │ Residual + Gate         │    │
   │  └─────────────────────────┘    │
   │           │                     │
   │           ▼                     │
   │  ┌─────────────────────────┐    │
   │  │ LayerNorm + Modulate    │    │
   │  └─────────────────────────┘    │
   │           │                     │
   │           ▼                     │
   │  ┌─────────────────────────┐    │
   │  │ MLP (4x expansion)      │    │
   │  │ • Linear → GELU → Linear│    │
   │  └─────────────────────────┘    │
   │           │                     │
   │           ▼                     │
   │  ┌─────────────────────────┐    │
   │  │ Residual + Gate         │    │
   │  └─────────────────────────┘    │
   └─────────────────────────────────┘
         │
         ▼
   ┌─────────────┐
   │    ...      │ (repeat N blocks)
   └─────────────┘
         │
         ▼
   ┌─────────────────────────────────┐
   │     Final Layer                 │
   │  ┌─────────────────────────┐    │
   │  │ AdaLN Modulation        │◄───┤ (condition c)
   │  │ (2 params from cond)    │    │
   │  └─────────────────────────┘    │
   │           │                     │
   │           ▼                     │
   │  ┌─────────────────────────┐    │
   │  │ LayerNorm + Modulate    │    │
   │  └─────────────────────────┘    │
   │           │                     │
   │           ▼                     │
   │  ┌─────────────────────────┐    │
   │  │ Linear → Vocab Logits   │    │
   │  └─────────────────────────┘    │
   └─────────────────────────────────┘
         │
         ▼
   Output: Denoised Logits
```

## Key Components

### 1. Timestep Embedder
- **Purpose**: Converts scalar timestep σ into a conditioning vector
- **Method**: Sinusoidal embeddings → MLP (2 layers with SiLU activation)
- **Output**: Conditioning vector that modulates all transformer blocks

### 2. Vocab Embedding
- **Purpose**: Converts discrete token indices to continuous vectors
- **Initialization**: Kaiming uniform initialization
- **Shape**: (vocab_size, hidden_dim)

### 3. Rotary Position Embedding
- **Purpose**: Encodes positional information into attention
- **Advantage**: Better extrapolation to longer sequences than absolute positions
- **Applied**: Directly to Q, K vectors before attention

### 4. DDiT Block (Diffusion Transformer Block)
Each block contains:

#### a. Adaptive Layer Norm (AdaLN)
- **Purpose**: Conditions normalization on timestep
- **Parameters**: 6 values per block (shift, scale, gate for attention & MLP)
- **Formula**: `modulate(norm(x), shift, scale) = norm(x) * (1 + scale) + shift`

#### b. Multi-Head Attention
- **QKV Projection**: Single linear layer → split into Q, K, V
- **Rotary Encoding**: Applied to Q, K for positional awareness
- **Flash Attention**: Memory-efficient attention computation
- **Non-causal**: Bidirectional attention (can attend to all positions)

#### c. Gated Residuals
- **Purpose**: Learnable control of information flow
- **Formula**: `x_out = gate * layer_output + x_in`
- **Benefit**: Helps with training stability

#### d. MLP (Feed-Forward Network)
- **Expansion**: 4x hidden dimension
- **Activation**: GELU (approximate tanh version)
- **Structure**: Linear → GELU → Linear

### 5. Final Layer
- **AdaLN Modulation**: 2 parameters (shift, scale)
- **Output Projection**: Linear layer to vocabulary size
- **Initialization**: Zero initialization for stability

## Key Features

### 1. Adaptive Layer Norm (AdaLN)
Conditions each layer on the timestep via shift/scale/gate parameters. This allows the model to adapt its behavior based on the noise level.

### 2. Rotary Position Encoding
Better positional awareness than standard learned or sinusoidal embeddings. Enables better length generalization.

### 3. Flash Attention
Optimized attention computation that reduces memory usage and increases speed without sacrificing accuracy.

### 4. Residual Gating
Learnable gates control how much information flows through residual connections, improving training dynamics.

### 5. Timestep Conditioning
Sinusoidal embeddings → MLP → conditioning vector that modulates every layer through AdaLN.

### 6. Non-Causal Attention
Unlike autoregressive models, SEDD uses bidirectional attention since it's denoising entire sequences at once.

## Model Flow

1. **Input Processing**
   - Token indices → Vocab embeddings
   - Timestep σ → Conditioning vector c

2. **Transformer Processing**
   - Apply rotary position embeddings
   - Pass through N DDiT blocks
   - Each block conditioned on timestep via AdaLN

3. **Output Generation**
   - Final layer with AdaLN conditioning
   - Project to vocabulary logits
   - Predict denoised token distribution

## Training Objective

The model learns to predict clean tokens from noisy ones:
- **Input**: Noisy tokens at timestep t
- **Condition**: Timestep embedding
- **Target**: Original clean tokens
- **Loss**: Cross-entropy between predicted and true token distributions

## Advantages

1. **Flexible Generation**: Can generate text in any order (not just left-to-right)
2. **Iterative Refinement**: Multiple denoising steps improve quality
3. **Controllable**: Timestep conditioning allows control over generation process
4. **Parallel**: Non-autoregressive generation can be faster
5. **High Quality**: Competitive with autoregressive models on many tasks
