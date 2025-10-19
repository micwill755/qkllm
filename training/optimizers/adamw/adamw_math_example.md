# AdamW Math Example: Step-by-Step Walkthrough

Let's trace through AdamW updates for a single parameter over 4 training steps with real numbers.

## Setup

**Parameter**: θ = 0.5 (initial weight)  
**Learning rate**: lr = 0.1  
**Beta1** (momentum): β₁ = 0.9  
**Beta2** (velocity): β₂ = 0.999  
**Weight decay**: λ = 0.01  
**Epsilon**: ε = 1e-8  

**Initial state**:
- m₀ = 0 (momentum)
- v₀ = 0 (velocity)

## Step 1: First Gradient

**Gradient**: g₁ = 0.2

### 1. Update momentum (exponential moving average of gradients)
m₁ = β₁ × m₀ + (1 - β₁) × g₁  
m₁ = 0.9 × 0 + 0.1 × 0.2 = **0.02**

### 2. Update velocity (exponential moving average of squared gradients)
v₁ = β₂ × v₀ + (1 - β₂) × g₁²  
v₁ = 0.999 × 0 + 0.001 × (0.2)² = 0.001 × 0.04 = **0.00004**

### 3. Bias correction (because early estimates are biased toward zero)
m̂₁ = m₁ / (1 - β₁¹) = 0.02 / (1 - 0.9) = 0.02 / 0.1 = **0.2**  
v̂₁ = v₁ / (1 - β₂¹) = 0.00004 / (1 - 0.999) = 0.00004 / 0.001 = **0.04**

### 4. Apply weight decay
θ₁ = θ₀ - λ × θ₀ = 0.5 - 0.01 × 0.5 = **0.495**

### 5. Parameter update
θ₁ = θ₁ - lr × m̂₁ / (√v̂₁ + ε)  
θ₁ = 0.495 - 0.1 × 0.2 / (√0.04 + 1e-8)  
θ₁ = 0.495 - 0.1 × 0.2 / 0.2 = 0.495 - 0.1 = **0.395**

**Result**: θ₁ = 0.395, m₁ = 0.02, v₁ = 0.00004

---

## Step 2: Second Gradient

**Gradient**: g₂ = 0.15

### 1. Update momentum
m₂ = β₁ × m₁ + (1 - β₁) × g₂  
m₂ = 0.9 × 0.02 + 0.1 × 0.15 = 0.018 + 0.015 = **0.033**

### 2. Update velocity
v₂ = β₂ × v₁ + (1 - β₂) × g₂²  
v₂ = 0.999 × 0.00004 + 0.001 × (0.15)² = 0.00003996 + 0.0000225 = **0.00006246**

### 3. Bias correction
m̂₂ = m₂ / (1 - β₁²) = 0.033 / (1 - 0.81) = 0.033 / 0.19 = **0.174**  
v̂₂ = v₂ / (1 - β₂²) = 0.00006246 / (1 - 0.998001) = 0.00006246 / 0.001999 = **0.0312**

### 4. Apply weight decay
θ₂ = θ₁ - λ × θ₁ = 0.395 - 0.01 × 0.395 = **0.39105**

### 5. Parameter update
θ₂ = θ₂ - lr × m̂₂ / (√v̂₂ + ε)  
θ₂ = 0.39105 - 0.1 × 0.174 / √0.0312  
θ₂ = 0.39105 - 0.1 × 0.174 / 0.177 = 0.39105 - 0.098 = **0.293**

**Result**: θ₂ = 0.293, m₂ = 0.033, v₂ = 0.00006246

---

## Step 3: Third Gradient

**Gradient**: g₃ = 0.25 (larger gradient)

### 1. Update momentum
m₃ = 0.9 × 0.033 + 0.1 × 0.25 = 0.0297 + 0.025 = **0.0547**

### 2. Update velocity
v₃ = 0.999 × 0.00006246 + 0.001 × (0.25)² = 0.0000624 + 0.0000625 = **0.0001249**

### 3. Bias correction
m̂₃ = 0.0547 / (1 - 0.729) = 0.0547 / 0.271 = **0.202**  
v̂₃ = 0.0001249 / (1 - 0.997) = 0.0001249 / 0.003 = **0.0416**

### 4. Apply weight decay
θ₃ = 0.293 - 0.01 × 0.293 = **0.29007**

### 5. Parameter update
θ₃ = 0.29007 - 0.1 × 0.202 / √0.0416  
θ₃ = 0.29007 - 0.1 × 0.202 / 0.204 = 0.29007 - 0.099 = **0.191**

**Result**: θ₃ = 0.191, m₃ = 0.0547, v₃ = 0.0001249

---

## Step 4: Fourth Gradient

**Gradient**: g₄ = 0.05 (smaller gradient)

### 1. Update momentum
m₄ = 0.9 × 0.0547 + 0.1 × 0.05 = 0.04923 + 0.005 = **0.05423**

### 2. Update velocity
v₄ = 0.999 × 0.0001249 + 0.001 × (0.05)² = 0.0001247 + 0.0000025 = **0.0001272**

### 3. Bias correction
m̂₄ = 0.05423 / (1 - 0.6561) = 0.05423 / 0.3439 = **0.158**  
v̂₄ = 0.0001272 / (1 - 0.996) = 0.0001272 / 0.004 = **0.0318**

### 4. Apply weight decay
θ₄ = 0.191 - 0.01 × 0.191 = **0.18909**

### 5. Parameter update
θ₄ = 0.18909 - 0.1 × 0.158 / √0.0318  
θ₄ = 0.18909 - 0.1 × 0.158 / 0.178 = 0.18909 - 0.089 = **0.100**

**Final Result**: θ₄ = 0.100

---

## Key Observations

### 1. Momentum Builds Up
- Step 1: m̂₁ = 0.2 (just the current gradient)
- Step 2: m̂₂ = 0.174 (smoothed with previous)
- Step 3: m̂₃ = 0.202 (builds momentum from consistent direction)
- Step 4: m̂₄ = 0.158 (dampens when gradient changes direction)

### 2. Adaptive Learning Rate
- **Large gradient** (Step 3, g=0.25): Effective LR = 0.1 × 0.202/0.204 = 0.099
- **Small gradient** (Step 4, g=0.05): Effective LR = 0.1 × 0.158/0.178 = 0.089

The parameter with larger gradients gets similar-sized updates to the one with smaller gradients - this is the "adaptive" part!

### 3. Bias Correction Impact
Early steps have strong bias correction:
- Step 1: Raw momentum = 0.02, Corrected = 0.2 (10x boost!)
- Step 4: Raw momentum = 0.054, Corrected = 0.158 (3x boost)

### 4. Weight Decay Effect
Each step applies: θ = θ × (1 - λ) = θ × 0.99
This gradually shrinks parameters toward zero for regularization.

## Summary

AdamW took our parameter from **0.5 → 0.100** over 4 steps by:
- **Smoothing gradients** with momentum
- **Adapting learning rate** based on gradient history  
- **Correcting early bias** in the estimates
- **Applying regularization** with weight decay

The math ensures stable, adaptive updates that work well across different parameter scales!