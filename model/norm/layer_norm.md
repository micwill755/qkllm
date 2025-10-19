## Step-by-Step Example

### Input
Token with embedding: `[1.0, 2.0, 3.0, 4.0]`

### Step 1: Calculate Mean

```
mean = (1.0 + 2.0 + 3.0 + 4.0) / 4
     = 10.0 / 4
     = 2.5
```

### Step 2: Calculate Variance

```
var = ((1.0 - 2.5)² + (2.0 - 2.5)² + (3.0 - 2.5)² + (4.0 - 2.5)²) / 4
    = ((-1.5)² + (-0.5)² + (0.5)² + (1.5)²) / 4
    = (2.25 + 0.25 + 0.25 + 2.25) / 4
    = 5.0 / 4
    = 1.25
```

### Step 3: Calculate Standard Deviation

```
std = sqrt(var + eps)
    = sqrt(1.25 + 1e-5)
    ≈ 1.118
```

### Step 4: Normalize (Z-score normalization)

For each dimension, apply: `(value - mean) / std`

```
normalized[0] = (1.0 - 2.5) / 1.118 = -1.5 / 1.118 ≈ -1.342
normalized[1] = (2.0 - 2.5) / 1.118 = -0.5 / 1.118 ≈ -0.447
normalized[2] = (3.0 - 2.5) / 1.118 =  0.5 / 1.118 ≈  0.447
normalized[3] = (4.0 - 2.5) / 1.118 =  1.5 / 1.118 ≈  1.342
```

**Result**: `[-1.342, -0.447, 0.447, 1.342]`

### Step 5: Apply Learnable Scale and Shift

```
output[i] = scale[i] * normalized[i] + shift[i]
```

With default parameters (`scale = [1, 1, 1, 1]` and `shift = [0, 0, 0, 0]`):

```
output = [-1.342, -0.447, 0.447, 1.342]
```

---

## Verification

Let's verify the output has mean ≈ 0 and std ≈ 1:

```
mean = (-1.342 + -0.447 + 0.447 + 1.342) / 4 = 0.0 ✓

variance = ((-1.342)² + (-0.447)² + (0.447)² + (1.342)²) / 4
         = (1.801 + 0.200 + 0.200 + 1.801) / 4
         = 4.002 / 4
         ≈ 1.0

std = sqrt(1.0) = 1.0 ✓
```

---

## Multiple Tokens Example

For a sequence with 2 tokens:

```
Input: [1.0, 2.0, 3.0, 4.0,    # Token 1
        5.0, 6.0, 7.0, 8.0]    # Token 2
```

**Token 1** is normalized independently:
- mean = 2.5, std = 1.118
- output: `[-1.342, -0.447, 0.447, 1.342]`

**Token 2** is normalized independently:
- mean = 6.5, std = 1.118
- output: `[-1.342, -0.447, 0.447, 1.342]`

Each token gets its own normalization statistics!

---

## Why LayerNorm?

1. **Stabilizes training**: Prevents values from exploding or vanishing
2. **Scale invariant**: Model is less sensitive to input scale
3. **Faster convergence**: Normalized inputs help gradient flow
4. **Per-token**: Works well for variable-length sequences in transformers

---

## LayerNorm vs RMSNorm

| Feature | LayerNorm | RMSNorm |
|---------|-----------|---------|
| Centers data (subtract mean) | ✓ | ✗ |
| Scales by std/RMS | ✓ | ✓ |
| Learnable shift | ✓ | ✗ |
| Computation | More | Less |
| Used in | BERT, GPT-2 | LLaMA, GPT-3 |

RMSNorm is simpler and faster while achieving similar performance.
