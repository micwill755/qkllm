## Step-by-Step Example

### Input
Token with embedding: `[1.0, 2.0, 3.0, 4.0]`

### Step 1: Calculate Sum of Squares

```
sum_squares = 1.0² + 2.0² + 3.0² + 4.0²
            = 1.0 + 4.0 + 9.0 + 16.0
            = 30.0
```

### Step 2: Calculate RMS (Root Mean Square)

```
rms = sqrt(sum_squares / emb_dim + eps)
    = sqrt(30.0 / 4 + 1e-5)
    = sqrt(7.5 + 1e-5)
    ≈ 2.739
```

### Step 3: Normalize by RMS

For each dimension, apply: `value / rms`

```
normalized[0] = 1.0 / 2.739 ≈ 0.365
normalized[1] = 2.0 / 2.739 ≈ 0.730
normalized[2] = 3.0 / 2.739 ≈ 1.095
normalized[3] = 4.0 / 2.739 ≈ 1.461
```

### Step 4: Apply Learnable Weight

```
output[i] = normalized[i] * weight[i]
```

With default parameters (`weight = [1, 1, 1, 1]`):

```
output = [0.365, 0.730, 1.095, 1.461]
```

---

## Verification

The output has RMS ≈ 1:

```
rms = sqrt((0.365² + 0.730² + 1.095² + 1.461²) / 4)
    = sqrt((0.133 + 0.533 + 1.199 + 2.135) / 4)
    = sqrt(4.0 / 4)
    = sqrt(1.0)
    = 1.0 ✓

mean = (0.365 + 0.730 + 1.095 + 1.461) / 4
     = 3.651 / 4
     ≈ 0.913  (NOT zero!)
```

---

## Key Differences from LayerNorm

1. **No mean subtraction**: RMSNorm doesn't center the data
2. **Only scales**: Divides by RMS instead of standard deviation
3. **Simpler**: Fewer operations (no mean calculation, no shift parameter)
4. **Non-zero mean**: Output mean is NOT zero (unlike LayerNorm)
5. **Faster**: Less computation makes it more efficient

---

## Multiple Tokens Example

For a sequence with 2 tokens:

```
Input: [1.0, 2.0, 3.0, 4.0,    # Token 1
        5.0, 6.0, 7.0, 8.0]    # Token 2
```

**Token 1**:
- RMS = 2.739
- output: `[0.365, 0.730, 1.095, 1.461]`

**Token 2**:
- RMS = 7.071
- output: `[0.707, 0.849, 0.990, 1.131]`

Each token is scaled independently by its own RMS value!
