# Memory Layout in C: Why We Use 1D Arrays

## The Problem with 2D Arrays in C

In Python/NumPy, we can easily create multidimensional arrays:
```python
mask = np.ones((1024, 1024))  # 2D array
```

In C, we have two approaches for 2D arrays, but only one is efficient.

## Approach 1: Fragmented Memory (SLOW)

```c
// Creates an array of pointers, then allocates each row separately
float **mask = malloc(context_length * sizeof(float*));
for(int i = 0; i < context_length; i++) {
    mask[i] = malloc(context_length * sizeof(float));
}

// Access: mask[i][j]
```

### Why This Is Slow

**Memory Layout:**
```
Row 0: [0.1][0.2][0.3][0.4] ← somewhere in memory
Row 1: [0.5][0.6][0.7][0.8] ← different location  
Row 2: [0.9][1.0][1.1][1.2] ← yet another location
```

**Problems:**
1. **Cache Misses**: Each row might be in different memory pages
2. **Pointer Chasing**: CPU must follow pointers to find each row
3. **Memory Fragmentation**: Rows scattered throughout RAM
4. **Extra Allocations**: N+1 malloc calls instead of 1

## Approach 2: Contiguous Memory (FAST)

```c
// Single allocation for entire matrix
float *mask = malloc(context_length * context_length * sizeof(float));

// Access: mask[i * context_length + j]
```

### Why This Is Fast

**Memory Layout:**
```
[0.1][0.2][0.3][0.4][0.5][0.6][0.7][0.8][0.9][1.0][1.1][1.2]
 ←---- Row 0 ----→ ←---- Row 1 ----→ ←---- Row 2 ----→
```

**Benefits:**
1. **Cache Friendly**: Sequential memory access
2. **No Pointer Chasing**: Direct calculation of address
3. **Single Allocation**: One malloc call
4. **Vectorization**: CPU can optimize sequential operations

## Index Mapping Formula

**2D to 1D conversion:**
```c
// Instead of: array[row][col]
// Use: array[row * width + col]

mask[i * context_length + j]  // equivalent to mask[i][j]
```

**3D to 1D conversion:**
```c
// array[batch][seq][emb] becomes:
array[(batch * seq_len + seq) * emb_dim + emb]
```

## Performance Impact

**Benchmark Example (1024x1024 matrix):**
- Fragmented approach: ~150ms for matrix multiplication
- Contiguous approach: ~45ms for matrix multiplication
- **3x speedup** just from memory layout!

## Real-World Applications

This pattern is used in:
- **BLAS libraries** (optimized linear algebra)
- **GPU computing** (CUDA, OpenCL)
- **Neural network frameworks** (PyTorch C++ backend)
- **Game engines** (matrix operations)

## Memory Access Patterns

**Good (Sequential):**
```c
for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
        process(array[i * cols + j]);  // Cache-friendly
    }
}
```

**Bad (Random):**
```c
for(int j = 0; j < cols; j++) {
    for(int i = 0; i < rows; i++) {
        process(array[i * cols + j]);  // Cache-unfriendly
    }
}
```

## Key Takeaway

In high-performance C code, **memory layout is performance**. The 1D array approach with manual indexing is not just a C quirk—it's a fundamental optimization that makes the difference between slow and fast code.