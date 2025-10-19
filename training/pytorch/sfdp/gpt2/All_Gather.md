# All-Gather: Step-by-Step Guide

## What is All-Gather?

**All-Gather**: Each GPU contributes its piece, and everyone gets the complete picture.

Think of it like a puzzle: each person has one piece, but everyone needs to see the complete puzzle to understand it.

## The Setup: After Reduce-Scatter

Remember our text example? After reduce-scatter, each GPU only has its assigned parameter:

```
GPU 0: batch = ["The cat sat", "Dogs are fun"]
GPU 1: batch = ["I love pizza", "Weather is nice"] 
GPU 2: batch = ["Python rocks", "AI is amazing"]
GPU 3: batch = ["Hello world", "Coding is cool"]
```

After reduce-scatter, the parameters are distributed:

```python
# Each GPU has only its assigned parameter (from reduce-scatter)
GPU 0: [param_A=1.0, _____, _____, _____]  # Only has A
GPU 1: [_____, param_B=1.0, _____, _____]  # Only has B
GPU 2: [_____, _____, param_C=1.0, _____]  # Only has C
GPU 3: [_____, _____, _____, param_D=1.0]  # Only has D
```

## The Problem: Need All Parameters for Computation

Now GPU 0 wants to process new text: **"I love cats"**

But GPU 0 only has parameter A! It needs ALL parameters (A, B, C, D) to process the text properly.

```python
# GPU 0 trying to process "I love cats"
# Needs: param_A (has it ✅), param_B (missing ❌), param_C (missing ❌), param_D (missing ❌)

GPU 0: "Help! I need parameters B, C, and D to process this text!"
```

## All-Gather to the Rescue!

### Step 1: Each GPU Shares Its Parameter

```python
# Each GPU broadcasts its parameter to everyone
GPU 0 broadcasts: param_A = 1.0
GPU 1 broadcasts: param_B = 1.0  
GPU 2 broadcasts: param_C = 1.0
GPU 3 broadcasts: param_D = 1.0
```

### Step 2: Everyone Receives All Parameters

```python
# After all-gather: Everyone has the complete set!
GPU 0: [param_A=1.0, param_B=1.0, param_C=1.0, param_D=1.0]  # Complete!
GPU 1: [param_A=1.0, param_B=1.0, param_C=1.0, param_D=1.0]  # Complete!
GPU 2: [param_A=1.0, param_B=1.0, param_C=1.0, param_D=1.0]  # Complete!
GPU 3: [param_A=1.0, param_B=1.0, param_C=1.0, param_D=1.0]  # Complete!
```

### Step 3: Process the Text

Now GPU 0 can process "I love cats" using all parameters:

```python
# GPU 0 processes "I love cats"
result = param_A * "I" + param_B * "love" + param_C * "cats" + param_D * context
# Uses all parameters A, B, C, D ✅
```

### Step 4: Throw Away Borrowed Parameters

After computation, GPU 0 keeps only its assigned parameter:

```python
# GPU 0 after computation: back to just its piece
GPU 0: [param_A=1.0, _____, _____, _____]  # Freed B, C, D to save memory
```

## Detailed Example: The Attention Matrix

Let's use a simple attention matrix `W_query` of size 4×4:

### Original Complete Matrix:
```python
W_query = [[0.1, 0.2, 0.3, 0.4],
           [0.5, 0.6, 0.7, 0.8], 
           [0.9, 1.0, 1.1, 1.2],
           [1.3, 1.4, 1.5, 1.6]]
```

### After FSDP Sharding:
```python
# Each GPU has only 1 row of the W_query matrix
GPU 0: W_query_shard = [[0.1, 0.2, 0.3, 0.4]]  # Row 0
GPU 1: W_query_shard = [[0.5, 0.6, 0.7, 0.8]]  # Row 1
GPU 2: W_query_shard = [[0.9, 1.0, 1.1, 1.2]]  # Row 2  
GPU 3: W_query_shard = [[1.3, 1.4, 1.5, 1.6]]  # Row 3
```

### The Problem:
```python
# GPU 0 wants to process "I love cats"
input_embeddings = [[1.0, 0.5, 0.2, 0.8],  # "I"
                    [0.3, 1.2, 0.9, 0.4],  # "love"  
                    [0.7, 0.1, 1.5, 0.6]]  # "cats"

# GPU 0 only has 1 row - can't compute full attention! ❌
# Need: (4×4) @ (4×3) = (4×3)  
# Have: (1×4) @ (4×3) = (1×3)  ❌ Wrong size!
```

### All-Gather Solution:

#### Step 1: Share Rows
```python
GPU 0 broadcasts: [[0.1, 0.2, 0.3, 0.4]]  # Row 0
GPU 1 broadcasts: [[0.5, 0.6, 0.7, 0.8]]  # Row 1
GPU 2 broadcasts: [[0.9, 1.0, 1.1, 1.2]]  # Row 2
GPU 3 broadcasts: [[1.3, 1.4, 1.5, 1.6]]  # Row 3
```

#### Step 2: Reconstruct Complete Matrix
```python
# All GPUs now have complete W_query matrix
W_query_complete = [[0.1, 0.2, 0.3, 0.4],  # From GPU 0
                    [0.5, 0.6, 0.7, 0.8],  # From GPU 1
                    [0.9, 1.0, 1.1, 1.2],  # From GPU 2
                    [1.3, 1.4, 1.5, 1.6]]  # From GPU 3
```

#### Step 3: Compute Attention
```python
# GPU 0 computes queries using complete matrix
queries = W_query_complete @ input_embeddings.T

# Result: (4×4) @ (4×3) = (4×3) ✅ Perfect!
queries = [[0.49, 0.43, 0.61],  # Queries from row 0
           [1.18, 1.20, 1.41],  # Queries from row 1  
           [1.87, 1.97, 2.21],  # Queries from row 2
           [2.56, 2.74, 3.01]]  # Queries from row 3
```

#### Step 4: Free Memory
```python
# GPU 0 keeps only its original row
GPU 0: W_query_shard = [[0.1, 0.2, 0.3, 0.4]]  # Back to 1 row
# Memory freed: 3 rows × 4 values = 12 parameters freed
```

## Memory Usage Timeline

```python
# GPU 0 memory during attention computation:

Before all-gather:  4 parameters  (1 row)
During all-gather: 16 parameters  (4 rows) ← 4x increase!
After computation:  4 parameters  (1 row)  ← Back to normal
```

## Why This Works for Everyone

Each GPU does the same process:

```python
# GPU 1 processing "Python rocks":
# 1. All-gather: Gets complete W_query matrix
# 2. Compute: Uses all rows to compute queries  
# 3. Free: Keeps only its row (row 1)

# GPU 2 processing "AI is amazing":
# 1. All-gather: Gets complete W_query matrix
# 2. Compute: Uses all rows to compute queries
# 3. Free: Keeps only its row (row 2)
```

## The Beautiful Result

🎉 **Every GPU can process any text perfectly, but memory usage stays low most of the time!**

- **GPU 0**: Processed "I love cats" using complete attention matrix
- **GPU 1**: Processed "Python rocks" using complete attention matrix  
- **GPU 2**: Processed "AI is amazing" using complete attention matrix
- **GPU 3**: Processed "Hello world" using complete attention matrix

But each GPU only stores 1/4 of the matrix permanently!

## Real Scale Example

In your actual GPT model:

```python
# Real dimensions:
W_query = 768 × 768 = 589,824 parameters

# With 4 GPUs:
GPU 0: Stores 192 × 768 = 147,456 parameters (25%) ← Most of the time
GPU 1: Stores 192 × 768 = 147,456 parameters (25%) ← Most of the time
GPU 2: Stores 192 × 768 = 147,456 parameters (25%) ← Most of the time
GPU 3: Stores 192 × 768 = 147,456 parameters (25%) ← Most of the time

# During all-gather: Temporarily 589,824 parameters ← Brief moments
# Memory savings: 75% most of the time!
```

## Real-World Analogy

Imagine 4 chefs making a complex dish:

```
Chef 0: Has salt recipe
Chef 1: Has pepper recipe  
Chef 2: Has garlic recipe
Chef 3: Has onion recipe
```

When Chef 0 needs to cook:
1. **All-gather**: "Hey everyone, share your recipes!"
2. **Cook**: Uses all 4 recipes to make the dish
3. **Clean up**: Gives back the borrowed recipes, keeps only salt recipe

Result: Chef 0 made a complete dish using everyone's knowledge, but only stores one recipe long-term!

## Summary

**All-gather is like borrowing from your friends:**

- You temporarily get what everyone else has
- You use it to do your work properly  
- You give it back to save space
- Everyone does the same thing

This lets FSDP maintain low memory usage while ensuring each GPU can perform complete computations when needed. It's the perfect complement to reduce-scatter!