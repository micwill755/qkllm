# Beam Search: Advanced Decoding for LLM Inference

## What is Beam Search?

Beam search is a decoding strategy that explores multiple possible sequences simultaneously to find higher-quality text generation. Instead of greedily selecting the most probable token at each step, beam search maintains several candidate sequences (beams) and selects the globally best sequence.

## Decoding Strategy Comparison

### Greedy Decoding (Beam Width = 1)
```
Input: "The cat"

Step 1: "The cat" → "sat" (highest probability: 0.6)
Step 2: "The cat sat" → "on" (highest probability: 0.5)
Step 3: "The cat sat on" → "the" (highest probability: 0.7)

Final: "The cat sat on the"
```

### Beam Search (Beam Width = 3)
```
Input: "The cat"

Step 1: Keep top 3 continuations
├── "The cat sat" (prob: 0.6)
├── "The cat is" (prob: 0.3)
└── "The cat was" (prob: 0.2)

Step 2: Expand each beam, keep top 3 overall
├── "The cat sat on" (cumulative prob: 0.6 × 0.5 = 0.30)
├── "The cat is sleeping" (cumulative prob: 0.3 × 0.8 = 0.24)
└── "The cat sat down" (cumulative prob: 0.6 × 0.3 = 0.18)

Step 3: Continue expansion...
├── "The cat sat on the" (cumulative prob: 0.30 × 0.7 = 0.21)
├── "The cat is sleeping peacefully" (cumulative prob: 0.24 × 0.6 = 0.144)
└── "The cat sat down quietly" (cumulative prob: 0.18 × 0.5 = 0.09)

Final: "The cat sat on the" (highest cumulative probability)
```

## Beam Search Tree Visualization

```
                    "The cat"
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   "sat" (0.6)     "is" (0.3)      "was" (0.2)
        │               │               │
    ┌───┼───┐       ┌───┼───┐       ┌───┼───┐
    │   │   │       │   │   │       │   │   │
  "on" "down" "up" "sleeping" "big" "running" "small"
  (0.5) (0.3) (0.2)  (0.8)   (0.1)   (0.6)   (0.3)

Beam Width = 3: Keep only top 3 paths at each level
Final Beams:
1. "The cat sat on" (0.6 × 0.5 = 0.30)
2. "The cat is sleeping" (0.3 × 0.8 = 0.24)  
3. "The cat sat down" (0.6 × 0.3 = 0.18)
```

## Quality vs Performance Trade-offs

### Beam Width Impact

| Beam Width | Quality | Speed | Memory Usage | Use Case |
|------------|---------|-------|--------------|----------|
| 1 (Greedy) | Lower | Fastest | Minimal | Chat, real-time |
| 2-4 | Good | Fast | Moderate | General purpose |
| 5-8 | Better | Slower | High | Translation, summarization |
| 10+ | Best | Slowest | Very High | Critical applications |

### Performance Characteristics

```
Generation Quality vs Beam Width:
Quality ▲
        │     ┌─────────────
        │    ╱
        │   ╱
        │  ╱
        │ ╱
        └─────────────────► Beam Width
         1  2  4  6  8  10

Memory Usage vs Beam Width:
Memory  ▲
        │           ╱╱╱
        │         ╱╱
        │       ╱╱
        │     ╱╱
        │   ╱╱
        └─────────────────► Beam Width
         1  2  4  6  8  10
```

## Memory Management with PagedAttention

### Traditional Memory Allocation
```
Without PagedAttention (beam_width=4):
┌─────────────────────────────────────────────────────────────┐
│ Beam 1: [████████████░░░░░░░░] Pre-allocated max_seq_len    │
│ Beam 2: [████████████░░░░░░░░] Pre-allocated max_seq_len    │  
│ Beam 3: [████████████░░░░░░░░] Pre-allocated max_seq_len    │
│ Beam 4: [████████████░░░░░░░░] Pre-allocated max_seq_len    │
│ Total: 4x memory waste per sequence                         │
└─────────────────────────────────────────────────────────────┘
```

### PagedAttention Memory Optimization
```
With PagedAttention (beam_width=4):
┌─────────────────────────────────────────────────────────────┐
│ Shared Prefix Blocks:                                       │
│ Block 0: [████████████████] "The cat" (shared by all beams) │
│                                                             │
│ Divergent Suffix Blocks:                                    │
│ Block 1: [████████████████] Beam 1: "sat on"               │
│ Block 2: [████████████████] Beam 2: "is sleeping"          │
│ Block 3: [████████████████] Beam 3: "was running"          │
│ Block 4: [████████████████] Beam 4: "might be"             │
│                                                             │
│ Memory Savings: 75% reduction through prefix sharing        │
└─────────────────────────────────────────────────────────────┘
```

## Beam Search Optimization Strategies

### 1. Dynamic Beam Pruning
```
Beam Evolution Over Time:

Initial: beam_width = 8
    ├── Beam 1 (score: 0.95)
    ├── Beam 2 (score: 0.92)
    ├── Beam 3 (score: 0.88)
    ├── Beam 4 (score: 0.85)
    ├── Beam 5 (score: 0.45) ← Low score
    ├── Beam 6 (score: 0.42) ← Low score  
    ├── Beam 7 (score: 0.38) ← Low score
    └── Beam 8 (score: 0.35) ← Low score

Mid-generation: Prune to beam_width = 4
    ├── Beam 1 (score: 0.95) ✓
    ├── Beam 2 (score: 0.92) ✓
    ├── Beam 3 (score: 0.88) ✓
    └── Beam 4 (score: 0.85) ✓

Final: Select best beam
    └── Beam 1 (score: 0.95) → Final output
```

### 2. Adaptive Beam Width
```
Request Analysis → Beam Width Selection:

Short Response (< 50 tokens):
├── Use Case: Chat, Q&A
├── Strategy: Greedy (beam_width = 1)
└── Benefit: Minimal latency

Medium Response (50-200 tokens):
├── Use Case: Explanations, summaries
├── Strategy: Small beam (beam_width = 2-4)
└── Benefit: Quality + speed balance

Long Response (200+ tokens):
├── Use Case: Articles, translations
├── Strategy: Large beam (beam_width = 4-8)
└── Benefit: Maximum quality
```

### 3. Early Stopping Optimization
```
Beam Score Monitoring:

Generation Step 1:
├── Best Beam: 0.85
├── Second Best: 0.82
└── Difference: 0.03 (continue)

Generation Step 5:
├── Best Beam: 0.91
├── Second Best: 0.73
└── Difference: 0.18 (threshold exceeded → stop early)

Result: 40% faster generation with minimal quality loss
```

## Batch Processing Optimizations

### Mixed Decoding Strategies
```
Single Batch with Multiple Strategies:
┌─────────────────────────────────────────────────────────────┐
│ Request 1: Chat response → Greedy (beam_width=1)            │
│ Request 2: Translation → Beam search (beam_width=4)         │
│ Request 3: Creative writing → Sampling (temperature=0.8)    │
│ Request 4: Summarization → Beam search (beam_width=6)       │
│                                                             │
│ Total GPU Utilization: Optimized across different needs     │
└─────────────────────────────────────────────────────────────┘
```

### Batch Beam Search vs Individual Requests
```
Traditional Approach:
4 requests × beam_width=4 = 16 total sequences
├── High memory usage
├── Lower batch efficiency
└── Suboptimal GPU utilization

Optimized Approach:
1 request × beam_width=4, replicated 4 times = 4 sequences
├── 75% memory reduction
├── Higher batch efficiency  
└── Better GPU utilization
```

## PagedAttention-Specific Benefits

### Block Sharing Across Beams
```
Beam Divergence Timeline:

Time 0: All beams identical
┌─────────────────────────────────────────┐
│ All Beams: [Block 0] "The weather is"   │
│ Memory Usage: 1 block shared            │
└─────────────────────────────────────────┘

Time 1: Beams start diverging  
┌─────────────────────────────────────────┐
│ Beam 1: [Block 0][Block 1] "...sunny"   │
│ Beam 2: [Block 0][Block 2] "...cloudy"  │
│ Beam 3: [Block 0][Block 3] "...rainy"   │
│ Beam 4: [Block 0][Block 4] "...windy"   │
│ Memory Usage: 1 shared + 4 unique       │
└─────────────────────────────────────────┘

Time 2: Full divergence
┌─────────────────────────────────────────┐
│ Each beam has unique suffix blocks      │
│ Prefix still shared across all beams    │
│ Memory Efficiency: 60-80% vs traditional│
└─────────────────────────────────────────┘
```

### Copy-on-Write Semantics
```
Beam Branching Process:

1. Initial State: Single sequence
   └── [Shared Blocks 0,1,2]

2. Beam Creation: Copy pointers, not data
   ├── Beam 1 → [Blocks 0,1,2] (shared)
   ├── Beam 2 → [Blocks 0,1,2] (shared)
   ├── Beam 3 → [Blocks 0,1,2] (shared)
   └── Beam 4 → [Blocks 0,1,2] (shared)

3. Divergence: Allocate new blocks only when needed
   ├── Beam 1 → [Blocks 0,1,2,5] (new block 5)
   ├── Beam 2 → [Blocks 0,1,2,6] (new block 6)
   ├── Beam 3 → [Blocks 0,1,2,7] (new block 7)
   └── Beam 4 → [Blocks 0,1,2,8] (new block 8)

Memory Efficiency: Only pay for differences, not duplicates
```

## Production Optimization Guidelines

### Use Case Recommendations

**Real-time Chat Applications:**
- Beam Width: 1 (greedy)
- Rationale: Minimize latency, acceptable quality loss
- Memory Impact: Minimal

**Translation Services:**
- Beam Width: 4-6
- Rationale: Quality critical, users expect some latency
- Memory Impact: Moderate, offset by PagedAttention sharing

**Content Generation:**
- Beam Width: 2-4 with sampling
- Rationale: Balance creativity and coherence
- Memory Impact: Low to moderate

**Critical Applications (Legal, Medical):**
- Beam Width: 6-10
- Rationale: Maximum quality, latency acceptable
- Memory Impact: High, but manageable with PagedAttention

### Performance Tuning Matrix

| Priority | Beam Width | Additional Settings | Expected Performance |
|----------|------------|-------------------|---------------------|
| **Latency** | 1 | Greedy decoding | 100% speed, 85% quality |
| **Balanced** | 2-4 | Early stopping | 60% speed, 95% quality |
| **Quality** | 4-8 | Full beam search | 25% speed, 100% quality |
| **Maximum Quality** | 8+ | Large beam + sampling | 15% speed, 105% quality |

## Summary

Beam search provides a powerful mechanism for improving generation quality at the cost of increased computational and memory requirements. When combined with PagedAttention:

1. **Memory Efficiency**: Block sharing reduces beam search memory overhead by 60-80%
2. **Flexible Batching**: Mixed decoding strategies in single batches optimize GPU utilization  
3. **Dynamic Optimization**: Adaptive beam width and pruning balance quality vs performance
4. **Production Ready**: Proven strategies for different use cases and performance requirements

The key insight is that PagedAttention transforms beam search from a memory-prohibitive technique into a practical optimization tool for production LLM serving.