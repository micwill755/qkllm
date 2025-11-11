# Attention Mechanism

## Query (Q): "What am I looking for?"

Each token learns to express what information it needs.

**Example:** The token "verb" might query for "subject" information

## Key (K): "How should I be found/accessed?"

Each token learns to advertise what information it contains.

**Example:** The token "cat" might advertise itself as a "noun" or "subject"

## Value (V): "What information do I actually provide?"

Each token learns what content to contribute when attended to. This can be different from its key representation.

**Example:** "cat" might provide semantic features like [animal, furry, pet]

## The Mechanism

```python
att_scores = matmul(query, key.T)  # "Which tokens match what I'm looking for?"
att_weights = softmax(att_scores)   # "How much should I attend to each?"
context = matmul(att_weights, value) # "Aggregate the actual information"
```

## Why Separate K and V?

- **Key:** Optimized for matching/similarity (used in dot product)
- **Value:** Optimized for information content (what gets passed forward)

This separation gives the model flexibility: a token can be "found" based on one representation but "provide" different information.

## What the Model Learns

During training, the model learns:
- How each token should query for relevant context
- How each token should advertise its relevance
- What information each token should contribute when attended to

This is why attention is so powerful—it's a fully differentiable, learned lookup mechanism!

## Grouped Query Attention (GQA)

An optimization where multiple Query heads share the same Key/Value pairs.

**Architecture:**
- Each head has its OWN unique Query (Q)
- Multiple heads SHARE the same Key (K) and Value (V)

### Example: 8 Q heads, 2 KV heads (4:1 grouping)

```
Group 1: 4 different queries, same K/V
Q1: "looking for subjects"        \
Q2: "looking for verbs"            } → K1, V1 (shared)
Q3: "looking for adjectives"       |
Q4: "looking for sentence start"  /

Group 2: 4 different queries, same K/V  
Q5: "looking for entities"        \
Q6: "looking for relationships"    } → K2, V2 (shared)
Q7: "looking for temporal info"    |
Q8: "looking for negations"       /
```

**Library Analogy:**

Think of it as one well-organized library (K/V) serving multiple researchers (Q) with different interests:

```python
# One catalog system (K/V)
keys = ["book_id_1", "book_id_2", "book_id_3"]  # How books are indexed
values = [book1_content, book2_content, book3_content]  # Actual books

# Different researchers asking different questions:
q1 = "books about history"     # Matches certain keys
q2 = "books from 1800s"        # Matches different keys  
q3 = "books with illustrations" # Matches yet other keys

# Same K/V, different attention patterns!
```

**Why It Works:**

Different queries can ask different questions about the SAME key-value database. The queries learn diverse attention patterns while K/V remains general enough to serve all of them.

**Benefits:**
- Memory efficient: Fewer K/V pairs to store (critical for long sequences)
- Queries are cheap, K/V cache is expensive during generation
- 4x less memory than Multi-Head Attention with minimal quality loss

**Comparison:**
- MHA: 8 queries need 8 separate K/V pairs = 8 × (K + V) parameters
- GQA: 8 queries share 2 K/V pairs = 8 × Q + 2 × (K + V) parameters
- MQA: 8 queries share 1 K/V pair = 8 × Q + 1 × (K + V) parameters (fastest, but lower quality)