Query (Q): "What am I looking for?"

Each token learns to express what information it needs

Example: The token "verb" might query for "subject" information

Key (K): "How should I be found/accessed?"

Each token learns to advertise what information it contains

Example: The token "cat" might advertise itself as a "noun" or "subject"

Value (V): "What information do I actually provide?"

Each token learns what content to contribute when attended to

This can be different from its key representation

Example: "cat" might provide semantic features like [animal, furry, pet]

The mechanism:

att_scores = matmul(query, key.T)  # "Which tokens match what I'm looking for?"
att_weights = softmax(att_scores)   # "How much should I attend to each?"
context = matmul(att_weights, value) # "Aggregate the actual information"

Why separate K and V?

Key: Optimized for matching/similarity (used in dot product)

Value: Optimized for information content (what gets passed forward)

This separation gives the model flexibility: a token can be "found" based on one representation but "provide" different information

So during training, the model learns:

How each token should query for relevant context

How each token should advertise its relevance

What information each token should contribute when attended to

This is why attention is so powerful—it's a fully differentiable, learned lookup mechanism!