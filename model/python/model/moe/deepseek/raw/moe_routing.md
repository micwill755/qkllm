# Mixture of Experts (MoE) Routing Optimization Guide

## What is MoE Routing?

Imagine you have a team of 256 specialists (experts), but for each question, you only want to consult the 8 most relevant ones. The **router** is like a smart dispatcher that decides which experts should handle each input token.

```
Input: "The capital of France is..."
Router: Send to [Geography Expert, Language Expert, Facts Expert]
Not to: [Math Expert, Code Expert, Science Expert, ...]
```

## The Core Problem

Without optimization, routing creates three major issues:

### 1. **Load Imbalance**
```
Expert 1: ████████████████████ (overloaded - 1000 tokens)
Expert 2: ██                   (underused - 100 tokens)  
Expert 3: ████████████████████ (overloaded - 950 tokens)
Expert 4: █                    (underused - 50 tokens)
```

### 2. **Communication Overhead**
```
GPU 1: [Expert A, Expert B] ←── Token from GPU 3 (expensive transfer)
GPU 2: [Expert C, Expert D] ←── Token from GPU 1 (expensive transfer)
GPU 3: [Expert E, Expert F] ←── Token from GPU 2 (expensive transfer)
```

### 3. **Training Instability**
Some experts never get trained, others get overtrained.

---

## Optimization Techniques

### 1. Top-K Routing with Load Balancing

**Basic Idea**: Route each token to K best experts, but add penalties to prevent overuse.

```python
class TopKRouter:
    def __init__(self, d_model, num_experts, k=2):
        self.gate = Linear(d_model, num_experts)
        self.k = k
        self.num_experts = num_experts
    
    def forward(self, x):
        # Get routing scores for all experts
        gate_scores = self.gate(x)  # [batch, tokens, num_experts]
        
        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.k)
        
        # Load balancing: count how many tokens each expert gets
        expert_counts = torch.zeros(self.num_experts)
        for expert_idx in top_k_indices.flatten():
            expert_counts[expert_idx] += 1
            
        # Auxiliary loss to encourage balanced usage
        load_balance_loss = torch.var(expert_counts)
        
        return top_k_indices, top_k_scores, load_balance_loss
```

**Example**:
```
Token: "Paris"
All Expert Scores: [0.1, 0.8, 0.3, 0.9, 0.2, 0.1, ...]
Top-2 Selection: Expert 4 (0.9), Expert 2 (0.8)
Load Balance Check: Expert 4 used 500 times, Expert 2 used 200 times
Penalty: Increase cost for overused Expert 4
```

### 2. Hierarchical Routing

**Basic Idea**: First choose expert groups, then specific experts within groups.

```
Step 1: Choose Group
Input → [Math Group, Language Group, Science Group, Art Group]
         ↓ (select Language Group)
         
Step 2: Choose Expert within Group  
Language Group → [Grammar Expert, Translation Expert, Literature Expert]
                 ↓ (select Grammar Expert)
```

```python
class HierarchicalRouter:
    def __init__(self, d_model, num_groups, experts_per_group):
        self.group_gate = Linear(d_model, num_groups)
        self.expert_gates = nn.ModuleList([
            Linear(d_model, experts_per_group) for _ in range(num_groups)
        ])
    
    def forward(self, x):
        # Step 1: Route to groups
        group_scores = softmax(self.group_gate(x))
        selected_group = torch.argmax(group_scores, dim=-1)
        
        # Step 2: Route within selected group
        expert_scores = self.expert_gates[selected_group](x)
        selected_expert = torch.argmax(expert_scores, dim=-1)
        
        return selected_group, selected_expert
```

**Benefits**:
- Reduces routing computation: O(√N) instead of O(N)
- Better locality for distributed training
- More interpretable routing decisions

### 3. Expert Choice Routing

**Traditional**: Tokens choose experts
**Expert Choice**: Experts choose tokens

```python
class ExpertChoiceRouter:
    def __init__(self, d_model, num_experts, capacity_factor=1.25):
        self.gate = Linear(d_model, num_experts)
        self.capacity_factor = capacity_factor
        
    def forward(self, x, num_tokens):
        # Each expert can handle this many tokens
        expert_capacity = int(num_tokens * self.capacity_factor / self.num_experts)
        
        # Get all routing scores
        gate_scores = self.gate(x)  # [batch, tokens, experts]
        
        # Each expert selects its top tokens
        expert_assignments = {}
        for expert_id in range(self.num_experts):
            expert_scores = gate_scores[:, :, expert_id]
            top_tokens = torch.topk(expert_scores, expert_capacity).indices
            expert_assignments[expert_id] = top_tokens
            
        return expert_assignments
```

**Example**:
```
Traditional: Token "Paris" → chooses [Geography, Language] experts
Expert Choice: Geography Expert → chooses ["Paris", "London", "Berlin"] tokens
               Language Expert → chooses ["hello", "bonjour", "Paris"] tokens
```

### 4. Switch Transformer Routing

**Key Innovation**: Simplified routing with better load balancing.

```python
class SwitchRouter:
    def __init__(self, d_model, num_experts):
        self.gate = Linear(d_model, num_experts)
        
    def forward(self, x):
        gate_logits = self.gate(x)
        
        # Route to single best expert (not top-k)
        expert_indices = torch.argmax(gate_logits, dim=-1)
        expert_scores = torch.softmax(gate_logits, dim=-1)
        
        # Load balancing loss
        expert_counts = torch.bincount(expert_indices.flatten(), minlength=self.num_experts)
        load_balance_loss = self.num_experts * torch.sum(expert_counts * expert_counts) / (torch.sum(expert_counts) ** 2)
        
        return expert_indices, expert_scores, load_balance_loss
```

**Benefits**:
- Simpler than top-k routing
- Better load balancing through auxiliary loss
- Faster inference (only 1 expert per token)

### 5. Hardware-Aware Routing

**Basic Idea**: Consider where experts are located when routing.

```python
class HardwareAwareRouter:
    def __init__(self, d_model, num_experts, device_map):
        self.gate = Linear(d_model, num_experts)
        self.device_map = device_map  # expert_id → device_id
        
    def forward(self, x, current_device):
        gate_scores = self.gate(x)
        
        # Boost scores for local experts
        for expert_id, device_id in self.device_map.items():
            if device_id == current_device:
                gate_scores[:, :, expert_id] += 0.1  # locality bonus
                
        # Standard top-k selection
        top_k_scores, top_k_indices = torch.topk(gate_scores, k=2)
        
        return top_k_indices, top_k_scores
```

**Example**:
```
GPU 0: [Expert 0, Expert 1, Expert 2, Expert 3]
GPU 1: [Expert 4, Expert 5, Expert 6, Expert 7]

Token on GPU 0: "Hello"
Raw scores: [0.3, 0.2, 0.1, 0.4, 0.9, 0.8, 0.1, 0.2]
Local bonus: [0.4, 0.3, 0.2, 0.5, 0.9, 0.8, 0.1, 0.2]
Selection: Expert 4 (0.9) + Expert 3 (0.5) - mixed local/remote
```

---

## Advanced Techniques

### Dynamic Expert Selection

Adjust the number of active experts based on input complexity:

```python
class DynamicRouter:
    def forward(self, x):
        # Measure input complexity
        complexity = torch.std(x, dim=-1)  # high std = complex input
        
        # Simple inputs use fewer experts
        k = 1 if complexity < 0.5 else 2 if complexity < 1.0 else 4
        
        return self.route_top_k(x, k)
```

### Routing with Memory

Cache routing decisions for similar patterns:

```python
class MemoryRouter:
    def __init__(self):
        self.routing_cache = {}  # pattern → expert_ids
        
    def forward(self, x):
        # Create pattern signature
        pattern = torch.mean(x, dim=0).round(decimals=2)
        pattern_key = tuple(pattern.tolist())
        
        if pattern_key in self.routing_cache:
            return self.routing_cache[pattern_key]
            
        # Compute routing normally
        expert_ids = self.compute_routing(x)
        self.routing_cache[pattern_key] = expert_ids
        
        return expert_ids
```

---

## Real-World Examples

### DeepSeek-V3 Configuration
```
Total Experts: 256
Active per Token: 9 (1 shared + 8 routed)
Total Parameters: 671B
Active Parameters: 37B (5.5% utilization)
Routing Strategy: Top-8 with load balancing
```

### Switch Transformer Configuration
```
Total Experts: 2048
Active per Token: 1
Routing Strategy: Single expert with auxiliary loss
Load Balance Factor: 0.01
```

### GLaM Configuration
```
Total Experts: 64
Active per Token: 2
Routing Strategy: Top-2 with expert parallelism
Communication: All-to-all between devices
```

---

## Key Takeaways

1. **Load Balancing is Critical**: Without it, most experts remain unused
2. **Communication Costs Matter**: Local experts are often better than optimal distant ones
3. **Simplicity Often Wins**: Switch Transformer's single-expert routing often outperforms complex schemes
4. **Hardware Co-design**: Best routing strategies consider the underlying distributed system
5. **Training vs Inference**: Different routing strategies may be optimal for training vs serving

The goal is finding the sweet spot between model quality, computational efficiency, and system practicality.