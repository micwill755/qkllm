# Example usage of expert monitoring and improvement
from deepseek.deepseek_from_scratch import MoE

# Initialize MoE with tracking
moe = MoE(emb_dim=512, num_experts=8, top_k=2, expert_dim=2048)

# During training, after processing batches:
def monitor_experts(moe, step):
    stats = moe.get_expert_stats()
    
    print(f"Step {step} - Expert Utilization:")
    for i, util in enumerate(stats['utilization']):
        print(f"  Expert {i}: {util:.3f}")
    
    if stats['unused_experts']:
        print(f"Unused experts: {stats['unused_experts']}")
    
    # Reinitialize unused experts every 1000 steps
    if step % 1000 == 0:
        reinitialized = moe.reinitialize_unused_experts()
        if reinitialized:
            print(f"Reinitialized experts: {reinitialized}")
            moe.tracker.reset()  # Reset tracking after reinitialization

# Common patterns for improving expert utilization:

# 1. Auxiliary Load Balancing Loss (add to your training loss)
def calculate_total_loss(model_loss, moe, expert_scores, seq_len):
    load_balance_loss = moe.apply_load_balancing_loss(expert_scores, seq_len)
    return model_loss + load_balance_loss

# 2. Expert Dropout (randomly disable some experts during training)
def expert_dropout(expert_scores, dropout_rate=0.1):
    import random
    num_experts = len(expert_scores) // seq_len
    for row in range(seq_len):
        for col in range(num_experts):
            if random.random() < dropout_rate:
                expert_scores[row * num_experts + col] = float('-inf')

# 3. Temperature Scaling (adjust router sharpness)
def apply_temperature(expert_scores, temperature=1.0):
    return [score / temperature for score in expert_scores]