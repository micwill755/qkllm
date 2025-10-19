def calculate_gpu_memory_requirements(num_parameters, precision="fp16", training=True):
    """
    Calculate GPU memory requirements for a large language model.
    
    Args:
        num_parameters: Number of model parameters (e.g., 670e9 for 670B)
        precision: Weight precision ("fp16", "fp32", "int8")
        training: Whether calculating for training (True) or inference (False)
    
    Returns:
        dict: Memory breakdown and GPU requirements
    """
    
    # Bytes per parameter based on precision
    precision_bytes = {
        "fp32": 4,
        "fp16": 2, 
        "bf16": 2,
        "int8": 1
    }
    
    bytes_per_param = precision_bytes.get(precision.lower(), 2)
    
    # Model weights
    model_memory = num_parameters * bytes_per_param
    
    if training:
        # Training requires: weights + gradients + optimizer states
        # Adam optimizer: 2x params for momentum + variance
        gradients_memory = num_parameters * bytes_per_param
        optimizer_memory = num_parameters * 2 * 4  # Adam states in FP32
        
        # Activations (rough estimate: ~20% of model size for large models)
        activations_memory = model_memory * 0.2
        
        total_memory = model_memory + gradients_memory + optimizer_memory + activations_memory
        
        breakdown = {
            "model_weights": model_memory,
            "gradients": gradients_memory, 
            "optimizer_states": optimizer_memory,
            "activations": activations_memory,
            "total": total_memory
        }
    else:
        # Inference only needs weights + small activation buffer
        activations_memory = model_memory * 0.05
        total_memory = model_memory + activations_memory
        
        breakdown = {
            "model_weights": model_memory,
            "activations": activations_memory,
            "total": total_memory
        }
    
    return breakdown

def calculate_gpu_requirements(memory_breakdown, gpu_memory_gb=80):
    """Calculate number of GPUs needed based on memory requirements."""
    
    total_memory_gb = memory_breakdown["total"] / (1024**3)
    num_gpus = int(np.ceil(total_memory_gb / gpu_memory_gb))
    
    return {
        "total_memory_gb": total_memory_gb,
        "gpu_memory_gb": gpu_memory_gb,
        "num_gpus_needed": num_gpus,
        "memory_per_gpu_gb": total_memory_gb / num_gpus if num_gpus > 0 else 0
    }

# Example calculation for 670B parameter model
if __name__ == "__main__":
    import numpy as np
    
    # 670 billion parameters
    params_670b = 670e9
    
    print("=== 670B Parameter Model Memory Requirements ===\n")
    
    # Training requirements
    training_memory = calculate_gpu_memory_requirements(params_670b, "fp16", training=True)
    training_gpus = calculate_gpu_requirements(training_memory, gpu_memory_gb=80)
    
    print("TRAINING (FP16):")
    print(f"Model weights: {training_memory['model_weights']/1e9:.1f} GB")
    print(f"Gradients: {training_memory['gradients']/1e9:.1f} GB") 
    print(f"Optimizer states: {training_memory['optimizer_states']/1e9:.1f} GB")
    print(f"Activations: {training_memory['activations']/1e9:.1f} GB")
    print(f"Total memory: {training_gpus['total_memory_gb']:.1f} GB")
    print(f"GPUs needed (80GB each): {training_gpus['num_gpus_needed']}")
    print()
    
    # Inference requirements  
    inference_memory = calculate_gpu_memory_requirements(params_670b, "fp16", training=False)
    inference_gpus = calculate_gpu_requirements(inference_memory, gpu_memory_gb=80)
    
    print("INFERENCE (FP16):")
    print(f"Model weights: {inference_memory['model_weights']/1e9:.1f} GB")
    print(f"Activations: {inference_memory['activations']/1e9:.1f} GB")
    print(f"Total memory: {inference_gpus['total_memory_gb']:.1f} GB") 
    print(f"GPUs needed (80GB each): {inference_gpus['num_gpus_needed']}")
    print()
    
    # Different GPU configurations
    gpu_configs = [
        ("A100 80GB", 80),
        ("H100 80GB", 80), 
        ("A6000 48GB", 48),
        ("RTX 4090 24GB", 24)
    ]
    
    print("=== GPU Requirements by Hardware ===")
    for gpu_name, gpu_mem in gpu_configs:
        training_req = calculate_gpu_requirements(training_memory, gpu_mem)
        inference_req = calculate_gpu_requirements(inference_memory, gpu_mem)
        print(f"{gpu_name}: Training={training_req['num_gpus_needed']}, Inference={inference_req['num_gpus_needed']}")