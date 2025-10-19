import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

from lora_linear import LoRALinear

class PEFTWrapper:
    """
    PEFT-style wrapper that adds LoRA adapters to existing GPT-2 models.
    Works by replacing target modules in-place with LoRA versions.
    """
    
    def __init__(self, model, target_modules=None, lora_rank=16, lora_alpha=16):
        self.base_model = model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # Default target modules for GPT-2
        if target_modules is None:
            target_modules = ["out_head"]  # Start with output layer
        
        self.target_modules = target_modules
        self.original_modules = {}  # Store originals for restoration
        
        # Replace modules with LoRA versions
        self._inject_lora_adapters()
        
    def _inject_lora_adapters(self):
        """Replace target modules with LoRA-enabled versions"""
        for module_name in self.target_modules:
            original_module = self._get_module_by_name(module_name)
            
            if original_module is None:
                print(f"Warning: Module '{module_name}' not found")
                continue
                
            # Store original for potential restoration
            self.original_modules[module_name] = original_module
            
            # Create LoRA version
            lora_module = self._create_lora_module(original_module)
            
            # Replace in model
            self._set_module_by_name(module_name, lora_module)
            
    def _get_module_by_name(self, module_name):
        """Navigate to module by dot-separated name"""
        parts = module_name.split('.')
        current = self.base_model
        
        try:
            for part in parts:
                current = getattr(current, part)
            return current
        except AttributeError:
            return None
    
    def _set_module_by_name(self, module_name, new_module):
        """Set module by dot-separated name"""
        parts = module_name.split('.')
        parent = self.base_model
        
        # Navigate to parent
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Set the module
        setattr(parent, parts[-1], new_module)
    
    def _create_lora_module(self, original_module):
        """Create LoRA version of a module"""
        if hasattr(original_module, 'weight'):
            # It's a Linear layer
            d_out, d_in = original_module.weight.shape
            has_bias = original_module.bias is not None
            
            lora_module = LoRALinear(
                d_in, d_out, 
                bias=has_bias,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha
            )
            
            # Copy original weights
            lora_module.weight = original_module.weight.copy()
            if has_bias:
                lora_module.bias = original_module.bias.copy()
                
            return lora_module
        else:
            raise ValueError(f"Don't know how to create LoRA version of {type(original_module)}")
    
    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model"""
        return self.base_model.forward(*args, **kwargs)
    
    def enable_lora_training(self):
        """Enable LoRA training mode - freeze original weights"""
        for module_name in self.target_modules:
            lora_module = self._get_module_by_name(module_name)
            if hasattr(lora_module, 'enable_lora'):
                lora_module.enable_lora()
                lora_module.freeze_original = True
    
    def disable_lora_training(self):
        """Disable LoRA training mode"""
        for module_name in self.target_modules:
            lora_module = self._get_module_by_name(module_name)
            if hasattr(lora_module, 'disable_lora'):
                lora_module.disable_lora()
    
    def get_lora_parameters(self):
        """Get all LoRA parameters for training"""
        lora_params = {}
        
        for module_name in self.target_modules:
            lora_module = self._get_module_by_name(module_name)
            if hasattr(lora_module, 'get_lora_parameters'):
                module_params = lora_module.get_lora_parameters()
                for param_name, param_value in module_params.items():
                    lora_params[f"{module_name}_{param_name}"] = param_value
        
        return lora_params
    
    def count_parameters(self):
        """Count total and LoRA parameters"""
        total_params = 0
        lora_params = 0
        
        # Count all parameters in base model
        def count_module_params(module, prefix=""):
            nonlocal total_params, lora_params
            
            for name, attr in vars(module).items():
                if isinstance(attr, np.ndarray):
                    total_params += attr.size
                    
                    # Check if it's a LoRA parameter
                    if 'lora_' in name:
                        lora_params += attr.size
                elif hasattr(attr, '__dict__'):  # It's another module
                    count_module_params(attr, f"{prefix}.{name}" if prefix else name)
        
        count_module_params(self.base_model)
        
        return {
            'total_parameters': total_params,
            'lora_parameters': lora_params,
            'trainable_percentage': (lora_params / total_params) * 100 if total_params > 0 else 0
        }
    
    def save_lora_weights(self, filepath):
        """Save only LoRA weights"""
        lora_params = self.get_lora_parameters()
        np.savez(filepath, **lora_params)
        print(f"LoRA weights saved to {filepath}")
    
    def load_lora_weights(self, filepath):
        """Load LoRA weights"""
        data = np.load(filepath)
        
        for param_name, param_value in data.items():
            # Parse module name and parameter name
            parts = param_name.split('_')
            if len(parts) >= 3:  # module_lora_A or module_lora_B
                module_name = '_'.join(parts[:-2])
                lora_param_name = '_'.join(parts[-2:])
                
                lora_module = self._get_module_by_name(module_name)
                if lora_module and hasattr(lora_module, lora_param_name):
                    setattr(lora_module, lora_param_name, param_value)
        
        print(f"LoRA weights loaded from {filepath}")
    
    def restore_original_modules(self):
        """Restore original modules (remove LoRA adapters)"""
        for module_name, original_module in self.original_modules.items():
            self._set_module_by_name(module_name, original_module)

def get_peft_model(model, target_modules=None, lora_rank=16, lora_alpha=16):
    """
    PEFT-style function to wrap any model with LoRA adapters.
    Similar to Hugging Face PEFT's get_peft_model().
    """
    return PEFTWrapper(model, target_modules, lora_rank, lora_alpha)