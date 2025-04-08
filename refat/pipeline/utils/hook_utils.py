import torch
from torch import nn
from typing import List, Tuple, Callable, Dict, Any, Optional
from contextlib import contextmanager

@contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]] = None,
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]] = None,
    module_backward_hooks: List[Tuple[torch.nn.Module, Callable]] = None
):
    """
    Context manager for adding and removing hooks from model modules.
    
    Args:
        module_forward_pre_hooks: List of (module, hook_fn) tuples for forward pre-hooks
        module_forward_hooks: List of (module, hook_fn) tuples for forward hooks
        module_backward_hooks: List of (module, hook_fn) tuples for backward hooks
        
    Yields:
        None
    """
    # Initialize lists to store hook handles
    handles = []
    
    try:
        # Add forward pre-hooks
        if module_forward_pre_hooks:
            for module, hook_fn in module_forward_pre_hooks:
                handle = module.register_forward_pre_hook(hook_fn)
                handles.append(handle)
        
        # Add forward hooks
        if module_forward_hooks:
            for module, hook_fn in module_forward_hooks:
                handle = module.register_forward_hook(hook_fn)
                handles.append(handle)
        
        # Add backward hooks
        if module_backward_hooks:
            for module, hook_fn in module_backward_hooks:
                handle = module.register_backward_hook(hook_fn)
                handles.append(handle)
        
        yield
        
    finally:
        # Remove all hooks
        for handle in handles:
            handle.remove()

def get_activation_capture_hook(cache: Dict[str, Any], key: str):
    """
    Returns a hook function that captures activations into a cache.
    
    Args:
        cache: Dictionary to store activations in
        key: Key to store activations under
    
    Returns:
        Hook function
    """
    def hook_fn(module, input, output):
        cache[key] = output.detach().clone()
    return hook_fn

def get_activation_modifier_hook(
    modify_fn: Callable[[torch.Tensor], torch.Tensor]
):
    """
    Returns a hook function that modifies activations.
    
    Args:
        modify_fn: Function that takes and returns a tensor
    
    Returns:
        Hook function
    """
    def hook_fn(module, input, output):
        return modify_fn(output)
    return hook_fn

def get_input_modifier_pre_hook(
    modify_fn: Callable[[torch.Tensor], torch.Tensor]
):
    """
    Returns a pre-hook function that modifies inputs.
    
    Args:
        modify_fn: Function that takes and returns a tensor
    
    Returns:
        Pre-hook function
    """
    def hook_fn(module, input):
        modified_input = list(input)
        modified_input[0] = modify_fn(input[0])
        return tuple(modified_input)
    return hook_fn 