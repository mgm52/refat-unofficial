import torch
import os
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
import random

from refat.pipeline.utils.hook_utils import add_hooks
from refat.model_utils.model_base import ModelBase

def get_mean_activations_pre_hook(layer, cache: torch.Tensor, n_samples, positions: List[int]):
    """
    Hook function to calculate mean activations for specific positions.
    
    Args:
        layer: Layer index
        cache: Tensor to store mean activations
        n_samples: Number of samples for averaging
        positions: List of position indices to extract
        
    Returns:
        Hook function
    """
    def hook_fn(module, input):
        activation = input[0].clone().to(cache)
        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)
    return hook_fn

def get_mean_activations(model, tokenizer, instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=32, positions=[-1]):
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store the mean activations in high-precision to avoid numerical issues
    mean_activations = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64, device=model.device)

    fwd_pre_hooks = [(block_modules[layer], get_mean_activations_pre_hook(layer=layer, cache=mean_activations, n_samples=n_samples, positions=positions)) for layer in range(n_layers)]

    with torch.no_grad():
        for i in tqdm(range(0, len(instructions), batch_size)):
            #print(f"Processing mean activations for batch {i // batch_size} of {len(instructions) // batch_size} (batch_size={batch_size}, len(instructions)={len(instructions)})...")
            inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
                model(
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                )

    return mean_activations

def get_mean_diff(
    model, 
    tokenizer, 
    harmful_instructions, 
    harmless_instructions, 
    tokenize_instructions_fn, 
    block_modules: List[torch.nn.Module], 
    batch_size=32, 
    positions=[-1]
):
    """
    Calculate the difference in mean activations between harmful and harmless instructions.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        harmful_instructions: List of harmful instructions
        harmless_instructions: List of harmless instructions
        tokenize_instructions_fn: Function to tokenize instructions
        block_modules: List of model block modules
        batch_size: Batch size for processing
        positions: List of token positions to extract
        
    Returns:
        Mean difference tensor of shape (n_positions, n_layers, d_model)
    """
    print("Calculating activations for harmful instructions...")
    mean_activations_harmful = get_mean_activations(
        model, 
        tokenizer, 
        harmful_instructions, 
        tokenize_instructions_fn, 
        block_modules, 
        batch_size=batch_size, 
        positions=positions
    )
    
    print("Calculating activations for harmless instructions...")
    mean_activations_harmless = get_mean_activations(
        model, 
        tokenizer, 
        harmless_instructions, 
        tokenize_instructions_fn, 
        block_modules, 
        batch_size=batch_size, 
        positions=positions
    )

    mean_diff = mean_activations_harmful - mean_activations_harmless

    return mean_diff

def generate_directions(model_base: ModelBase, harmful_instructions, harmless_instructions, artifact_dir, batch_size=32):
    """
    Generate and save refusal directions.
    
    Args:
        model_base: ModelBase instance
        harmful_instructions: List of harmful instructions
        harmless_instructions: List of harmless instructions
        artifact_dir: Directory to save artifacts
        
    Returns:
        Mean difference tensor
    """
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    # Use only the last token of the user turn
    positions = [-1]
    
    mean_diffs = get_mean_diff(
        model_base.model, 
        model_base.tokenizer, 
        harmful_instructions, 
        harmless_instructions, 
        model_base.tokenize_instructions_fn, 
        model_base.model_block_modules, 
        positions=positions,
        batch_size=batch_size
    )

    assert mean_diffs.shape == (1, model_base.model.config.num_hidden_layers, model_base.model.config.hidden_size)
    assert not mean_diffs.isnan().any()

    torch.save(mean_diffs, f"{artifact_dir}/mean_diffs.pt")
    
    return mean_diffs

def get_ablation_hook(mean_diff: torch.Tensor, ablation_strength: float = 1.0, harmful_indices: List[int] = None):
    """
    Create a hook function that ablates (removes) the refusal direction with probability p(RFA).
    
    Args:
        mean_diff: The refusal direction to ablate
        ablation_strength: Strength of the ablation (1.0 = full ablation)
        harmful_indices: List of indices in the batch to apply ablation to (if None, apply to all)
        
    Returns:
        Hook function
    """
    def hook_fn(module, input_tensors):
        # Clone the input to avoid in-place modification
        input_tensor = input_tensors[0].clone()
        
        # Get the positions to modify (last token position)
        position_idx = -1
        
        # Project out the refusal direction
        # For each sample in the batch
        batch_indices = harmful_indices if harmful_indices is not None else range(input_tensor.shape[0])
        
        for i in batch_indices:
            if i >= input_tensor.shape[0]:
                continue  # Skip if index is out of bounds
                
            # Calculate the projection of the activation onto the refusal direction
            activation = input_tensor[i, position_idx, :]
            
            # Normalize the refusal direction to get unit vector
            # Ensure mean_diff has the correct shape by squeezing any extra dimensions
            mean_diff_norm = mean_diff.squeeze() / torch.norm(mean_diff.squeeze())
            
            # Calculate the projection: (r^T h) where r is the unit refusal direction
            projection = torch.sum(activation * mean_diff_norm)
            
            # Subtract the projection from the activation (ablating the harmful component)
            # According to the paper: h(x) <- h(x) - r r^T h(x)
            input_tensor[i, position_idx, :] -= ablation_strength * projection * mean_diff_norm
        
        # Return a tuple with the modified tensor
        return (input_tensor,) + input_tensors[1:]
    
    return hook_fn 