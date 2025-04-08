import random
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from datasets import load_dataset

def sample_examples_from_datasets(
    datasets, proportions, total_examples=1000, only_prompts=False
):
    """
    Sample examples from multiple datasets with specific proportions.
    
    Args:
        datasets: List of datasets to sample from
        proportions: List of proportions for each dataset (should sum to 1)
        total_examples: Total number of examples to sample
        only_prompts: If True, return only prompts (without completions)
        
    Returns:
        List of sampled examples
    """
    # Ensure the proportions sum to 1
    if len(datasets) != len(proportions):
        raise ValueError("Number of datasets must match number of proportions")

    if abs(sum(proportions) - 1) > 1e-6:
        raise ValueError("Proportions must sum to 1")

    examples = []
    np.random.seed(42)
    for dataset, proportion in zip(datasets, proportions):
        n_samples = int(total_examples * proportion)

        # Ensure we don't try to sample more examples than available
        sampled_indices = np.random.choice(len(dataset), size=n_samples, replace=True)
        sampled = dataset.select(sampled_indices)

        if only_prompts:
            examples.extend([item["prompt"] for item in sampled])
        else:
            examples.extend(
                [f"{item['prompt']} {item['completion']}" for item in sampled]
            )

    # Shuffle the final list to mix examples from different datasets
    random.Random(42).shuffle(examples)

    return examples

def load_jailbreak_datasets():
    """
    Load and prepare the jailbreak datasets according to the Refat paper.
    
    Returns:
        Dictionary with harmful and harmless examples
    """
    # Load the dataset from Hugging Face
    jailbreaks_dataset = load_dataset(
        "Mechanistic-Anomaly-Detection/gemma2-jailbreaks"
    )

    # Dreject: 5000 samples from harmful Circuit Breakers dataset
    harmful_examples = sample_examples_from_datasets(
        [jailbreaks_dataset["circuit_breakers_train"]], [1.0],
        total_examples=5000
    )

    # Dutility: 5000 samples from harmless Circuit Breakers dataset + 150 from XSTest
    harmless_circuit_breakers = sample_examples_from_datasets(
        [jailbreaks_dataset["benign_instructions_train"]], [1.0],
        total_examples=5000
    )
    
    harmless_xstest = sample_examples_from_datasets(
        [jailbreaks_dataset["xstest"]], [1.0],
        total_examples=150
    )
    
    harmless_examples = harmless_circuit_breakers + harmless_xstest

    # Also get examples with just the prompts
    harmful_examples_prompts = sample_examples_from_datasets(
        [jailbreaks_dataset["circuit_breakers_train"]], [1.0],
        total_examples=5000, only_prompts=True
    )

    harmless_examples_prompts = sample_examples_from_datasets(
        [jailbreaks_dataset["benign_instructions_train"], jailbreaks_dataset["xstest"]],
        [0.97, 0.03],  # 5000/(5000+150) ≈ 0.97, 150/(5000+150) ≈ 0.03
        total_examples=5150, only_prompts=True
    )

    return {
        "harmful_train": harmful_examples,
        "harmless_train": harmless_examples,
        "harmful_train_prompts": harmful_examples_prompts,
        "harmless_train_prompts": harmless_examples_prompts,
    } 