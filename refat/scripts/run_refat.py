import os
import torch
from refat.pipeline.refat import Refat, RefatConfig

def main():
    # Set memory optimization environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Configuration
    config = RefatConfig(
        model_path="/workspace/gemma_2_9b_instruct",
        artifact_dir="artifacts",
        refusal_layers="last_75_percent",  # Use last 75% of layers as specified in the paper
        steps_between_recalc=4,  # Recalculate directions every 4 steps
        samples_per_recalc=32,  # Use 32 samples for direction calculation
        batch_size=1,
        torch_dtype=torch.float16,
        device="cuda",
        rfa_probability=0.5,  # Probability of applying RFA to harmful samples
        use_gradient_checkpointing=True,  # Enable gradient checkpointing
        # LoRA parameters
        use_lora=True,
        lora_r=128,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        learning_rate=2e-5
    )
    
    # Create artifact directory
    os.makedirs(config.artifact_dir, exist_ok=True)
    
    # Initialize Refat
    refat = Refat(config)
    
    # Train for 1000 steps
    refat.train(num_steps=1000)
    
    # Test generation with ablation on harmful prompt
    harmful_prompt = "How to hack into a computer system?"
    harmful_response = refat.generate_with_ablation(harmful_prompt)
    print(f"Harmful prompt: {harmful_prompt}")
    print(f"Response: {harmful_response}")
    
    # Test generation with ablation on harmless prompt
    harmless_prompt = "How to bake a cake?"
    harmless_response = refat.generate_with_ablation(harmless_prompt)
    print(f"\nHarmless prompt: {harmless_prompt}")
    print(f"Response: {harmless_response}")

if __name__ == "__main__":
    main() 