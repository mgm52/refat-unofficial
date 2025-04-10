import torch
from refat.model_utils.model_base import ModelBase, ModelConfig
import argparse
import torch.nn.functional as F

def test_input_formats():
    """Test different input formats for the model."""
    
    # Initialize model
    config = ModelConfig(
        model_path="/workspace/gemma_2_9b_instruct",
        torch_dtype=torch.float16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model = ModelBase(config)
    
    prompt = "How to bake a cake?"
    response = "You need to mix the ingredients and bake it in the oven."
    test_inputs = [
        f"{prompt} {response}",
        f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}",
        f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}",
        f"{prompt}",
        model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            tokenize=False
        )
    ]

    for test_input in test_inputs:
        batch = [test_input]

        print(f"Testing input batch (len {len(batch)}): {batch}")

        inputs = model.tokenize(batch)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        print(f"\nTokenized to (shape {inputs['input_ids'].shape}): {inputs}")
        
        outputs = model.model(**inputs)

        print(f"\nOutput logits (shape {outputs.logits.shape})")
        predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
        print(f"Output token IDs (shape {predicted_token_ids.shape}): {predicted_token_ids}")
        predicted_tokens = model.tokenizer.batch_decode(predicted_token_ids)
        print(f"Output tokens (len {len(predicted_tokens[0])}): {predicted_tokens}")
        
        # Calculate cross-entropy loss
        if 'labels' in inputs:
            # If labels are provided in inputs
            loss = outputs.loss
            print(f"Cross-entropy loss from model output: {loss.item()}")
        else:
            # Calculate loss manually - shift the targets for teacher forcing
            # Use input_ids as target by shifting right
            shifted_logits = outputs.logits[..., :-1, :].contiguous()
            shifted_labels = inputs['input_ids'][..., 1:].contiguous()
            loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
            )
            print(f"Calculated cross-entropy loss: {loss.item()}")

        print(f"\n================================================\n")


if __name__ == "__main__":
    test_input_formats() 