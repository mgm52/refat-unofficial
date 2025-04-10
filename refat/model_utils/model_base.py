import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import os
from peft import get_peft_model, LoraConfig, TaskType

@dataclass
class ModelConfig:
    model_path: str
    torch_dtype: torch.dtype = torch.float16
    device: str = "cuda"
    eoi_toks: List[str] = None  # End of instruction tokens
    # LoRA parameters
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

class ModelBase:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_path = config.model_path
        self.torch_dtype = config.torch_dtype
        self.device = config.device
        
        # Default EOI tokens for Gemma 2 9B-IT model
        self.eoi_toks = config.eoi_toks or ["<end_of_turn>", "\n"]
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Apply LoRA if enabled
        if config.use_lora:
            self.apply_lora(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules
            )
        
        # Get model block modules for hooking
        self.model_block_modules = self._get_block_modules()
        
    def apply_lora(self, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=None):
        """Apply LoRA adapters to the model for memory-efficient training."""
        if target_modules is None:
            # Default target modules for Gemma
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )
        
        self.model = get_peft_model(self.model, peft_config)
        
        return self.model
    
    def _get_block_modules(self) -> List[torch.nn.Module]:
        """Get the transformer block modules for hooking."""
        blocks = []
        for name, module in self.model.named_modules():
            # For Gemma models, the transformer blocks are in model.layers
            if isinstance(module, torch.nn.Module) and hasattr(module, 'self_attn'):
                blocks.append(module)
        return blocks
    
    def tokenize(self, strings: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of instructions."""
        return self.tokenizer(
            strings,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def generate(self, prompt: str, max_length: int = 512, **kwargs) -> str:
        """Generate text from a prompt."""
        inputs = self.tokenize([prompt])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                **kwargs
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_logits(self, prompt: str) -> torch.Tensor:
        """Get logits for a prompt."""
        inputs = self.tokenize([prompt])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.logits 