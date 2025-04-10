import time
import torch
import random
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
import os
import psutil
import gc

from refat.model_utils.model_base import ModelBase, ModelConfig
from refat.data_utils.dataset_utils import load_jailbreak_datasets
from refat.pipeline.refusal_direction import generate_directions, get_ablation_hook
from refat.pipeline.utils.hook_utils import add_hooks

class RefatConfig:
    def __init__(
        self,
        model_path: str,
        artifact_dir: str,
        refusal_layers: Union[str, List[int]] = "last_75_percent",
        steps_between_recalc: int = 4,
        samples_per_recalc: int = 32,
        mean_diff_batch_size: int = 1, # Batch size for mean difference calculation
        training_batch_size: int = 1, # Batch size for model training
        gradient_accumulation_steps: int = 4, # Number of steps to accumulate gradients
        torch_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        rfa_probability: float = 0.5,  # Probability of applying RFA to harmful samples
        use_gradient_checkpointing: bool = False,  # Whether to use gradient checkpointing
        use_amp: bool = True,  # Whether to use automatic mixed precision
        debug: bool = True,  # Whether to print debug information
        # LoRA parameters
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16, 
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        learning_rate: float = 1e-4
    ):
        self.model_path = model_path
        self.artifact_dir = artifact_dir
        self.refusal_layers = refusal_layers
        self.steps_between_recalc = steps_between_recalc
        self.samples_per_recalc = samples_per_recalc
        self.mean_diff_batch_size = mean_diff_batch_size
        self.training_batch_size = training_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.torch_dtype = torch_dtype
        self.device = device
        self.rfa_probability = rfa_probability
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_amp = use_amp
        self.debug = debug
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.learning_rate = learning_rate

class Refat:
    def __init__(self, config: RefatConfig):
        self.config = config
        
        # Print memory usage at initialization
        if self.config.debug:
            self._print_memory_usage("Initialization")
        
        # Initialize model
        model_config = ModelConfig(
            model_path=config.model_path,
            torch_dtype=config.torch_dtype,
            device=config.device,
            use_lora=config.use_lora,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_target_modules=config.lora_target_modules
        )
        self.model_base = ModelBase(model_config)
        
        if self.config.debug:
            print(f"Model loaded from {config.model_path}")
            print(f"Model device: {self.model_base.model.device}")
            print(f"Model dtype: {self.model_base.model.dtype}")
            print(f"Model parameters: {sum(p.numel() for p in self.model_base.model.parameters())}")
            self._print_memory_usage("After model loading")
        
        # Enable gradient checkpointing if specified
        if config.use_gradient_checkpointing:
            self.model_base.model.gradient_checkpointing_enable()
            if self.config.debug:
                print("Gradient checkpointing enabled")
            
        # Load datasets
        self.datasets = load_jailbreak_datasets()
        
        if self.config.debug:
            print(f"Datasets loaded: {list(self.datasets.keys())}")
            for key, dataset in self.datasets.items():
                print(f"  - {key}: {len(dataset)} samples")
            self._print_memory_usage("After dataset loading")
        
        # Initialize refusal directions
        self.refusal_directions = None
        self._initialize_refusal_directions()
        
    def _print_memory_usage(self, stage: str):
        """Print memory usage information."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        # Get GPU memory info if available
        gpu_mem = "N/A"
        if torch.cuda.is_available():
            gpu_mem = f"{torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, {torch.cuda.memory_reserved() / 1024**2:.2f}MB reserved"
        
        print(f"Memory usage at {stage}:")
        print(f"  - RSS: {mem_info.rss / 1024**2:.2f}MB")
        print(f"  - VMS: {mem_info.vms / 1024**2:.2f}MB")
        print(f"  - GPU: {gpu_mem}")
        
    def _initialize_refusal_directions(self):
        """Initialize or load refusal directions."""
        if os.path.exists(f"{self.config.artifact_dir}/mean_diffs.pt"):
            if self.config.debug:
                print(f"Loading refusal directions from {self.config.artifact_dir}/mean_diffs.pt")
            self.refusal_directions = torch.load(f"{self.config.artifact_dir}/mean_diffs.pt")
            if self.config.debug:
                print(f"Refusal directions loaded, shape: {self.refusal_directions.shape}")
                self._print_memory_usage("After loading refusal directions")
        else:
            if self.config.debug:
                print("No existing refusal directions found, calculating new ones")
            self._recalculate_directions()

    def _get_refusal_layers(self) -> List[int]:
        """Get the layers to apply refusal direction ablation to."""
        n_layers = self.model_base.model.config.num_hidden_layers
        
        if isinstance(self.config.refusal_layers, str):
            if self.config.refusal_layers == "last_75_percent":
                start_layer = int(n_layers * 0.25)
                layers = list(range(start_layer, n_layers))
            elif self.config.refusal_layers == "last_50_percent":
                start_layer = int(n_layers * 0.5)
                layers = list(range(start_layer, n_layers))
            else:
                raise ValueError(f"Unknown refusal_layers value: {self.config.refusal_layers}")
        else:
            layers = self.config.refusal_layers
            
        if self.config.debug:
            print(f"Using refusal layers: {layers} (out of {n_layers} total layers)")
            
        return layers
    
    def _should_recalculate_directions(self, step: int) -> bool:
        """Check if we should recalculate refusal directions."""
        should_recalc = step % self.config.steps_between_recalc == 0
        if should_recalc and self.config.debug:
            print(f"Step {step}: Recalculating refusal directions")
        return should_recalc
    
    def _recalculate_directions(self):
        """Recalculate refusal directions using random samples.
        
        According to the paper, we use n=32 samples from each dataset.
        """
        if self.config.debug:
            print("Recalculating refusal directions...")
            self._print_memory_usage("Before direction calculation")
        
        # Sample from datasets
        harmful_samples = random.sample(
            self.datasets["harmful_train"],
            self.config.samples_per_recalc
        )
        harmless_samples = random.sample(
            self.datasets["harmless_train"],
            self.config.samples_per_recalc
        )
        
        if self.config.debug:
            print(f"Sampled {self.config.samples_per_recalc} harmful and {self.config.samples_per_recalc} harmless samples")
        
        # Generate new directions
        self.refusal_directions = generate_directions(
            self.model_base,
            harmful_samples,
            harmless_samples,
            self.config.artifact_dir,
            batch_size=self.config.mean_diff_batch_size # Batch size for mean diff calculation
        )
        
        if self.config.debug:
            print(f"Recalculated refusal directions using {self.config.samples_per_recalc} samples from each dataset")
            print(f"Refusal directions shape: {self.refusal_directions.shape}")
            self._print_memory_usage("After direction calculation")
    
    def train_step(self, batch: List[str], step: int, harmful_indices: List[int]):
        """Perform a single training step.
        
        Args:
            batch: List of input texts
            step: Current training step
            harmful_indices: Indices of harmful samples in the batch
            
        Returns:
            Tuple of (model outputs, input_ids)
        """
        if self.config.debug:
            print(f"Step {step}: Processing mixed batch of size {len(batch)} ({len(harmful_indices)} harmful)")
            #self._print_memory_usage(f"Step {step} start")
                
        # Get refusal layers
        refusal_layers = self._get_refusal_layers()
        
        # Create ablation hooks for refusal layers
        # Only apply ablation to harmful samples with probability p(RFA)
        ablation_hooks = []
        
        if harmful_indices and random.random() < self.config.rfa_probability:
            ablation_strength = 1.0  # Full ablation
            
            if self.config.debug:
                print(f"Step {step}: Applying RFA with strength {ablation_strength} to harmful samples")
                
            ablation_hooks = [
                (self.model_base.model_block_modules[layer], 
                 get_ablation_hook(
                     self.refusal_directions[:, layer, :],
                     ablation_strength,
                     harmful_indices=harmful_indices
                 ))
                for layer in refusal_layers
            ]
        
        # Process batch with ablation hooks
        inputs = self.model_base.tokenize(batch, max_length=512)
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        if self.config.debug:
            print(f"Tokenized into input shape: {inputs['input_ids'].shape}")
        
        with add_hooks(module_forward_pre_hooks=ablation_hooks, module_forward_hooks=[]):
            # Use torch.cuda.amp for mixed precision training if using cuda
            if self.config.use_amp and torch.cuda.is_available() and self.config.torch_dtype in [torch.float16, torch.bfloat16]:
                with torch.cuda.amp.autocast():
                    outputs = self.model_base.model(**inputs)
            else:
                outputs = self.model_base.model(**inputs)
        
        #if self.config.debug and step % 10 == 0:
        #    print(f"Step {step}: Output logits shape: {outputs.logits.shape}")
        #    self._print_memory_usage(f"Step {step} end")
        
        # Only keep the necessary output for loss calculation
        input_ids = inputs["input_ids"].detach()
        
        # Clear hooks and inputs to free memory
        for module, hook in ablation_hooks:
            if hasattr(module, '_forward_pre_hooks'):
                # Remove the hook if it exists
                for hook_id, hook_fn in list(module._forward_pre_hooks.items()):
                    if hook_fn == hook:
                        module._forward_pre_hooks.pop(hook_id, None)
        
        del inputs, ablation_hooks
        
        return outputs, input_ids
    
    def train(self):
        """Train the model using the Refat algorithm, for one epoch.
        """

        num_steps = min(len(self.datasets["harmful_train"]), len(self.datasets["harmless_train"])) // (self.config.training_batch_size//2)

        if self.config.debug:
            print(f"Starting training for {num_steps} steps")
            self._print_memory_usage("Before training")
        
        # Initialize optimizer to only train LoRA parameters
        optimizer = torch.optim.AdamW(
            [p for p in self.model_base.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate
        )
        
        # Initialize gradient scaler for AMP
        scaler = torch.cuda.amp.GradScaler() if self.config.use_amp and torch.cuda.is_available() else None
        
        if self.config.debug:
            print(f"Using AMP: {self.config.use_amp and torch.cuda.is_available()}")
        
        # Shuffle datasets
        harmful_samples = self.datasets["harmful_train"] # Could .copy() but disabling to save memory
        harmless_samples = self.datasets["harmless_train"]
        random.shuffle(harmful_samples)
        random.shuffle(harmless_samples)
        
        if self.config.debug:
            print(f"Shuffled datasets: {len(harmful_samples)} harmful, {len(harmless_samples)} harmless samples")
        
        half_batch_size = self.config.training_batch_size // 2
        
        # Add accumulation steps parameter
        accumulation_steps = self.config.gradient_accumulation_steps  # Use the parameter from config
        actual_batch_size = max(1, self.config.training_batch_size // accumulation_steps) # bs 32 -> 32 chunks -> abs 1
        
        for step in tqdm(range(num_steps)):
            # Check if we need to recalculate directions
            if self._should_recalculate_directions(step):
                self._recalculate_directions()

            # Sample half batch from each dataset
            harmful_batch = harmful_samples[step * half_batch_size:(step + 1) * half_batch_size]
            harmless_batch = harmless_samples[step * half_batch_size:(step + 1) * half_batch_size]
            
            # Combine batches
            combined_batch = harmful_batch + harmless_batch
            
            # Reset gradients at the beginning of each large batch
            optimizer.zero_grad()
            total_loss = 0.0
            
            # Process in smaller chunks
            for chunk_idx in range(0, len(combined_batch), actual_batch_size):
                chunk_end = min(chunk_idx + actual_batch_size, len(combined_batch))
                batch_chunk = combined_batch[chunk_idx:chunk_end]
                
                # Track harmful indices within this chunk
                harmful_indices = [i for i in range(min(half_batch_size - chunk_idx, actual_batch_size)) 
                                  if chunk_idx + i < half_batch_size]
                
                if self.config.debug:
                    print(f"Step {step} time {time.strftime('%H:%M:%S')}, starting forward Chunk {chunk_idx//actual_batch_size}: Processing {len(batch_chunk)} samples ({len(harmful_indices)} harmful)")
                    print(f"Harmful indices: {harmful_indices}")
                    print(f"Batch chunk (sizes {[len(x) for x in batch_chunk]}): {batch_chunk}")
                
                # Skip empty chunks
                if not batch_chunk:
                    continue
                    
                # Perform forward pass on chunk
                outputs, input_ids = self.train_step(batch_chunk, step, harmful_indices)
                if self.config.debug:
                    print(f"Outputs shape: {outputs.logits.shape}")
                    print(f"Input IDs shape: {input_ids.shape}")
                
                # Calculate loss
                logits = outputs.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                
                if self.config.debug:
                    print(f"Step {step} time {time.strftime('%H:%M:%S')}, starting backward")

                # Handle loss calculation and backward pass with or without AMP
                if scaler:
                    with torch.cuda.amp.autocast():
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        scaled_loss = loss / accumulation_steps
                    
                    # Use scaler for backward
                    scaler.scale(scaled_loss).backward()
                else:
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    scaled_loss = loss / accumulation_steps
                    scaled_loss.backward()
                
                total_loss += loss.item()
                
                # Free up memory
                del outputs, logits, shift_logits, shift_labels, loss, scaled_loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Clip gradients and update parameters once per effective batch
            if scaler:
                # Unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in self.model_base.model.parameters() if p.requires_grad], 1.0)
                
                # Update weights with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_([p for p in self.model_base.model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
            
            # Report average loss
            avg_loss = total_loss / (len(combined_batch) / actual_batch_size)
            
            if step % 100 == 0 or (self.config.debug):
                print(f"Step {step}, Avg Loss: {avg_loss:.4f}")
                if self.config.debug and step % 100 == 0:
                    self._print_memory_usage(f"Step {step}, before forced gc")
                    # Force garbage collection
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self._print_memory_usage(f"Step {step}, after forced gc")
    
    def generate_with_ablation(self, prompt: str, max_length: int = 512) -> str:
        """Generate text with refusal direction ablation.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            
        Returns:
            Generated text
        """
        if self.config.debug:
            print(f"Generating with ablation for prompt: {prompt[:50]}...")
            self._print_memory_usage("Before generation")
        
        # Force garbage collection before generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        refusal_layers = self._get_refusal_layers()
        
        # Create ablation hooks with full ablation strength
        ablation_hooks = [
            (self.model_base.model_block_modules[layer], 
             get_ablation_hook(
                 self.refusal_directions[:, layer, :],
                 ablation_strength=1.0,  # Full ablation
                 harmful_indices=None  # Apply to all indices
             ))
            for layer in refusal_layers
        ]
        
        # Tokenize input
        inputs = self.model_base.tokenize([prompt])
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        if self.config.debug:
            print(f"Input shape: {inputs['input_ids'].shape}")
        
        # Generate with ablation hooks
        with add_hooks(module_forward_pre_hooks=ablation_hooks, module_forward_hooks=[]):
            with torch.no_grad():
                # Use AMP for generation if enabled
                if self.config.use_amp and torch.cuda.is_available() and self.config.torch_dtype in [torch.float16, torch.bfloat16]:
                    with torch.cuda.amp.autocast():
                        outputs = self.model_base.model.generate(
                            **inputs,
                            max_length=max_length,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9
                        )
                else:
                    outputs = self.model_base.model.generate(
                        **inputs,
                        max_length=max_length,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
        
        if self.config.debug:
            print(f"Generated output length: {len(outputs[0])}")
            self._print_memory_usage("After generation")
        
        result = self.model_base.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up hooks
        for module, hook in ablation_hooks:
            if hasattr(module, '_forward_pre_hooks'):
                for hook_id, hook_fn in list(module._forward_pre_hooks.items()):
                    if hook_fn == hook:
                        module._forward_pre_hooks.pop(hook_id, None)
        
        # Force garbage collection after generation
        del inputs, outputs, ablation_hooks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return result 