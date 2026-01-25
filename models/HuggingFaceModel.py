"""
HuggingFace Model Wrapper for MiCo Framework

This module provides wrappers for loading and using pretrained HuggingFace 
models (particularly decoder-only LLMs) within the MiCo mixed precision 
quantization framework.

Supported pre-defined models include edge-focused LLMs around 1B size and below:
- TinyLlama-1.1B
- Qwen2-0.5B
- SmolLM-135M/360M/1.7B
- GPT2/GPT2-Medium
- OPT-125M/350M

Additional models can be loaded by passing any HuggingFace model identifier
to HuggingFaceModel.from_pretrained().
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from MiCoModel import MiCoModel

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoConfig,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class HFModelArgs:
    """Configuration arguments for HuggingFace model loading."""
    model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    max_seq_len: int = 512
    trust_remote_code: bool = True
    torch_dtype: torch.dtype = torch.float32
    low_cpu_mem_usage: bool = True
    device_map: Optional[str] = None  # None, "auto", "cpu", or specific device


class HuggingFaceModel(MiCoModel):
    """
    Wrapper for HuggingFace pretrained causal language models.
    
    This class wraps pretrained LLMs from HuggingFace Hub and provides
    integration with the MiCo quantization framework for layer-wise
    precision search.
    
    Note: This wrapper is designed for inference/testing only, not training.
    The models can be used for perplexity evaluation and layer-wise quantization.
    
    Example:
        >>> model = HuggingFaceModel.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
        >>> model.set_qscheme([[8]*model.n_layers, [8]*model.n_layers])
        >>> result = model.test(test_loader)
    """
    last_loss: Optional[torch.Tensor]
    
    def __init__(self, hf_model: nn.Module, tokenizer, args: HFModelArgs):
        super().__init__()
        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace transformers is required. "
                "Install with: pip install transformers"
            )
        
        self.model = hf_model
        self.tokenizer = tokenizer
        self.params = args
        self.vocab_size = tokenizer.vocab_size
        
        # For compatibility with LLaMa training interface
        self.train_loader = None
        self.test_loader = None
        self.last_loss = None
        
        # Count quantizable layers
        self.n_layers = len(self.get_qlayers())
        
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        max_seq_len: int = 512,
        trust_remote_code: bool = True,
        torch_dtype: torch.dtype = torch.float32,
        low_cpu_mem_usage: bool = True,
        device_map: Optional[str] = None,
    ) -> "HuggingFaceModel":
        """
        Load a pretrained model from HuggingFace Hub.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
            max_seq_len: Maximum sequence length for generation
            trust_remote_code: Whether to trust remote code for custom models
            torch_dtype: Torch dtype for model weights
            low_cpu_mem_usage: Use low CPU memory mode for loading
            device_map: Device map for model sharding (None, "auto", "cpu", etc.)
            
        Returns:
            HuggingFaceModel instance
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace transformers is required. "
                "Install with: pip install transformers"
            )
        
        args = HFModelArgs(
            model_name=model_name,
            max_seq_len=max_seq_len,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
        )
        
        return cls(hf_model, tokenizer, args)
    
    def forward(
        self, 
        tokens: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            tokens: Input token IDs [batch_size, seq_len]
            targets: Optional target token IDs for loss calculation
            attention_mask: Optional attention mask
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
        """
        outputs = self.model(
            input_ids=tokens,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits
        
        if targets is not None:
            # Shift logits and targets for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            self.last_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,  # Ignore padding tokens
            )
        else:
            self.last_loss = None
            
        return logits
    
    @torch.no_grad()
    def estimate_loss(self, eval_iters: int = 100) -> Dict[str, float]:
        """
        Estimate loss on train and test sets.
        
        Args:
            eval_iters: Number of iterations to evaluate
            
        Returns:
            Dictionary with TrainLoss, TestLoss, TestAcc
        """
        out = {}
        self.eval()
        
        # Test Loss Estimate
        losses = torch.zeros(eval_iters)
        test_correct = 0
        actual_iters = 0
        
        for k in range(eval_iters):
            try:
                X, Y = next(self.test_loader)
                actual_iters += 1
            except (StopIteration, TypeError):
                # Iterator exhausted or not an iterator, try to reset
                try:
                    self.test_loader = iter(self._test_loader_source)
                    X, Y = next(self.test_loader)
                    actual_iters += 1
                except (StopIteration, TypeError, AttributeError):
                    break
                
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)
            if not isinstance(Y, torch.Tensor):
                Y = torch.tensor(Y)
                
            X = X.to(device)
            Y = Y.to(device)
            
            logits = self(X, Y)
            loss = self.last_loss
            losses[k] = loss.item() if loss is not None else 0
            
            # Calculate accuracy (next token prediction)
            predict = torch.max(logits[:, :-1, :].view(-1, logits.size(-1)), dim=1).indices
            correct = Y[:, 1:].contiguous().view(-1)
            mask = correct != -100  # Ignore padding
            if mask.sum() > 0:
                test_correct += (predict[mask] == correct[mask]).sum().item() / mask.sum().item()
                
        out["TrainLoss"] = losses[:actual_iters].mean().item() if actual_iters > 0 else 0
        out["TestLoss"] = losses[:actual_iters].mean().item() if actual_iters > 0 else 0
        out["TestAcc"] = test_correct / actual_iters if actual_iters > 0 else 0
        
        self.train()
        return out
    
    def test(
        self, 
        test_loader, 
        train_loader=None,
        eval_iters: int = 100,
    ) -> Dict[str, float]:
        """
        Test the model on given data loader.
        
        Args:
            test_loader: Test data loader (yields (input_ids, labels) tuples)
            train_loader: Optional train loader (not used, for API compatibility)
            eval_iters: Number of evaluation iterations
            
        Returns:
            Dictionary with TestLoss, TestAcc
        """
        # Store loader source for potential reset
        self._test_loader_source = test_loader
        self._train_loader_source = test_loader if train_loader is None else train_loader
        self.test_loader = iter(test_loader)
        self.train_loader = iter(test_loader) if train_loader is None else iter(train_loader)
        return self.estimate_loss(eval_iters)
    
    def train_loop(self, *args, **kwargs):
        """
        Training is not supported for HuggingFace models in this wrapper.
        Use the original HuggingFace training pipeline instead.
        """
        raise NotImplementedError(
            "Training is not supported for HuggingFace models in this wrapper. "
            "Use the original HuggingFace training pipeline or fine-tuning tools."
        )
    
    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate text tokens autoregressively.
        
        Args:
            idx: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        # Use HuggingFace generate for simplicity
        generated = self.model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_k=top_k,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return generated
    
    def get_model_size(self) -> int:
        """Get total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_mb(self) -> float:
        """Get model size in megabytes (assuming float32)."""
        return self.get_model_size() * 4 / (1024 * 1024)


# Convenience functions for specific edge-focused models

def TinyLlama1B(max_seq_len: int = 512, torch_dtype=torch.float32) -> HuggingFaceModel:
    """
    Load TinyLlama 1.1B model.
    
    TinyLlama is a compact 1.1B parameter model trained on 3T tokens,
    making it efficient for edge deployment while maintaining good performance.
    """
    return HuggingFaceModel.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        max_seq_len=max_seq_len,
        torch_dtype=torch_dtype,
    )


def Qwen2_0_5B(max_seq_len: int = 512, torch_dtype=torch.float32) -> HuggingFaceModel:
    """
    Load Qwen2 0.5B model.
    
    Qwen2-0.5B is a small but capable model from Alibaba's Qwen series,
    suitable for edge deployment scenarios.
    """
    return HuggingFaceModel.from_pretrained(
        "Qwen/Qwen2-0.5B",
        max_seq_len=max_seq_len,
        torch_dtype=torch_dtype,
    )


def SmolLM_135M(max_seq_len: int = 512, torch_dtype=torch.float32) -> HuggingFaceModel:
    """
    Load SmolLM 135M model.
    
    SmolLM-135M is an extremely small model from HuggingFace,
    designed for on-device applications with very limited resources.
    """
    return HuggingFaceModel.from_pretrained(
        "HuggingFaceTB/SmolLM-135M",
        max_seq_len=max_seq_len,
        torch_dtype=torch_dtype,
    )


def SmolLM_360M(max_seq_len: int = 512, torch_dtype=torch.float32) -> HuggingFaceModel:
    """
    Load SmolLM 360M model.
    
    SmolLM-360M is a small model from HuggingFace,
    designed for on-device applications with limited resources.
    """
    return HuggingFaceModel.from_pretrained(
        "HuggingFaceTB/SmolLM-360M",
        max_seq_len=max_seq_len,
        torch_dtype=torch_dtype,
    )


def SmolLM_1_7B(max_seq_len: int = 512, torch_dtype=torch.float32) -> HuggingFaceModel:
    """
    Load SmolLM 1.7B model.
    
    SmolLM-1.7B is a medium-sized model from HuggingFace,
    balancing performance and efficiency for edge applications.
    """
    return HuggingFaceModel.from_pretrained(
        "HuggingFaceTB/SmolLM-1.7B",
        max_seq_len=max_seq_len,
        torch_dtype=torch_dtype,
    )


def GPT2_Small(max_seq_len: int = 512, torch_dtype=torch.float32) -> HuggingFaceModel:
    """
    Load GPT-2 Small (124M) model.
    
    GPT-2 Small is a classic small language model,
    useful for testing and baseline comparisons.
    """
    return HuggingFaceModel.from_pretrained(
        "gpt2",
        max_seq_len=max_seq_len,
        torch_dtype=torch_dtype,
    )


def GPT2_Medium(max_seq_len: int = 512, torch_dtype=torch.float32) -> HuggingFaceModel:
    """
    Load GPT-2 Medium (355M) model.
    
    GPT-2 Medium provides a good balance between size and capability.
    """
    return HuggingFaceModel.from_pretrained(
        "gpt2-medium",
        max_seq_len=max_seq_len,
        torch_dtype=torch_dtype,
    )


def OPT_125M(max_seq_len: int = 512, torch_dtype=torch.float32) -> HuggingFaceModel:
    """
    Load OPT-125M model.
    
    OPT-125M from Meta is a small decoder-only transformer,
    designed for efficient inference.
    """
    return HuggingFaceModel.from_pretrained(
        "facebook/opt-125m",
        max_seq_len=max_seq_len,
        torch_dtype=torch_dtype,
    )


def OPT_350M(max_seq_len: int = 512, torch_dtype=torch.float32) -> HuggingFaceModel:
    """
    Load OPT-350M model.
    
    OPT-350M from Meta provides more capability while remaining edge-friendly.
    """
    return HuggingFaceModel.from_pretrained(
        "facebook/opt-350m",
        max_seq_len=max_seq_len,
        torch_dtype=torch_dtype,
    )


# Model registry for easy access
HF_MODEL_REGISTRY: Dict[str, callable] = {
    "tinyllama-1.1b": TinyLlama1B,
    "qwen2-0.5b": Qwen2_0_5B,
    "smollm-135m": SmolLM_135M,
    "smollm-360m": SmolLM_360M,
    "smollm-1.7b": SmolLM_1_7B,
    "gpt2": GPT2_Small,
    "gpt2-medium": GPT2_Medium,
    "opt-125m": OPT_125M,
    "opt-350m": OPT_350M,
}


def list_available_models() -> List[str]:
    """List all available pre-defined HuggingFace models."""
    return list(HF_MODEL_REGISTRY.keys())


def load_model(name: str, **kwargs) -> HuggingFaceModel:
    """
    Load a pre-defined HuggingFace model by name.
    
    Args:
        name: Model name (see list_available_models())
        **kwargs: Additional arguments passed to the model loader
        
    Returns:
        HuggingFaceModel instance
    """
    if name not in HF_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. "
            f"Available models: {list_available_models()}"
        )
    return HF_MODEL_REGISTRY[name](**kwargs)
