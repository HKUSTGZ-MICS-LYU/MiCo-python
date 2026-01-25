"""
LLM Evaluation Utilities for Mixed Precision Quantization

This module provides evaluation methods for comparing MPQ (Mixed Precision Quantized)
LLM models against their original full-precision (FP) counterparts.

Key evaluation methods:
1. Token-level agreement analysis (Top-K agreement, KL divergence)
2. Generation comparison (exact match, token overlap, BLEU)
3. Perplexity comparison

See doc/LLM_EVALUATION.md for detailed methodology.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TokenAgreementResult:
    """Results from token-level agreement analysis."""
    top1_agreement: float  # Fraction of positions with same top-1 prediction
    top5_agreement: float  # Fraction where FP's top-1 is in MPQ's top-5
    top10_agreement: float  # Fraction where FP's top-1 is in MPQ's top-10
    kl_divergence: float   # Average KL divergence between distributions
    rank_correlation: float  # Spearman correlation of token rankings
    n_samples: int


@dataclass
class GenerationResult:
    """Results from generation comparison."""
    exact_match: float      # Fraction of exactly matching outputs
    token_overlap: float    # Jaccard similarity of token sets
    prefix_match_len: float  # Average length of matching prefix
    n_samples: int
    fp_outputs: List[str]   # Generated text from FP model
    mpq_outputs: List[str]  # Generated text from MPQ model


@dataclass
class EvalResult:
    """Combined evaluation results."""
    perplexity_fp: float
    perplexity_mpq: float
    perplexity_ratio: float  # mpq_ppl / fp_ppl (lower is better)
    token_agreement: TokenAgreementResult
    generation: Optional[GenerationResult] = None


class LLMEvaluator:
    """
    Evaluator for comparing FP and MPQ LLM models.
    
    Example:
        >>> from MiCoLLMEval import LLMEvaluator
        >>> evaluator = LLMEvaluator(fp_model, mpq_model, tokenizer)
        >>> results = evaluator.quick_eval(test_loader, n_batches=50)
        >>> print(f"Top-1 Agreement: {results.top1_agreement:.2%}")
    """
    
    def __init__(
        self,
        fp_model: torch.nn.Module,
        mpq_model: torch.nn.Module,
        tokenizer=None,
    ):
        """
        Initialize evaluator with FP and MPQ models.
        
        Args:
            fp_model: Full-precision reference model
            mpq_model: Mixed-precision quantized model to evaluate
            tokenizer: Tokenizer for text encoding/decoding (optional, for generation)
        """
        self.fp_model = fp_model
        self.mpq_model = mpq_model
        self.tokenizer = tokenizer
        
        # Ensure both models are in eval mode
        self.fp_model.eval()
        self.mpq_model.eval()
    
    @torch.no_grad()
    def compute_token_agreement(
        self,
        fp_logits: torch.Tensor,
        mpq_logits: torch.Tensor,
        ignore_index: int = -100,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute token-level agreement between FP and MPQ model outputs.
        
        Args:
            fp_logits: Logits from FP model [batch, seq_len, vocab_size]
            mpq_logits: Logits from MPQ model [batch, seq_len, vocab_size]
            ignore_index: Index to ignore in labels (padding)
            labels: Optional labels to create mask [batch, seq_len]
            
        Returns:
            Dictionary with agreement metrics
        """
        # Flatten for easier computation
        batch_size, seq_len, vocab_size = fp_logits.shape
        
        # Create mask (all positions if no labels provided)
        if labels is not None:
            mask = (labels != ignore_index).reshape(-1)
        else:
            mask = torch.ones(batch_size * seq_len, dtype=torch.bool, device=fp_logits.device)
        
        fp_flat = fp_logits.reshape(-1, vocab_size)[mask]
        mpq_flat = mpq_logits.reshape(-1, vocab_size)[mask]
        
        if fp_flat.numel() == 0:
            return {
                "top1_agreement": 0.0,
                "top5_agreement": 0.0,
                "top10_agreement": 0.0,
                "kl_divergence": 0.0,
                "rank_correlation": 0.0,
            }
        
        # Top-1 agreement
        fp_top1 = fp_flat.argmax(dim=-1)
        mpq_top1 = mpq_flat.argmax(dim=-1)
        top1_agreement = (fp_top1 == mpq_top1).float().mean().item()
        
        # Top-5 agreement (FP's top-1 in MPQ's top-5)
        mpq_top5 = mpq_flat.topk(5, dim=-1).indices
        top5_agreement = (mpq_top5 == fp_top1.unsqueeze(-1)).any(dim=-1).float().mean().item()
        
        # Top-10 agreement
        mpq_top10 = mpq_flat.topk(10, dim=-1).indices
        top10_agreement = (mpq_top10 == fp_top1.unsqueeze(-1)).any(dim=-1).float().mean().item()
        
        # KL divergence (MPQ || FP)
        fp_probs = F.softmax(fp_flat, dim=-1)
        mpq_log_probs = F.log_softmax(mpq_flat, dim=-1)
        kl_div = F.kl_div(mpq_log_probs, fp_probs, reduction='batchmean').item()
        
        # Rank correlation (sample for efficiency)
        if fp_flat.size(0) > 100:
            sample_idx = torch.randperm(fp_flat.size(0))[:100]
            fp_sample = fp_flat[sample_idx]
            mpq_sample = mpq_flat[sample_idx]
        else:
            fp_sample = fp_flat
            mpq_sample = mpq_flat
        
        # Compute average rank correlation
        rank_corrs = []
        for i in range(fp_sample.size(0)):
            fp_ranks = fp_sample[i].argsort(descending=True).argsort().float()
            mpq_ranks = mpq_sample[i].argsort(descending=True).argsort().float()
            # Spearman correlation
            n = vocab_size
            d_sq = ((fp_ranks - mpq_ranks) ** 2).sum()
            rho = 1 - (6 * d_sq) / (n * (n**2 - 1))
            rank_corrs.append(rho.item())
        rank_correlation = np.mean(rank_corrs)
        
        return {
            "top1_agreement": top1_agreement,
            "top5_agreement": top5_agreement,
            "top10_agreement": top10_agreement,
            "kl_divergence": kl_div,
            "rank_correlation": rank_correlation,
        }
    
    @torch.no_grad()
    def quick_eval(
        self,
        data_loader,
        n_batches: int = 50,
    ) -> TokenAgreementResult:
        """
        Quick evaluation using token agreement metrics.
        
        This is fast enough to use during MPQ search iterations.
        
        Args:
            data_loader: DataLoader yielding (input_ids, labels) tuples
            n_batches: Number of batches to evaluate
            
        Returns:
            TokenAgreementResult with agreement metrics
        """
        self.fp_model.eval()
        self.mpq_model.eval()
        
        all_metrics = {
            "top1_agreement": [],
            "top5_agreement": [],
            "top10_agreement": [],
            "kl_divergence": [],
            "rank_correlation": [],
        }
        
        data_iter = iter(data_loader)
        for _ in tqdm(range(n_batches), desc="Quick eval", leave=False):
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            
            if isinstance(batch, (list, tuple)):
                input_ids, labels = batch[0], batch[1]
            else:
                input_ids = batch
                labels = None
            
            input_ids = input_ids.to(device)
            if labels is not None:
                labels = labels.to(device)
            
            # Get logits from both models
            fp_outputs = self.fp_model(input_ids)
            mpq_outputs = self.mpq_model(input_ids)
            
            # Handle different output formats
            if hasattr(fp_outputs, 'logits'):
                fp_logits = fp_outputs.logits
            elif isinstance(fp_outputs, torch.Tensor):
                fp_logits = fp_outputs
            else:
                fp_logits = fp_outputs[0]
                
            if hasattr(mpq_outputs, 'logits'):
                mpq_logits = mpq_outputs.logits
            elif isinstance(mpq_outputs, torch.Tensor):
                mpq_logits = mpq_outputs
            else:
                mpq_logits = mpq_outputs[0]
            
            # Compute metrics
            metrics = self.compute_token_agreement(fp_logits, mpq_logits, labels=labels)
            
            for key, value in metrics.items():
                all_metrics[key].append(value)
        
        n_samples = len(all_metrics["top1_agreement"])
        
        return TokenAgreementResult(
            top1_agreement=np.mean(all_metrics["top1_agreement"]),
            top5_agreement=np.mean(all_metrics["top5_agreement"]),
            top10_agreement=np.mean(all_metrics["top10_agreement"]),
            kl_divergence=np.mean(all_metrics["kl_divergence"]),
            rank_correlation=np.mean(all_metrics["rank_correlation"]),
            n_samples=n_samples,
        )
    
    @torch.no_grad()
    def generation_eval(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        temperature: float = 0.0,  # Greedy by default for reproducibility
    ) -> GenerationResult:
        """
        Compare generation outputs between FP and MPQ models.
        
        Args:
            prompts: List of text prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            
        Returns:
            GenerationResult with comparison metrics
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for generation evaluation")
        
        self.fp_model.eval()
        self.mpq_model.eval()
        
        fp_outputs = []
        mpq_outputs = []
        exact_matches = 0
        token_overlaps = []
        prefix_matches = []
        
        for prompt in tqdm(prompts, desc="Generation eval"):
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate with FP model
            if hasattr(self.fp_model, 'generate'):
                fp_generated = self.fp_model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                fp_generated = self.fp_model.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Generate with MPQ model
            if hasattr(self.mpq_model, 'generate'):
                mpq_generated = self.mpq_model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                mpq_generated = self.mpq_model.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode
            fp_text = self.tokenizer.decode(fp_generated[0], skip_special_tokens=True)
            mpq_text = self.tokenizer.decode(mpq_generated[0], skip_special_tokens=True)
            
            fp_outputs.append(fp_text)
            mpq_outputs.append(mpq_text)
            
            # Compute metrics
            # Exact match
            if fp_text == mpq_text:
                exact_matches += 1
            
            # Token overlap (Jaccard)
            fp_tokens = set(fp_generated[0].tolist())
            mpq_tokens = set(mpq_generated[0].tolist())
            intersection = len(fp_tokens & mpq_tokens)
            union = len(fp_tokens | mpq_tokens)
            token_overlaps.append(intersection / union if union > 0 else 0)
            
            # Prefix match length (after prompt)
            prompt_len = inputs.input_ids.size(1)
            fp_new = fp_generated[0][prompt_len:].tolist()
            mpq_new = mpq_generated[0][prompt_len:].tolist()
            prefix_len = 0
            for f, m in zip(fp_new, mpq_new):
                if f == m:
                    prefix_len += 1
                else:
                    break
            prefix_matches.append(prefix_len)
        
        return GenerationResult(
            exact_match=exact_matches / len(prompts),
            token_overlap=np.mean(token_overlaps),
            prefix_match_len=np.mean(prefix_matches),
            n_samples=len(prompts),
            fp_outputs=fp_outputs,
            mpq_outputs=mpq_outputs,
        )
    
    @torch.no_grad()
    def compute_perplexity(
        self,
        model: torch.nn.Module,
        data_loader,
        n_batches: int = 100,
    ) -> float:
        """
        Compute perplexity for a model on given data.
        
        Args:
            model: Model to evaluate
            data_loader: DataLoader yielding (input_ids, labels) tuples
            n_batches: Number of batches to evaluate
            
        Returns:
            Perplexity value
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        data_iter = iter(data_loader)
        for _ in range(n_batches):
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            
            if isinstance(batch, (list, tuple)):
                input_ids, labels = batch[0], batch[1]
            else:
                input_ids = batch
                labels = input_ids.clone()
            
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Get outputs
            outputs = model(input_ids)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                logits = outputs[0]
            
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='sum'
            )
            
            n_tokens = (shift_labels != -100).sum().item()
            total_loss += loss.item()
            total_tokens += n_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def full_eval(
        self,
        data_loader,
        prompts: Optional[List[str]] = None,
        n_batches: int = 100,
    ) -> EvalResult:
        """
        Comprehensive evaluation comparing FP and MPQ models.
        
        Args:
            data_loader: DataLoader for perplexity and token agreement
            prompts: Optional prompts for generation comparison
            n_batches: Number of batches for perplexity/agreement
            
        Returns:
            EvalResult with all metrics
        """
        # Perplexity
        ppl_fp = self.compute_perplexity(self.fp_model, data_loader, n_batches)
        ppl_mpq = self.compute_perplexity(self.mpq_model, data_loader, n_batches)
        
        # Token agreement
        token_agreement = self.quick_eval(data_loader, n_batches)
        
        # Generation (optional)
        generation = None
        if prompts is not None and self.tokenizer is not None:
            generation = self.generation_eval(prompts)
        
        return EvalResult(
            perplexity_fp=ppl_fp,
            perplexity_mpq=ppl_mpq,
            perplexity_ratio=ppl_mpq / ppl_fp if ppl_fp > 0 else float('inf'),
            token_agreement=token_agreement,
            generation=generation,
        )


# Default evaluation prompts for generation comparison
DEFAULT_PROMPTS = [
    "The capital of France is",
    "In machine learning, a neural network is",
    "The quick brown fox jumps over",
    "Once upon a time, there was",
    "The best way to learn programming is",
    "Artificial intelligence will",
    "The meaning of life is",
    "To solve this math problem, first",
    "In the year 2050, humans will",
    "The most important scientific discovery was",
]


def compare_models(
    fp_model: torch.nn.Module,
    mpq_model: torch.nn.Module,
    data_loader,
    tokenizer=None,
    n_batches: int = 50,
    prompts: Optional[List[str]] = None,
    verbose: bool = True,
) -> EvalResult:
    """
    Convenience function to compare FP and MPQ models.
    
    Args:
        fp_model: Full-precision model
        mpq_model: Mixed-precision quantized model
        data_loader: DataLoader for evaluation
        tokenizer: Tokenizer for generation comparison
        n_batches: Number of batches to evaluate
        prompts: Prompts for generation (uses defaults if tokenizer provided)
        verbose: Print results
        
    Returns:
        EvalResult with comparison metrics
    """
    evaluator = LLMEvaluator(fp_model, mpq_model, tokenizer)
    
    if prompts is None and tokenizer is not None:
        prompts = DEFAULT_PROMPTS
    
    results = evaluator.full_eval(data_loader, prompts, n_batches)
    
    if verbose:
        print("\n" + "=" * 60)
        print("LLM Evaluation Results: FP vs MPQ")
        print("=" * 60)
        
        print("\nPerplexity:")
        print(f"  FP Model:  {results.perplexity_fp:.2f}")
        print(f"  MPQ Model: {results.perplexity_mpq:.2f}")
        print(f"  Ratio:     {results.perplexity_ratio:.3f}x")
        
        print("\nToken Agreement:")
        print(f"  Top-1:  {results.token_agreement.top1_agreement:.2%}")
        print(f"  Top-5:  {results.token_agreement.top5_agreement:.2%}")
        print(f"  Top-10: {results.token_agreement.top10_agreement:.2%}")
        print(f"  KL Div: {results.token_agreement.kl_divergence:.4f}")
        print(f"  Rank ρ: {results.token_agreement.rank_correlation:.4f}")
        
        if results.generation is not None:
            print("\nGeneration Comparison:")
            print(f"  Exact Match:       {results.generation.exact_match:.2%}")
            print(f"  Token Overlap:     {results.generation.token_overlap:.2%}")
            print(f"  Avg Prefix Match:  {results.generation.prefix_match_len:.1f} tokens")
        
        print("=" * 60 + "\n")
    
    return results
