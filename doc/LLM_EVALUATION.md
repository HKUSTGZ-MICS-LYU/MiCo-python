# LLM Evaluation Methods for Mixed Precision Quantization

## Overview

This document proposes evaluation methods for comparing Mixed Precision Quantized (MPQ) LLM models against their original full-precision (FP) counterparts. The goal is to establish reliable metrics that capture both accuracy degradation and performance characteristics.

## Motivation

Direct perplexity evaluation on WikiText-2 alone is insufficient for understanding MPQ model quality because:

1. **Perplexity is aggregate**: It doesn't show where errors occur or their severity
2. **Token-level errors vary**: Some prediction errors are semantically worse than others
3. **Real-world usage differs**: Actual generation quality may not correlate with perplexity

## Proposed Evaluation Methods

### 1. Token-Level Agreement Analysis

Compare the predicted token distributions between FP and MPQ models:

```python
# For each input sequence:
fp_logits = fp_model(input_ids)
mpq_logits = mpq_model(input_ids)

# Metrics to compute:
- Top-1 Agreement: % of positions where top prediction matches
- Top-5 Agreement: % where FP's top-1 is in MPQ's top-5
- KL Divergence: Distribution similarity between outputs
- Rank Correlation: Spearman correlation of token rankings
```

**Benefits**:
- Shows exactly where MPQ diverges from FP
- Identifies problematic layers/positions
- Correlates with generation quality

### 2. Generation Comparison

Compare generated outputs for the same prompts:

```python
prompts = [
    "The capital of France is",
    "In machine learning, a neural network",
    "The quick brown fox",
    ...
]

for prompt in prompts:
    fp_output = fp_model.generate(prompt, max_tokens=50)
    mpq_output = mpq_model.generate(prompt, max_tokens=50)
    
    # Metrics:
    - Exact match rate
    - Token overlap (Jaccard similarity)
    - BLEU/ROUGE scores
    - Semantic similarity (embedding distance)
```

**Benefits**:
- Directly measures generation quality
- Human-interpretable outputs
- Catches cascading errors from quantization

### 3. Task-Based Benchmarks

Use established benchmarks to measure capability retention:

| Benchmark | Task Type | Metric |
|-----------|-----------|--------|
| HellaSwag | Commonsense reasoning | Accuracy |
| MMLU (subset) | Knowledge | Accuracy |
| ARC-Easy | Science QA | Accuracy |
| WinoGrande | Coreference | Accuracy |
| TruthfulQA | Factuality | MC1/MC2 |

**Implementation approach**:
- Use lm-evaluation-harness or similar framework
- Compare FP vs MPQ scores on same benchmarks
- Report relative accuracy retention: `mpq_score / fp_score`

### 4. Inference Quality Score (IQS)

A composite metric combining multiple evaluations:

```
IQS = w1 * (TopK_Agreement) + w2 * (1 - Normalized_KL) + w3 * (Perplexity_Retention)

Where:
- TopK_Agreement: Fraction of top-k predictions matching FP
- Normalized_KL: KL divergence normalized to [0,1]
- Perplexity_Retention: fp_ppl / mpq_ppl (capped at 1.0)
```

## Recommended Evaluation Pipeline

### Quick Evaluation (for MPQ search iterations)

During search, use fast metrics:
1. **Token Agreement Rate** (Top-1 and Top-5)
2. **Average KL Divergence**

These are computed per-batch and don't require generation.

### Full Evaluation (for final model comparison)

After search completes:
1. Token-level analysis on validation set
2. Generation comparison on standard prompts
3. Optional: Subset of task benchmarks

## Implementation

See `MiCoLLMEval.py` for the evaluation utilities:

```python
from MiCoLLMEval import LLMEvaluator

evaluator = LLMEvaluator(fp_model, mpq_model, tokenizer)

# Quick evaluation
quick_results = evaluator.quick_eval(test_loader)
print(f"Top-1 Agreement: {quick_results['top1_agreement']:.2%}")
print(f"Top-5 Agreement: {quick_results['top5_agreement']:.2%}")
print(f"KL Divergence: {quick_results['kl_divergence']:.4f}")

# Generation comparison
gen_results = evaluator.generation_eval(prompts, max_tokens=50)
print(f"Token Overlap: {gen_results['token_overlap']:.2%}")
print(f"Exact Match: {gen_results['exact_match']:.2%}")

# Full benchmark (optional, requires lm-eval-harness)
bench_results = evaluator.benchmark_eval(tasks=["hellaswag", "arc_easy"])
```

## Metrics Summary

| Metric | Speed | Usefulness | When to Use |
|--------|-------|------------|-------------|
| Perplexity | Fast | Medium | Baseline sanity check |
| Top-K Agreement | Fast | High | MPQ search iterations |
| KL Divergence | Fast | High | MPQ search iterations |
| Generation Comparison | Medium | Very High | Final evaluation |
| Task Benchmarks | Slow | Very High | Final comparison |

## Future Work

1. **Calibration analysis**: Check if quantized models are well-calibrated
2. **Layer-wise impact**: Identify which layers are most sensitive to quantization
3. **Efficiency metrics**: Include latency/throughput in composite score
4. **Domain-specific evaluation**: Specialized prompts for target use cases

## References

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [HELM](https://crfm.stanford.edu/helm/latest/)
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
