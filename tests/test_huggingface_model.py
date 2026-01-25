"""
Tests for HuggingFace model integration and WikiText dataset.
"""
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Test basic imports
def test_huggingface_model_import():
    """Test that HuggingFaceModel can be imported."""
    from models import HuggingFaceModel, list_available_models, HF_MODEL_REGISTRY
    
    assert HuggingFaceModel is not None
    assert callable(list_available_models)
    assert isinstance(HF_MODEL_REGISTRY, dict)
    
    # Check that some models are registered
    available = list_available_models()
    assert len(available) > 0
    assert "gpt2" in available
    assert "opt-125m" in available


def test_wikitext_import():
    """Test that WikiText dataset functions can be imported."""
    from MiCoDatasets import wikitext, wikitext2, wikitext103, hf_text_dataset
    
    assert callable(wikitext)
    assert callable(wikitext2)
    assert callable(wikitext103)
    assert callable(hf_text_dataset)


def test_model_zoo_hf_names():
    """Test that model zoo lists HuggingFace models."""
    from models.model_zoo import list_zoo_models, _get_hf_model_names
    
    hf_names = _get_hf_model_names()
    assert len(hf_names) > 0
    
    zoo_models = list_zoo_models()
    assert any("hf_" in name for name in zoo_models)


def test_huggingface_model_gpt2_load():
    """Test loading GPT-2 small model (fast, lightweight test)."""
    try:
        import pytest
    except ImportError:
        pytest = None
    
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        if pytest:
            pytest.skip("transformers not installed")
        return
    
    from models import GPT2_Small
    
    # Load model with float32 for CPU compatibility
    model = GPT2_Small(max_seq_len=64, torch_dtype=torch.float32)
    
    # Check model properties
    assert model is not None
    assert model.n_layers > 0
    assert model.vocab_size > 0
    assert model.tokenizer is not None
    
    # Test forward pass with dummy input
    dummy_input = torch.randint(0, 100, (1, 32))
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape[0] == 1
    assert output.shape[1] == 32
    assert output.shape[2] == model.vocab_size


def test_huggingface_model_get_qlayers():
    """Test that HuggingFace model can identify quantizable layers."""
    try:
        import pytest
    except ImportError:
        pytest = None
    
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        if pytest:
            pytest.skip("transformers not installed")
        return
    
    from models import GPT2_Small
    
    model = GPT2_Small(max_seq_len=64, torch_dtype=torch.float32)
    
    # Get quantizable layers
    qlayers = model.get_qlayers()
    
    # GPT-2 should have multiple linear layers
    assert len(qlayers) > 0
    assert model.n_layers == len(qlayers)


def test_huggingface_model_generate():
    """Test text generation with HuggingFace model."""
    try:
        import pytest
    except ImportError:
        pytest = None
    
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        if pytest:
            pytest.skip("transformers not installed")
        return
    
    from models import GPT2_Small
    
    model = GPT2_Small(max_seq_len=64, torch_dtype=torch.float32)
    model.eval()
    
    # Generate from a simple prompt (GPT-2 uses EOS token as start)
    prompt_ids = torch.tensor([[50256]])  # EOS token for GPT-2
    
    with torch.no_grad():
        generated = model.generate(prompt_ids, max_new_tokens=10, temperature=0.0)
    
    assert generated.shape[0] == 1
    assert generated.shape[1] == 11  # 1 + 10 new tokens


def test_huggingface_model_size():
    """Test model size calculation."""
    try:
        import pytest
    except ImportError:
        pytest = None
    
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        if pytest:
            pytest.skip("transformers not installed")
        return
    
    from models import GPT2_Small
    
    model = GPT2_Small(max_seq_len=64, torch_dtype=torch.float32)
    
    size = model.get_model_size()
    size_mb = model.get_model_size_mb()
    
    # GPT-2 small has ~124M parameters
    assert size > 100_000_000  # At least 100M params
    assert size < 200_000_000  # Less than 200M params
    assert size_mb > 0


def test_wikitext2_shape():
    """Test WikiText-2 dataset returns correct shapes."""
    try:
        import pytest
    except ImportError:
        pytest = None
    
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        if pytest:
            pytest.skip("datasets or transformers not installed")
        return
    
    from MiCoDatasets import wikitext2
    
    # Load with small batch and sequence length for testing
    train_loader, test_loader = wikitext2(
        batch_size=2,
        max_seq_len=64,
        num_workers=0,
    )
    
    # Get one batch
    batch = next(iter(test_loader))
    input_ids, labels = batch
    
    assert input_ids.shape == (2, 64)
    assert labels.shape == (2, 64)
    assert input_ids.dtype == torch.int64
    assert labels.dtype == torch.int64


def test_huggingface_model_with_wikitext():
    """Test HuggingFace model evaluation on WikiText dataset."""
    try:
        import pytest
    except ImportError:
        pytest = None
    
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM
    except ImportError:
        if pytest:
            pytest.skip("datasets or transformers not installed")
        return
    
    from models import GPT2_Small
    from MiCoDatasets import wikitext2
    
    # Load model
    model = GPT2_Small(max_seq_len=64, torch_dtype=torch.float32)
    model.eval()
    
    # Load WikiText-2 with model's tokenizer
    train_loader, test_loader = wikitext2(
        batch_size=2,
        max_seq_len=64,
        tokenizer=model.tokenizer,
        num_workers=0,
    )
    
    # Test evaluation
    result = model.test(test_loader, eval_iters=5)
    
    assert "TestLoss" in result
    assert "TestAcc" in result
    assert result["TestLoss"] > 0  # Loss should be positive


def test_list_available_hf_models():
    """Test listing available HuggingFace models."""
    from models import list_available_models
    
    models = list_available_models()
    
    expected_models = [
        "tinyllama-1.1b",
        "qwen2-0.5b",
        "smollm-135m",
        "smollm-360m",
        "smollm-1.7b",
        "gpt2",
        "gpt2-medium",
        "opt-125m",
        "opt-350m",
    ]
    
    for model_name in expected_models:
        assert model_name in models, f"Expected {model_name} in available models"


if __name__ == "__main__":
    # Run basic import tests
    test_huggingface_model_import()
    print("✓ HuggingFace model import test passed")
    
    test_wikitext_import()
    print("✓ WikiText import test passed")
    
    test_model_zoo_hf_names()
    print("✓ Model zoo HF names test passed")
    
    test_list_available_hf_models()
    print("✓ List available HF models test passed")
    
    print("\nAll basic tests passed! Run with pytest for full test suite.")
