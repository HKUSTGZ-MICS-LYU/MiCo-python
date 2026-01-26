"""
Example: Evaluate HuggingFace pretrained models on WikiText for MPQ search.

This example demonstrates how to:
1. Load a pretrained HuggingFace model (GPT-2, OPT, SmolLM, etc.)
2. Evaluate it on WikiText-2 dataset
3. Prepare it for mixed precision quantization search

Note: This example uses GPT-2 small by default as it's lightweight and 
downloads quickly. For edge deployment research, consider using:
- SmolLM-135M/360M: Very small, designed for on-device
- OPT-125M/350M: Efficient decoder-only models
- TinyLlama-1.1B: Good balance of size and capability
- Qwen2-0.5B: Strong performance for its size
"""
import time
import torch

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    HuggingFaceModel,
    list_available_models,
    load_hf_model,
)
from MiCoDatasets import wikitext2
from MiCoUtils import list_quantize_layers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model_name: str = "gpt2", batch_size: int = 4, max_seq_len: int = 128):
    """
    Load and evaluate a HuggingFace model on WikiText-2.
    
    Args:
        model_name: Name of the model to load (see list_available_models())
        batch_size: Batch size for evaluation
        max_seq_len: Maximum sequence length
    """
    print(f"Device: {device}")
    print(f"Loading model: {model_name}")
    
    # Load model
    model = load_hf_model(
        name=model_name,
        max_seq_len=max_seq_len,
        dtype=torch.float32,
    )
    
    model = model.to(device)
    model.eval()

    # Print model info
    print(f"Model: {model.params.model_name}")
    print(f"Parameters: {model.get_model_size() / 1e6:.2f}M")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    print(f"Vocab size: {model.vocab_size}")
    
    # Count quantizable layers
    qlayers = list_quantize_layers(model)
    print(f"Quantizable layers: {len(qlayers)}")
    print(f"Layer types: {set(type(l).__name__ for l in qlayers)}")

    print("Model Architecture:")
    print(model)
    
    # Test generation
    print("\nTesting generation...")
    prompt = "What is mixed precision quantization?"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    print(f"Prompt: '{prompt}'")
    try:
        text = model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except ValueError:
        # Fail back to default template
        text = prompt
    print(f"Input text: '{text}'")
    model_inputs = model.tokenizer([text], return_tensors="pt").to(device)

    torch.random.manual_seed(0)
    torch.cuda.random.manual_seed(0)
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids, 
            max_new_tokens=max_seq_len,
            attention_mask = model_inputs["attention_mask"]
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Generated: '{response}'")

    print("\n" + "="*50)
    print("Preparing for MPQ Search")
    print("="*50)
    
    n_layers = model.n_layers
    print(f"Number of quantizable layers: {n_layers}")
    
    # Example: Set uniform INT8 quantization
    weight_bits = [8] * n_layers
    activation_bits = [8] * n_layers
    
    print(f"Setting quantization scheme: {weight_bits[:5]}... (first 5 layers)")
    
    # Note: set_qscheme will replace the layers with quantized versions
    # This is typically done during search to evaluate different configurations
    model.set_qscheme([weight_bits, activation_bits], group_size=32)
    
    print("Quantization scheme applied successfully!")
    model_inputs = model.tokenizer([text], return_tensors="pt").to(device)

    torch.random.manual_seed(0)
    torch.cuda.random.manual_seed(0)
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids, 
            max_new_tokens=max_seq_len,
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Generated: '{response}'")

    return model



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate HuggingFace models on WikiText")
    parser.add_argument(
        "--model", 
        type=str, 
        default="qwen3-0.6b",
        help="Model name (gpt2, opt-125m, smollm-135m, etc.)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available pre-defined models:")
        for name in list_available_models():
            print(f"  - {name}")
        print("\nYou can also use any HuggingFace model identifier.")
        sys.exit(0)
    
    # Evaluate model
    model = evaluate_model(
        model_name=args.model,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )
