import operator
import os
import inspect
from typing import Any, Dict, List, Tuple, Callable

import numpy as np
import torch
import torch.nn
import torch.fx
import struct


from MiCoQLayers import BitConv2d, BitLinear, weight_quant
from MiCoUtils import weight_export, fuse_model, fuse_model_seq, get_model_macs

from models.LLaMa import Transformer, ModelArgs

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def groupwise_mixed_quant_w4w8(tensor: torch.Tensor, gsize: int):
    """ Group-wise mixed quantization for a given tensor.
        tensor: the input tensor to be quantized
        qbits: a list of quantization bits for each group
        gsize: the size of each group
    """
    assert tensor.dim() == 2, "Only support 2D tensor"
    assert tensor.size(0) % gsize == 0, "Tensor size must be divisible by group size"

    n_groups = tensor.size(0) // gsize
    qbits = [4] * (n_groups)
    qbits[-1] = 8  # Last group uses 8-bit quantization

    qweight_list = []
    scale_list = []
    for i in range(n_groups):
        start = i * gsize
        end = (i + 1) * gsize
        w_group = tensor[start:end, :]
        qbit = qbits[i]
        qweight, scale = weight_quant(w_group, qbit)
        qweight_list.append(qweight)
        scale_list.append(scale)

    return qweight_list, scale_list, qbits

def mico_export(model: Transformer, filepath: str, 
                quantize_final_classifier: bool = True,
                int8_outliner: bool = False):
    version = 1

    out_file = open(filepath, 'wb')
    # first write out the header. the header will be 256 bytes
    # 1) write magic, which will be uint32 of "ak42" in ASCII
    out_file.write(struct.pack('I', 0x616b3432))
    # 2) write version, which will be int
    out_file.write(struct.pack('i', version))
    # 3) write the params, which will be 7 ints
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                    n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    # 4) write some other flags
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    out_file.write(struct.pack('B', int(shared_classifier)))
    pad = 256 - out_file.tell() # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b'\0' * pad)
    print("Header End Addr: ", hex(out_file.tell()))
    # now let's write out all the params
    weights = [
        *[layer.attention_norm.weight for layer in model.layers],
        *[layer.ffn_norm.weight for layer in model.layers],
        model.norm.weight,
    ]
    for w in weights:
        serialize_fp32(out_file, w)
    print("Non-NN Weight End Addr: ", hex(out_file.tell()))
    
    mico_weights = [
        *[layer.attention.wq for layer in model.layers],
        *[layer.attention.wk for layer in model.layers],
        *[layer.attention.wv for layer in model.layers],
        *[layer.attention.wo for layer in model.layers],
        *[layer.feed_forward.w1 for layer in model.layers],
        *[layer.feed_forward.w2 for layer in model.layers],
        *[layer.feed_forward.w3 for layer in model.layers],
    ]

    if not shared_classifier:
        mico_weights.append(model.output)
    
    for linear in mico_weights:
        if isinstance(linear, BitLinear):
            out_file.write(struct.pack('b', linear.qtype))
            out_file.write(struct.pack('b', linear.act_q))
    if out_file.tell() % 4 != 0:
        pad = 4 - out_file.tell() % 4
        out_file.write(b'\0' * pad)
    
    ws_buffer = []

    print("Quantization Padding End Addr: ", hex(out_file.tell()))
    for linear in mico_weights:
        if isinstance(linear, BitLinear):
                qweight, scale = weight_quant(linear.weight, linear.qtype)
                wb = weight_export(qweight, linear.qtype)
                ws = scale.detach().cpu().numpy()
                # out_file.write(struct.pack('f', ws))
                ws_buffer.append(struct.pack('f', ws))
                out_file.write(wb)
        elif isinstance(linear, torch.nn.Linear):
            serialize_fp32(out_file, linear.weight)

    if quantize_final_classifier:
        qweight, scale = weight_quant(model.tok_embeddings.weight, 8)
        wb = weight_export(qweight, 8)
        ws = scale.detach().cpu().numpy()
        # out_file.write(struct.pack('f', ws))
        ws_buffer.append(struct.pack('f', ws))
        out_file.write(wb)
    else:
        serialize_fp32(out_file, model.tok_embeddings.weight)

    for b in ws_buffer:
        out_file.write(b) # write all scales at the end

    print("Final End Addr: ", hex(out_file.tell()))
    # write to binary file
    out_file.close()
    print(f"Wrote {filepath}")
    # Get File Size
    file_size = os.path.getsize(filepath)
    print(f"File Size: {file_size / 1e6:.2f} MB")
    return

if __name__ == "__main__":
    from models import TinyLLaMa1M, TinyLLaMa3M, TinyLLaMa7M, TinyLLaMa28M
    
    model_path = "output/ckpt/llama_tiny_1M.pth"
    bin_path = "project/llama2/llama_1M_W8A8.bin"

    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model = TinyLLaMa1M()
    model.load_state_dict(ckpt["model"])
    model.eval()
    qscheme = [
        [8] * model.n_layers, # weight qscheme
        [8] * model.n_layers, # activation qscheme
    ]

    model.set_qscheme(qscheme)

    # (Optional) Keep only the first layer for testing
    # model.layers = torch.nn.ModuleList([model.layers[0]])
    # model.params.n_layers = 1

    mico_export(model, bin_path)
