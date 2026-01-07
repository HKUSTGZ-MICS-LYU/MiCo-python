from typing import Any, Dict, List, Tuple, Callable

import logging
import operator
import numpy as np
import torch
import torch.nn
import torch.fx

# Add hw.bitfusion to the Python path 
# Remember to add it to the linter of your ide as well
import sys
sys.path.append("hw/bitfusion")

from dnnweaver2.graph import Graph

from dnnweaver2 import get_tensor
from dnnweaver2.tensorOps.cnn import (
    conv2D, maxPool, flatten, matmul, addBias, batch_norm, 
    reorg, concat, leakyReLU, add, globalAvgPool, typecast
)
from dnnweaver2.scalar.dtypes import FQDtype, FixedPoint, Dtype
from bitfusion.src.benchmarks.benchmarks import conv, fc
from sim_utils import sim_results

from MiCoQLayers import BitConv2d, BitLinear, weight_quant
from MiCoUtils import weight_export, fuse_model, fuse_model_seq, get_model_macs
from MiCoCodeGen import MiCoTrace, MiCoCodeGen

'''
Export MiCo Model into the DNNWeaver Op Graph (for BitFusion)
'''

class Int(Dtype):
    def __init__(self, bits):
        self.bits = bits
        self.int_bits = bits
        self.frac_bits = 0
        self.op_str = 'INT{}'.format(self.bits)

def int2dtype(q):
    return Int(q)

def relu(data, name=None, dtype=None):
    """Create a ReLU operation in the DNNWeaver2 graph"""
    if dtype is None:
        dtype = FQDtype.FP32  # Default to FXP32 if no dtype is specified
    return leakyReLU(data, alpha=0.0, name=name, dtype=dtype)

class MiCoGraphGen(MiCoCodeGen):

    def __init__(self, model: torch.nn.Module, graph: Graph):
        # Dictionary to store tensors in the DNNWeaver2 graph
        self.dnn_tensors = {}
        self.op_graph = graph
        # Initialize the parent class
        super().__init__(model)
        
        # Batch size for the input
        self.batch_size = 1
    
    def reset(self):
        super().reset()
        self.dnn_tensors = {}
    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def get_dnn_dtype(self, tensor, qbit=0):
        """Convert PyTorch tensor dtype to DNNWeaver2 dtype"""
        if qbit > 0:
            return int2dtype(qbit)
        else:
            return FQDtype.FXP32  # Default to FXP32
    
    def handle_placeholder(self, n: torch.fx.node.Node, out: torch.Tensor):
        """Handle the input placeholder node"""
        super().handle_placeholder(n, out)
        with self.op_graph.as_default():
            with self.op_graph.name_scope('input'):
                # Create input tensor with proper shape
                if out.dim() == 4:  # For images (NCHW -> NHWC)
                    shape = (self.batch_size, out.shape[2], out.shape[3], out.shape[1])
                elif out.dim() == 2:  # For vectors (NC)
                    shape = (self.batch_size, out.shape[1])
                else:
                    shape = tuple([self.batch_size] + list(out.shape[1:]))

                tensor = get_tensor(
                    shape=shape, 
                    name=n.name, 
                    dtype=self.get_dnn_dtype(out),
                    trainable=False
                )
                self.dnn_tensors[n.name] = tensor
    
    def handle_call_module(self, n: torch.fx.node.Node, out: torch.Tensor):
        """Handle calls to modules like Conv2d, Linear, etc."""
        super().handle_call_module(n, out)
        
        module = self.get_module(n.target)
        input_node = self.node_info[n.name][0][0]
        input_tensor = self.dnn_tensors[input_node.name]
        with self.op_graph.as_default():
            with self.op_graph.name_scope(n.name):
                if isinstance(module, (BitConv2d, torch.nn.Conv2d)):
                    # Get quantization info for BitConv2d
                    w_dtype = int2dtype(module.qtype) if hasattr(module, 'qtype') else FQDtype.FXP32
                    a_dtype = int2dtype(module.act_q) if hasattr(module, 'act_q') else FQDtype.FXP32
                    # Handle padding format differences
                    padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
                    stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                    # Create weight tensor
                    weight_shape = (module.out_channels, module.kernel_size[0], module.kernel_size[1], module.in_channels)
                    weights = get_tensor(
                        shape=weight_shape,
                        name=f"{n.name}_weights",
                        dtype=w_dtype,
                        trainable=True
                    )
                    # Create bias tensor
                    bias_shape = (module.out_channels,)
                    bias = get_tensor(
                        shape=bias_shape,
                        name=f"{n.name}_bias",
                        dtype=FQDtype.FXP32,
                        trainable=True
                    )
                    # Create conv operation - note we're passing the tensors
                    result = conv2D(
                        typecast(input_tensor, a_dtype), 
                        weights,
                        bias,
                        name=n.name,
                        stride=(1, stride, stride, 1),
                        pad='SAME' if padding > 0 else 'VALID',
                        group=module.groups,
                        dtype=FQDtype.FXP32 
                    )
                    self.dnn_tensors[n.name] = result
                elif isinstance(module, (BitLinear, torch.nn.Linear)):
                    # Get quantization info for BitLinear
                    w_dtype = int2dtype(module.qtype) if hasattr(module, 'qtype') else FQDtype.FXP32
                    a_dtype = int2dtype(module.act_q) if hasattr(module, 'act_q') else FQDtype.FXP32
                    # Create weight tensor
                    weights = get_tensor(
                        shape=(module.out_features, module.in_features),
                        name=f"{n.name}_weights",
                        dtype=w_dtype,
                        trainable=True
                    )
                    # Create bias tensor (or zeros if no bias)
                    bias = get_tensor(
                        shape=(module.out_features,),
                        name=f"{n.name}_bias",
                        dtype=FQDtype.FXP32,
                        trainable=True
                    )
                    # Create matmul operation with proper tensors
                    result = matmul(
                        typecast(input_tensor, a_dtype),
                        weights,
                        bias,
                        name=n.name,
                        dtype=FQDtype.FXP32
                    )
                    self.dnn_tensors[n.name] = result
                elif isinstance(module, torch.nn.BatchNorm2d):
                    # Create gamma and beta tensors for batch norm
                    gamma_shape = (module.num_features,)
                    gamma = get_tensor(
                        shape=gamma_shape,
                        name=f"{n.name}_gamma",
                        trainable=True
                    )
                    beta_shape = (module.num_features,)
                    beta = get_tensor(
                        shape=beta_shape,
                        name=f"{n.name}_beta",
                        trainable=True
                    )
                    # Create batch normalization operation
                    result = batch_norm(
                        input_tensor,
                        gamma,
                        beta,
                        name=n.name,
                        dtype=FQDtype.FXP32
                    )
                    self.dnn_tensors[n.name] = result
                elif isinstance(module, torch.nn.MaxPool2d):
                    # Get kernel size
                    kernel_size = module.kernel_size
                    if isinstance(kernel_size, int):
                        kernel_size = (kernel_size, kernel_size)
                    # Get stride
                    stride = module.stride
                    if isinstance(stride, int):
                        stride = (stride, stride)
                    # Create maxpool operation
                    result = maxPool(
                        input_tensor,
                        pooling_kernel=(1, kernel_size[0], kernel_size[1], 1),
                        stride=(1, stride[0], stride[1], 1),
                        pad='VALID',
                        name=n.name
                    )
                    self.dnn_tensors[n.name] = result
                elif isinstance(module, torch.nn.AvgPool2d):
                    # DNNWeaver2 might not have avgpool directly
                    kernel_size = module.kernel_size
                    if isinstance(kernel_size, int):
                        kernel_size = (kernel_size, kernel_size)
                    stride = module.stride
                    if isinstance(stride, int):
                        stride = (stride, stride)
                    # Using maxPool as placeholder (or would need to implement avgPool)
                    result = maxPool(
                        input_tensor,
                        pooling_kernel=(1, kernel_size[0], kernel_size[1], 1),
                        stride=(1, stride[0], stride[1], 1),
                        pad='VALID',
                        name=n.name
                    )
                    self.dnn_tensors[n.name] = result
                elif isinstance(module, torch.nn.AdaptiveAvgPool2d):

                    output_size = module.output_size
                    # Infer Kernel size and Stride based on output size
                    if isinstance(output_size, int):
                        output_size = (output_size, output_size)
                    kernel_size = (input_tensor.shape[1] // output_size[0],
                                   input_tensor.shape[2] // output_size[1])
                    stride = kernel_size
                    # Using maxPool as placeholder (or would need to implement avgPool)
                    result = maxPool(
                        input_tensor,
                        pooling_kernel=(1, kernel_size[0], kernel_size[1], 1),
                        stride=(1, stride[0], stride[1], 1),
                        pad='VALID',
                        name=n.name
                    )
                    self.dnn_tensors[n.name] = result
                elif isinstance(module, torch.nn.Flatten):
                    result = flatten(
                        input_tensor,
                        name=n.name
                    )
                    self.dnn_tensors[n.name] = result
                elif isinstance(module, (torch.nn.ReLU, torch.nn.ReLU6)):
                    result = leakyReLU(
                        input_tensor,
                        name=n.name,
                        alpha=0.0  # ReLU is leakyReLU with alpha=0
                    )
                    self.dnn_tensors[n.name] = result
                elif isinstance(module, (torch.nn.Identity, torch.nn.Dropout)):
                    # Pass through
                    self.dnn_tensors[n.name] = input_tensor
                else:
                    raise NotImplementedError(f"Module type {type(module)} not supported in DNNWeaver2 conversion")
    
    def handle_call_function(self, n: torch.fx.node.Node, out: torch.Tensor):
        """Handle function calls like torch.add, torch.relu, etc."""
        super().handle_call_function(n, out)

        function = n.target
        args = self.node_info[n.name][0]
        with self.op_graph.as_default():
            with self.op_graph.name_scope(n.name):
                if function == torch.nn.functional.relu:
                    input_node = args[0]
                    input_tensor = self.dnn_tensors[input_node.name]
                    result = leakyReLU(
                        input_tensor,
                        name=n.name,
                        alpha=0.0  # ReLU is leakyReLU with alpha=0
                    )
                    self.dnn_tensors[n.name] = result
                elif function == torch.flatten:
                    input_node = args[0]
                    input_tensor = self.dnn_tensors[input_node.name]
                    result = flatten(
                        input_tensor,
                        name=n.name
                    )
                    self.dnn_tensors[n.name] = result
                elif function == torch.cat:
                    # Get all tensors to concatenate
                    tensors = [self.dnn_tensors[node.name] for node in args[0]]
                    # Check for concat dimension parameter
                    dim = args[1] if len(args) > 1 else 1
                    # Convert PyTorch dim to DNNWeaver dim (accounting for NHWC vs NCHW)
                    if dim == 1 and len(tensors[0].shape) == 4:
                        # Channel dim in PyTorch (1) -> channel dim in DNNWeaver (3)
                        dim = -1
                    result = concat(
                        tensors,
                        concat_dim=dim,
                        name=n.name
                    )
                    self.dnn_tensors[n.name] = result
                elif function == torch.nn.functional.max_pool2d:
                    input_node = args[0]
                    input_tensor = self.dnn_tensors[input_node.name]
                    # Get kernel size
                    kernel_size = args[1]
                    if isinstance(kernel_size, tuple):
                        kernel_size = kernel_size[0]
                    # Get stride (default to kernel_size if not specified)
                    stride = args[2] if len(args) > 2 else kernel_size
                    if isinstance(stride, tuple):
                        stride = stride[0]
                    result = maxPool(
                        input_tensor,
                        pooling_kernel=(1, kernel_size, kernel_size, 1),
                        stride=(1, stride, stride, 1),
                        pad='VALID',
                        name=n.name
                    )
                    self.dnn_tensors[n.name] = result
                elif function == torch.add or function == operator.__add__:
                    # Get the two input tensors
                    input1 = self.dnn_tensors[args[0].name]
                    input2 = self.dnn_tensors[args[1].name]
                    # Create add operation
                    result = add(
                        (input1, input2),
                        name=n.name
                    )
                    self.dnn_tensors[n.name] = result
                else:
                    print(f"Warning: Function {function} not supported in DNNWeaver2 conversion")
                    # For unsupported operations, pass the input through
                    if args and hasattr(args[0], 'name') and args[0].name in self.dnn_tensors:
                        self.dnn_tensors[n.name] = self.dnn_tensors[args[0].name]
            return

    def handle_output(self, n: torch.fx.node.Node, out: torch.Tensor):
        """Handle the output node"""
        super().handle_output(n, out)
        # No special handling needed for output in DNNWeaver2 graph
        return
    
    def forward(self, *args):
        """Forward pass to create the DNNWeaver2 graph from the PyTorch model"""
        self.reset()
        self.example_inputs = args
        # Explicitly use the graph context for the entire forward pass
        with self.op_graph.as_default():
            self.run(*args)
        return
    
    def __call__(self, *args):
        return self.forward(*args)
    
    def sim(self):
        return sim_results(self.op_graph)

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import MiCoUtils as mico
    from models import MLP, LeNet, CmsisCNN, VGG, SqueezeNet, MobileNetV2, \
          resnet_alt_8, resnet_alt_18
    
    from torchvision.models import (
        resnet18, ResNet18_Weights, 
        mobilenet_v2, MobileNet_V2_Weights
    )

    from MiCoModel import from_torch
    from sim_utils import sim_results

    torch.manual_seed(0)

    # example_input = torch.randn(1, 256)
    # example_input = torch.randn(1, 1, 28, 28)
    example_input = torch.randn(1, 3, 32, 32)
    # example_input = torch.randn(1, 3, 224, 224)

    # m = MLP(in_features=256, config={"Layers": [64, 64, 64, 10]})
    # ckpt = torch.load("output/ckpt/mlp_mnist_mp.pth")

    # m = LeNet(1)
    # ckpt = torch.load("output/ckpt/lenet_mnist.pth")

    # m = CmsisCNN(in_channels=3)
    # ckpt = torch.load("output/ckpt/cmsiscnn_cifar10_mp.pth")

    m = VGG(in_channels=3, num_class=10)
    ckpt = torch.load("output/ckpt/vgg_cifar10.pth")

    # m = MobileNetV2(10)
    # ckpt = torch.load("output/ckpt/mobilenetv2_cifar10.pth")
    # m.default_dataset = "CIFAR10"

    # m = SqueezeNet(class_num=10)
    # ckpt = torch.load("output/ckpt/squeeze_cifar10.pth")
    # m.default_dataset = "CIFAR10"

    # m = resnet_alt_8(10)
    # m.default_dataset = "CIFAR10"
    # ckpt = torch.load("output/ckpt/resnet8_cifar10.pth")

    # m = resnet_alt_18(100)
    # ckpt = torch.load("output/ckpt/resnet18_cifar100.pth", map_location="cpu")
    # m.load_state_dict(ckpt)

    # m = from_torch(
    #     resnet18(weights=ResNet18_Weights.IMAGENET1K_V1))

    weight_q = [8] * m.n_layers
    activation_q = [8] * m.n_layers

    m.set_qscheme([weight_q, activation_q])
    # m=fuse_model(m)
    m.eval()
    graph = Graph("Model", "Dataset", logging.INFO)
    with graph.as_default():
        m = MiCoGraphGen(m, graph)
        m(example_input)


    print(f"Graph has {len(graph.op_registry)} operations")
    print(f"Graph has {len(graph.tensor_registry)} tensors")

    res = m.sim()
    print("-" * 50)
    for k, v in res.items():
        print(f"{k}: {v}")
    