# MiCo: End-to-End Mixed Precision Neural Network Co-Exploration Framework for Edge AI

This is the Python codebase for MiCo Framework. (W.I.P.)

## Pre-requisite

```shell
conda create -n mico_env python=3.10
conda activate mico_env

conda install pygmo
pip install -r requirements.txt
```

If you encounter `ModuleNotFoundError` when trying to import local packages here, before you run the code:
```shell
export PYTHONPATH=$PYTHONPATH:.
```

## Getting Start

**To run Mixed Precision Search**:

Check `examples`, run the training code first and use the trained model for MPQ Search.
For example:
```
python examples/lenet_mnist.py # Train LeNet on MNIST
python examples/lenet_mnist_search.py # MPQ Search on trained LeNet
```

**To use the CodeGen**, check the code to change the models/datasets/precisions:
```
python MiCoCodeGen.py
```

**To compile the inference code** after generating the model header with the CodeGen:
```
git submodule update --init
cd project
make clean
make MAIN=main TARGET=<host, vexii>
```
**To run the inference on your host machine** after compilation:
```
make run-host
```
**To run the inference simulation on the VexiiRiscv** after compilation:

Check the [VexiiRiscv document](https://spinalhdl.github.io/VexiiRiscv-RTD/master/VexiiRiscv/HowToUse/index.html#run-a-simulation), load the elf from `project` to the simulator.

## Supported Models
| Model | Layers | MPQ Search | MPQ Deploy |
| ----- | ------ | ---------- | ---------- |
| MLP   | Linear         | Supported | Supported |
| LeNet | Linear, Conv2D | Supported | Supported |
| CNN   | Linear, Conv2D | Supported | Supported |
| VGG   | Linear, Conv2D | Supported | Supported |
| ResNet | Linear, BottleNeck (Conv2D) | Supported | Supported |
| MobileNetV2 | Linear, BottleNeck (Conv2D) | Supported | Supported |
| SqueezeNet | Linear, Conv2D | Supported | Supported |
| ShuffleNet | Linear, Conv2D | Supported | Supported |
| LLaMa | Transformers (Linear) | Supported | Supported |
| ViT   | Transformers (Linear) | Supported | Not Yet |

## Supported Datasets

Currently MiCo includes the following datasets:

+ MNIST
+ Fashion MNIST
+ CIFAR-10
+ CIFAR-100
+ TinyStories

## Main Components
Here are the main components/modules of MiCo.

**Basics**
+ `MiCoUtils`: Utilities for MiCo framework, including layer replacing, exporting, etc..
+ `MiCoModel`: Basic model class of MiCo, offering unified training/testing methods, and the layer-wise bitwidth assignment method.
+ `MiCoQLayers`: Fundamental quantized layer classes for MiCo models and quantization functions.
+ `MiCoEval`: Model evaluation for MiCo models, evaluating accuracy, BOPs, MACs, end-to-end latency results.
+ `MiCoAnalysis`: Various statistics for quantized models.

**Codegen**
+ `MiCoCodeGen`： C code generator for MiCo models.
+ `MiCoGraphGen`: DNN Weaver Op graph generator for MiCo models.
+ `MiCoLLaMaGen`: C code generator for MiCo TinyLLaMa models.

**Searchers**
+ `searchers.MiCoSearcher`: Main MPQ searcher of MiCo framework.

**Hardware-Aware**
+ `SimUtils`: Invoke simulations for BitFusion or VexiiRiscv hardware.
+ `MiCoProxy`: CBOPs proxy models for hardware latency predictions.

## Acknowledgement

ucb-bar/Baremetal-NN (For Codegen with Torch FX Interpreter):

https://github.com/ucb-bar/Baremetal-NN

mit-han-lab/haq (For HAQ Searcher Implementation):

https://github.com/mit-han-lab/haq

## Roadmap

Check our [Roadmap](/../../issues/1) to see what's on the plan!

---
<p align="center">
<img src="doc/icon_v1.jpg" width="50%" height="50%"/>
</p>
Generated with Gemini-2.5 Flash.
