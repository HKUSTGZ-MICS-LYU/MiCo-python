# MiCo: Mixed Precision Neural Networks On SIMD-extended RISC-V CPUs

This is the Python codebase for MiCo Framework. (W.I.P.)

## Pre-requisite

```shell
# 
conda create -n mico_env python=3.10
conda activate mico_env

conda install pygmo
pip install -r requirements.txt
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
make run_host
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
| MobileNetV2 | Linear, BottleNeck (Conv2D) | Supported | Supported |
| SqueezeNet | Linear, Conv2D | Supported | Supported |
| LLaMa | Transformers (Linear) | Supported | Not Yet |

## Supported Datasets

Currently MiCo includes the following datasets:

+ MNIST
+ Fashion MNIST
+ CIFAR-10
+ CIFAR-100
+ TinyStories

## Acknowledgement

ucb-bar/Baremetal-NN (For Codegen with Torch FX Interpreter):

https://github.com/ucb-bar/Baremetal-NN

mit-han-lab/haq (For HAQ Searcher Implementation):

https://github.com/mit-han-lab/haq

## Roadmap

Check our [Roadmap](issues/1) to see what's on the plan!