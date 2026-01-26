from models.MLP import MLP
from models.LeNet import LeNet, LeNetBN
from models.ResNet import ResNet8, ResNet18, ResNet34
from models.ResNetAlt import resnet_alt_8, resnet_alt_18, resnet_alt_34
from models.MobileNetV2 import MobileNetV2
from models.CmsisCNN import CmsisCNN
from models.VGG import VGG
from models.SqueezeNet import SqueezeNet
from models.ShuffleNet import shufflenet
from models.HARMLP import HARMLP
from models.KWSConv1d import KWSConv1d
from models.DSCNN import DSCNN
from models.M5 import M5

from models.LLaMa import Transformer, TinyLLaMa28M, TinyLLaMa11M, TinyLLaMa7M, TinyLLaMa3M, TinyLLaMa1M, TinyLLaMa447K
from models.ViT import ViT, ViT1M_cifar10, ViT1M_cifar100, ViTNAS

# HuggingFace pretrained models for edge deployment
from models.HuggingFaceModel import (
    HuggingFaceModel,
    TinyLlama1B,
    Qwen2_0_5B,
    SmolLM_135M,
    SmolLM_360M,
    SmolLM_1_7B,
    GPT2_Small,
    GPT2_Medium,
    OPT_125M,
    OPT_350M,
    HF_MODEL_REGISTRY,
    list_available_models,
    load_model as load_hf_model,
)
