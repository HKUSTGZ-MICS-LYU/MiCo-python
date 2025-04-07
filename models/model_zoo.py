from models import (
    MLP,
    LeNet,
    CmsisCNN,
    VGG,
    TinyLLaMa1M,
    resnet_alt_8,
    resnet_alt_18,
    SqueezeNet,
    MobileNetV2,
)

from datasets import (
    mnist,
    fashion_mnist,
    cifar10,
    cifar100,
    tinystories
)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def from_zoo(name: str):

    model = None
    train_loader, test_loader = None, None
    if name == "mlp_mnist":
        model = MLP(256, config={"Layers": [64, 64, 64, 10]})
        train_loader, test_loader = mnist(shuffle=False, resize=16)
    elif name == "lenet_mnist":
        model = LeNet(1)
        train_loader, test_loader = mnist(shuffle=False)
    elif name == "cmsiscnn_cifar10":
        model = CmsisCNN(3)
        train_loader, test_loader = cifar10(shuffle=False)
    elif name == "vgg_cifar10":
        model = VGG(3)
        train_loader, test_loader = cifar10(shuffle=False)
    elif name == "resnet8_cifar100":
        model = resnet_alt_8(100)
        train_loader, test_loader = cifar100(shuffle=False)
    elif name == "resnet18_cifar100":
        model = resnet_alt_18(100)
        train_loader, test_loader = cifar100(shuffle=False)
    elif name == "mobilenetv2_cifar100":
        model = MobileNetV2()
        train_loader, test_loader = cifar100(shuffle=False)
    elif name == "squeezenet_cifar100":
        model = SqueezeNet()
        train_loader, test_loader = cifar100(shuffle=False)
    elif name == "tinyllama":
        model = TinyLLaMa1M()
        train_loader, test_loader = tinystories(
            max_seq_len=model.params.max_seq_len,
            vocab_size=model.params.vocab_size,
            device=device,
            shuffle=False)
    else:
        raise ValueError(f"Model {name} not found in zoo.")
    return model, train_loader, test_loader