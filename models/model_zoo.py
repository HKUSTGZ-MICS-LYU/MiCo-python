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

def from_zoo(name: str, shuffle = False, batch_size: int = 32):

    model = None
    train_loader, test_loader = None, None
    if name == "mlp_mnist":
        model = MLP(256, config={"Layers": [64, 64, 64, 10]}).to(device)
        train_loader, test_loader = mnist(shuffle=shuffle, batch_size=batch_size, resize=16)
    elif name == "lenet_mnist":
        model = LeNet(1).to(device)
        train_loader, test_loader = mnist(shuffle=shuffle, batch_size=batch_size)
    elif name == "cmsiscnn_cifar10":
        model = CmsisCNN(3).to(device)
        train_loader, test_loader = cifar10(shuffle=shuffle, batch_size=batch_size)
    elif name == "vgg_cifar10":
        model = VGG(3).to(device)
        train_loader, test_loader = cifar10(shuffle=shuffle, batch_size=batch_size)
    elif name == "resnet8_cifar100":
        model = resnet_alt_8(100).to(device)
        train_loader, test_loader = cifar100(shuffle=shuffle, batch_size=batch_size)
    elif name == "resnet18_cifar100":
        model = resnet_alt_18(100).to(device)
        train_loader, test_loader = cifar100(shuffle=shuffle, batch_size=batch_size)
    elif name == "mobilenetv2_cifar100":
        model = MobileNetV2().to(device)
        train_loader, test_loader = cifar100(shuffle=shuffle, batch_size=batch_size)
    elif name == "squeezenet_cifar100":
        model = SqueezeNet().to(device)
        train_loader, test_loader = cifar100(shuffle=shuffle, batch_size=batch_size)
    elif name == "tinyllama":
        model = TinyLLaMa1M().to(device)
        train_loader, test_loader = tinystories(
            max_seq_len=model.params.max_seq_len,
            vocab_size=model.params.vocab_size,
            device=device,
            batch_size=batch_size,
            shuffle=shuffle)
    else:
        raise ValueError(f"Model {name} not found in zoo.")
    return model, train_loader, test_loader