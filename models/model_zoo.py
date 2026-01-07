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
    tinystories,
    imagenet
)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_WORKS = 8

IMAGENET_ROOT = "data"

def from_zoo(name: str, shuffle = False, batch_size: int = 32):

    model = None
    train_loader, test_loader = None, None
    if name == "mlp_mnist":
        model = MLP(256, config={"Layers": [64, 64, 64, 10]}).to(device)
        train_loader, test_loader = mnist(shuffle=shuffle, batch_size=batch_size, resize=16, num_works=NUM_WORKS)
    elif name == "lenet_mnist":
        model = LeNet(1).to(device)
        train_loader, test_loader = mnist(shuffle=shuffle, batch_size=batch_size, num_works=NUM_WORKS)
    elif "cifar10" in name:
        # Parse Model + Dataset
        model_name, dataset_name = name.split("_")
        n_classes = 10
        if dataset_name == "cifar10":
            train_loader, test_loader = cifar10(shuffle=shuffle, batch_size=batch_size, num_works=NUM_WORKS)
        elif dataset_name == "cifar100":
            n_classes = 100
            train_loader, test_loader = cifar100(shuffle=shuffle, batch_size=batch_size, num_works=NUM_WORKS)
        model_dict = {
            "cmsiscnn": CmsisCNN(3),
            "vgg": VGG(3, n_classes),
            "resnet8": resnet_alt_8(n_classes),
            "resnet18": resnet_alt_18(n_classes),
            "mobilenetv2": MobileNetV2(n_classes),
            "squeezenet": SqueezeNet(n_classes)
        }
        model = model_dict[model_name].to(device)
    elif name == "tinyllama":
        model = TinyLLaMa1M().to(device)
        train_loader, test_loader = tinystories(
            max_seq_len=model.params.max_seq_len,
            vocab_size=model.params.vocab_size,
            device=device,
            batch_size=batch_size,
            num_works=NUM_WORKS)
    elif "imagenet" in name:
        model = name.replace("_imagenet", "")
        train_loader, test_loader = imagenet(shuffle=shuffle,batch_size=batch_size, 
                                             num_works=NUM_WORKS, root=IMAGENET_ROOT)
    else:
        raise ValueError(f"Model {name} not found in zoo.")
    return model, train_loader, test_loader