from models import (
    MLP,
    LeNet,
    CmsisCNN,
    VGG,
    TinyLLaMa1M,
    resnet_alt_8,
    resnet_alt_18,
    SqueezeNet,
    shufflenet,
    MobileNetV2,
    ViT,
    HARMLP,
)

from datasets import (
    mnist,
    fashion_mnist,
    cifar10,
    cifar100,
    tinystories,
    imagenet,
    uci_har,
)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_WORKERS = 8

IMAGENET_ROOT = "data"

def from_zoo(name: str, shuffle = False, batch_size: int = 32):

    model = None
    train_loader, test_loader = None, None
    if name == "mlp_mnist":
        model = MLP(256, config={"Layers": [64, 64, 64, 10]}).to(device)
        train_loader, test_loader = mnist(shuffle=shuffle, batch_size=batch_size, resize=16, num_works=NUM_WORKERS)
    elif name == "lenet_mnist":
        model = LeNet(1).to(device)
        train_loader, test_loader = mnist(shuffle=shuffle, batch_size=batch_size, num_works=NUM_WORKERS)
    elif "cifar10" in name:
        # Parse Model + Dataset
        model_name, dataset_name = name.split("_")
        n_classes = 10
        if dataset_name == "cifar10":
            train_loader, test_loader = cifar10(shuffle=shuffle, batch_size=batch_size, num_works=NUM_WORKERS)
        elif dataset_name == "cifar100":
            n_classes = 100
            train_loader, test_loader = cifar100(shuffle=shuffle, batch_size=batch_size, num_works=NUM_WORKERS)
        model_dict = {
            "vit": ViT(3,n_classes),
            "cmsiscnn": CmsisCNN(3),
            "vgg": VGG(3, n_classes),
            "resnet8": resnet_alt_8(n_classes),
            "resnet18": resnet_alt_18(n_classes),
            "mobilenetv2": MobileNetV2(n_classes),
            "squeezenet": SqueezeNet(n_classes),
            "shufflenet": shufflenet(n_classes),
        }
        model = model_dict[model_name].to(device)
    elif name == "tinyllama":
        model = TinyLLaMa1M().to(device)
        train_loader, test_loader = tinystories(
            max_seq_len=model.params.max_seq_len,
            vocab_size=model.params.vocab_size,
            device=device,
            batch_size=batch_size,
            num_works=NUM_WORKERS)
    elif "imagenet" in name:
        model = name.replace("_imagenet", "")
        train_loader, test_loader = imagenet(shuffle=shuffle,batch_size=batch_size, 
                                             num_works=NUM_WORKERS, root=IMAGENET_ROOT)
    elif name == "har_mlp":
        model = HARMLP().to(device)
        train_loader, test_loader = uci_har(shuffle=shuffle, batch_size=batch_size, num_works=NUM_WORKERS)
    elif name == "kws_conv1d":
        from models import KWSConv1d
        from datasets import speechcommands

        model = KWSConv1d(n_classes=35).to(device)
        train_loader, test_loader = speechcommands(
            shuffle=shuffle, batch_size=batch_size, num_works=NUM_WORKERS)
    elif name == "m5_kws":
        from models.M5 import M5
        from datasets import speechcommands

        model = M5(n_input=1, n_output=35, stride=16, n_channel=32).to(device)
        train_loader, test_loader = speechcommands(
            shuffle=shuffle, batch_size=batch_size, num_works=NUM_WORKERS)
    elif name == "dscnn_kws":
        from models.DSCNN import DSCNN
        from datasets import speechcommands

        model = DSCNN(n_classes=35, input_size=[64, 81]).to(device)
        train_loader, test_loader = speechcommands(
            shuffle=shuffle, batch_size=batch_size, num_works=NUM_WORKERS, 
            preprocess="mfcc")
    # HuggingFace pretrained models with WikiText dataset
    elif name.startswith("hf_") or name in _get_hf_model_names():
        from models import HuggingFaceModel, load_hf_model, HF_MODEL_REGISTRY
        from datasets import wikitext2

        # Parse model name (remove "hf_" prefix if present)
        hf_name = name[3:] if name.startswith("hf_") else name
        
        if hf_name in HF_MODEL_REGISTRY:
            model = load_hf_model(hf_name).to(device)
        else:
            # Try loading as a HuggingFace model identifier
            model = HuggingFaceModel.from_pretrained(hf_name).to(device)
        
        # Load WikiText-2 with the model's tokenizer
        train_loader, test_loader = wikitext2(
            batch_size=batch_size,
            max_seq_len=model.params.max_seq_len,
            tokenizer=model.tokenizer,
            num_workers=NUM_WORKERS,
            shuffle=shuffle,
        )
    else:
        raise ValueError(f"Model {name} not found in zoo.")
    return model, train_loader, test_loader


def _get_hf_model_names():
    """Get list of available HuggingFace model names."""
    try:
        from models import HF_MODEL_REGISTRY
        return list(HF_MODEL_REGISTRY.keys())
    except ImportError:
        return []


def list_zoo_models():
    """List all available models in the zoo."""
    base_models = [
        "mlp_mnist",
        "lenet_mnist",
        "vit_cifar10",
        "cmsiscnn_cifar10",
        "vgg_cifar10",
        "resnet8_cifar10",
        "resnet18_cifar10",
        "mobilenetv2_cifar10",
        "squeezenet_cifar10",
        "shufflenet_cifar10",
        "tinyllama",
        "har_mlp",
        "kws_conv1d",
        "m5_kws",
        "dscnn_kws",
    ]
    
    # Add HuggingFace models
    hf_models = ["hf_" + name for name in _get_hf_model_names()]
    
    return base_models + hf_models
