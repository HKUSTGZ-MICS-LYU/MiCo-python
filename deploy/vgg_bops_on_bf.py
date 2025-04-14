import torch
import random
import numpy as np

from matplotlib import pyplot as plt

from MiCoEval import MiCoEval
from MiCoProxy import get_bitfusion_matmul_proxy, get_bitfusion_conv2d_proxy

from models import VGG
from datasets import cifar10

from tqdm import tqdm

from searchers import (
    RegressionSearcher, BayesSearcher, 
    NLPSearcher, HAQSearcher, MiCoSearcher
)

random.seed(0)
np.random.seed(0)

if __name__ == "__main__":

    model = VGG(3, 10)
    train_loader, test_loader = cifar10(shuffle=False)
    evaluator = MiCoEval(model, 10, train_loader, test_loader, 
                         "output/ckpt/vgg_cifar10.pth",
                         model_name="vgg",
                         output_json="output/json/vgg_cifar10_search.json",)
    

    matmul_proxy = get_bitfusion_matmul_proxy()
    conv2d_proxy = get_bitfusion_conv2d_proxy()
    evaluator.set_proxy(matmul_proxy, conv2d_proxy)

    dim = model.n_layers * 2
    bitwidths = [2, 4, 8]
    max_latency = evaluator.eval_bops([8] * dim)
    print("INT8 Predicted Latency:", max_latency)
    min_latency = evaluator.eval_bops([2] * dim)
    print("INT2 Predicted Latency:", min_latency)

    random.seed(0)
    np.random.seed(0)
    searcher = MiCoSearcher(
        evaluator, n_inits=10, qtypes=bitwidths
    )
    res_x, res_y = searcher.search(
        20, 'ptq_acc', 'bops', max_latency*0.7)
        
    print(f"Best Scheme: {res_x}")
    print(f"Best Accuracy: {res_y}")
    print(f"Deploying Model to BitFusion....")
    res = evaluator.eval_latency(res_x, target="bitfusion")
    print(f"MPQ Real Latency: {res}")
    res_int8 = evaluator.eval_latency([8] * dim, target="bitfusion")
    print(f"INT8 Real Latency: {res_int8}")
    print(f"Real Speedup: {res/res_int8}")