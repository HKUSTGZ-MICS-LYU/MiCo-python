import torch
import random
import numpy as np

from matplotlib import pyplot as plt

from MiCoEval import MiCoEval
from MiCoProxy import get_mico_matmul_proxy, get_mico_conv2d_proxy, get_mico_misc_kernel_proxy

from models import CmsisCNN
from datasets import cifar10

from tqdm import tqdm

from searchers import (
    RegressionSearcher, BayesSearcher, 
    NLPSearcher, HAQSearcher, MiCoSearcher
)

random.seed(0)
np.random.seed(0)

mico_target = "high"

if __name__ == "__main__":

    model = CmsisCNN(3)
    train_loader, test_loader = cifar10(shuffle=False, num_works=4)
    evaluator = MiCoEval(model, 3, train_loader, test_loader, 
                         "output/ckpt/cmsiscnn_cifar10.pth",
                         output_json="output/json/cmsiscnn_cifar10_search_mico.json",)

    matmul_proxy = get_mico_matmul_proxy(mico_target)
    conv2d_proxy = get_mico_conv2d_proxy(mico_target)
    evaluator.set_proxy(matmul_proxy, conv2d_proxy)
    evaluator.set_misc_proxy(get_mico_misc_kernel_proxy)
    evaluator.set_mico_target(mico_target)
    evaluator.get_misc_latency()

    dim = model.n_layers * 2
    bitwidths = [1, 2, 4, 8]
    max_latency = evaluator.eval_pred_latency([8] * dim)
    print("INT8 Predicted Latency:", max_latency)
    min_latency = evaluator.eval_pred_latency([1] * dim)
    print("INT1 Predicted Latency:", min_latency)

    random.seed(0)
    np.random.seed(0)
    searcher = MiCoSearcher(
        evaluator, n_inits=10, qtypes=bitwidths
    )
    res_x, res_y = searcher.search(
        10, 'qat_acc', 'latency_proxy', max_latency*0.8)
    print(f"Best Scheme: {res_x}")
    print(f"Deploying Model to MiCo CPU....")
    res = evaluator.eval_latency(res_x, target="mico")
    print(f"MPQ Real Latency: {res}")
    res_int8 = evaluator.eval_latency([8] * dim, target="mico")
    print(f"INT8 Real Latency: {res_int8}")
    print(f"Real Speedup: {res/res_int8}")