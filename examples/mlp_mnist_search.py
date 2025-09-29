import torch
import random
import numpy as np

from matplotlib import pyplot as plt

from MiCoEval import MiCoEval

from models import MLP
from datasets import mnist

from tqdm import tqdm

from searchers import (
    RegressionSearcher, BayesSearcher, 
    NLPSearcher, HAQSearcher, MiCoSearcher
)

from tqdm import tqdm

random.seed(0)
np.random.seed(0)

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device", device)
    config = {
        "Layers": [64, 64, 64, 10],
    }
    model = MLP(in_features=256, config=config).to(device)


    train_loader, test_loader = mnist(batch_size=32, num_works=0, resize=16)

    evaluator = MiCoEval(model, 10, train_loader, test_loader, 
                         "output/ckpt/mlp_mnist.pth")
    
    dim = model.n_layers * 2
    # bitwidths = [2, 4, 6, 8]
    bitwidths = [4, 5, 6, 7, 8]
    max_bops = evaluator.eval_bops([8] * evaluator.n_layers*2)

    for model in ["bo", "mico", "nlp", "haq"]:
        random.seed(0)
        np.random.seed(0)
        print("Model Type:", model)

        if model == "bo":
            searcher = BayesSearcher(
                evaluator, n_inits=10, qtypes=bitwidths)
        elif model == "nlp":
            searcher = NLPSearcher(
                evaluator, n_inits=10, qtypes=bitwidths
            )
        elif model == "mico":
            searcher = MiCoSearcher(
                evaluator, n_inits=10, qtypes=bitwidths
            )
        elif model == "haq":
            searcher = HAQSearcher(
                evaluator, n_inits=10, qtypes=bitwidths
                )
        else:
            searcher = RegressionSearcher(
                evaluator, n_inits=10, qtypes=bitwidths,
                model_type=model)

        res_x, res_y = searcher.search(20, 'ptq_acc', 'bops', max_bops*0.5)
        print(f"Best Scheme: {res_x}")
        print(f"Best Accuracy: {res_y}")
        # print(f"Bitfusion SpeedUp: {max_latency / evaluator.eval_latency(res_x):.2f}x")
        plt.plot(searcher.best_trace, label=model)

    plt.hlines(evaluator.baseline_acc, 0, len(searcher.best_trace), 
               colors='r', linestyles='dashed')
    
    plt.legend()
    plt.savefig("output/figs/mlp_mnist_search.pdf")