import torch
import random
import numpy as np

from matplotlib import pyplot as plt

from MiCoEval import MiCoEval

from MiCoModel import from_torch
from MiCoDatasets import imagenet
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v2, MobileNet_V2_Weights

from tqdm import tqdm

from searchers import (
    RegressionSearcher, BayesSearcher, 
    NLPSearcher, HAQSearcher, MiCoSearcher
)

from tqdm import tqdm

random.seed(0)
np.random.seed(0)

num_epoch = 5
batch_size = 32

BUDGET = 32
INIT   = 16

torch.manual_seed(0)
torch.cuda.manual_seed(0)

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device", device)

    model = "resnet18"

    train_loader, test_loader = imagenet(
        batch_size=32, num_works=8, 
        shuffle=False, root="~/work/MiCo/nn/data")
    
    evaluator = MiCoEval(
        model, 10, train_loader, test_loader,
        output_json = 'output/json/resnet18_imagenet.json')

    dim = evaluator.n_layers * 2
    bitwidths = [4, 5, 6, 7, 8]
    max_bops = evaluator.eval_bops([8] * evaluator.n_layers*2)

    # for model in ["bo", "mico", "nlp", "haq"]:
    for model in ["nlp"]:
        random.seed(0)
        np.random.seed(0)
        print("Model Type:", model)

        if model == "bo":
            searcher = BayesSearcher(
                evaluator, n_inits=INIT, qtypes=bitwidths)
        elif model == "nlp":
            searcher = NLPSearcher(
                evaluator, n_inits=INIT, qtypes=bitwidths
            )
        elif model == "mico":
            searcher = MiCoSearcher(
                evaluator, n_inits=INIT, qtypes=bitwidths
            )
        elif model == "haq":
            searcher = HAQSearcher(
                evaluator, n_inits=INIT, qtypes=bitwidths
                )
        else:
            searcher = RegressionSearcher(
                evaluator, n_inits=INIT, qtypes=bitwidths,
                model_type=model)
            
        res_x, res_y = searcher.search(BUDGET, 'ptq_acc', 'bops', max_bops*0.5)
        print(f"Best Scheme: {res_x}")
        print(f"Best Accuracy: {res_y}")