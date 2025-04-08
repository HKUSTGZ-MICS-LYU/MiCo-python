import torch
from torch.utils.data.dataloader import DataLoader

import numpy as np

from matplotlib import pyplot as plt

import os
import json
import time
import warnings

from tqdm import tqdm

from MiCoModel import MiCoModel
from MiCoCodeGen import MiCoCodeGen
from MiCoUtils import fuse_model
from SimUtils import sim_bitfusion, sim_mico

from copy import deepcopy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")

class MiCoEval:
    def __init__(self, model: MiCoModel, 
                 epochs: int, 
                 train_loader: DataLoader, 
                 test_loader: DataLoader, 
                 pretrained_model,
                 lr=0.0001, model_name = "", 
                 objective='ptq_acc',
                 constraint='bops',
                 output_json='output/json/mico_eval.json') -> None:
        
        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.pretrained_model = pretrained_model
        self.lr = lr

        self.output_json = output_json
        self.model_name = model_name
        self.n_layers = self.model.n_layers
        
        self.data_trace = {}
        if os.path.exists(self.output_json):
            with open(self.output_json, 'r') as f:
                self.data_trace = json.load(f)
        
        self.set_eval(objective)
        self.set_constraint(constraint)

        # Initial Conversion and Test
        res = self.eval_f([8]*self.n_layers*2)
        self.baseline_acc = res
        self.layers = model.get_qlayers()
        assert self.n_layers > 0, "No QLayer found! Please convert the model first."
        
        self.layer_macs = [layer.get_mac() for layer in self.layers]
        self.layer_params = [layer.get_params() for layer in self.layers]
        
        # A Regression Model for Hardware Latency
        self.latency_model = None
        self.mico_target = "small"

        print("Model:", model._get_name())
        print("Number of QLayers: ", self.n_layers)
        print("Total MACs: ", np.sum(self.layer_macs))
        print("Total Params: ", np.sum(self.layer_params))
        print("INT8 Model Accuracy: ", res)
        return
    
    def eval(self, scheme: list):
        res = None
        if str(scheme) in self.data_trace:
            if self.objective in self.data_trace[str(scheme)]:
                res = self.data_trace[str(scheme)][self.objective]
        if res is None:
            res = self.eval_f(scheme)
            if str(scheme) not in self.data_trace:
                self.data_trace[str(scheme)] = {}
            if self.objective not in self.data_trace[str(scheme)]:
                self.data_trace[str(scheme)][self.objective] = res
            with open(self.output_json, 'w') as f:
                json.dump(self.data_trace, f, indent=4)
        else:
            print("[MiCoEval] Found result in cache.")
        return res

    def set_eval(self, objective: str):
        self.objective = objective
        self.eval_f = self.eval_dict()[objective]
        return
    
    def set_constraint(self, constraint: str):
        self.constraint = constraint
        self.constr = self.eval_dict()[constraint]
        return

    def eval_dict(self):
        return {
            'ptq_acc': self.eval_ptq,
            'qat_acc': self.eval_qat,
            'torchao_acc': self.eval_torchao,
            'bops': self.eval_bops,
            'size': self.eval_size,
            'latency_bitfusion': lambda scheme: self.eval_latency(scheme, 'bitfusion'),
            'latency_mico': lambda scheme: self.eval_latency(scheme, 'mico'),
            'latency_proxy': lambda scheme: self.eval_latency(scheme, 'proxy'),
            'latency_torchao': lambda scheme: self.eval_latency(scheme, 'torchao')
        }

    def load_pretrain(self):
        ckpt = torch.load(self.pretrained_model)
        self.model.load_state_dict(ckpt)
        return

    def set_proxy(self, model):
        self.latency_model = model

    def set_mico_target(self, mico_type: str):
        self.mico_target = mico_type

    def eval_ptq(self, scheme: list):
        wq = scheme[:self.n_layers]
        aq = scheme[self.n_layers:]
        self.load_pretrain()
        self.model.set_qscheme([wq, aq])
        return self.model.test(self.test_loader)['TestAcc']
    
    def eval_qat(self, scheme: list):
        wq = scheme[:self.n_layers]
        aq = scheme[self.n_layers:]
        self.load_pretrain()
        self.model.set_qscheme([wq, aq], qat=True)
        self.model.train_loop(self.epochs, self.train_loader, self.lr)
        return self.model.test(self.test_loader)['TestAcc']
    
    def eval_torchao(self, scheme: list):
        wq = scheme[:self.n_layers]
        aq = scheme[self.n_layers:]
        self.load_pretrain()
        self.model.set_qscheme_torchao([wq, aq])
        return self.model.test(self.test_loader)['TestAcc']

    # Classic Bit Operation Counts
    def eval_bops(self, scheme: list):
        bops = 0
        wq = np.array(scheme[:self.n_layers])
        aq = np.array(scheme[self.n_layers:])
        bops = np.dot(wq*aq, self.layer_macs)
        return bops
    
    def eval_pred_latency(self, scheme: list):

        assert self.latency_model is not None, "Latency model not set."

        pred_latency = 0
        wq = np.array(scheme[:self.n_layers])
        aq = np.array(scheme[self.n_layers:])
        macs = np.array(self.layer_macs)
        bmacs = macs * np.max([aq, wq], axis=0)
        wloads = macs * wq
        aloads = macs * aq

        X = np.column_stack((bmacs, wloads, aloads))
        pred_latency = self.latency_model.predict(X)
        pred_latency = np.sum(pred_latency)
        return pred_latency

    def eval_size(self, scheme:list):
        size = 0
        wq = np.array(scheme[:self.n_layers])
        size = np.dot(wq, self.layer_params)
        return size

    def eval_latency(self, scheme: list, target: str = 'bitfusion'):

        wq = scheme[:self.n_layers]
        aq = scheme[self.n_layers:]

        res = -1

        if target == 'bitfusion':
            res = sim_bitfusion(self.model_name, wq, aq)
        elif target == 'mico':
            assert self.mico_target is not None, "MiCo Target not set."
            gen_model = deepcopy(self.model)
            gen_model = fuse_model(gen_model)
            gen_model.set_qscheme([wq, aq])
            codegen = MiCoCodeGen(gen_model)
            self.input_size = self.train_loader.dataset[0][0].shape
            example_input = torch.randn(1, *self.input_size).to(DEVICE)
            codegen.forward(example_input)
            codegen.convert()
            if self.mico_target == "high": 
                codegen.build(target="mico_fpu")
            else:
                codegen.build(target="mico")
            res = sim_mico(self.mico_target)
        elif target == 'proxy':
            res = self.eval_pred_latency(scheme)
        elif target == 'torchao':
            self.model.set_qscheme_torchao([wq, aq])
            start_time = time.time()
            self.model.test(self.test_loader)
            end_time = time.time()
            res = end_time - start_time
        else:
            raise ValueError(f"Target \"{target}\" not supported.")
        if res is None:
            raise ValueError(f"Error in simulating the model on {target}.")
        return res
    
# Test
if __name__ == "__main__":
    from models import MLP, LeNet, VGG
    from datasets import mnist, cifar10

    # model = LeNet(1)
    # train_loader, test_loader = mnist(shuffle=False)
    # evaluator = MiCoEval(model, 10, train_loader, test_loader, 
    #                      "output/ckpt/lenet_mnist.pth", model_name="lenet")
    
    # model = VGG(3, num_class=10)
    # train_loader, test_loader = cifar10(shuffle=False)
    # evaluator = MiCoEval(model, 10, train_loader, test_loader, 
    #                      "output/ckpt/vgg_cifar10.pth", model_name="vgg")
    # res = evaluator.eval([8]*model.n_layers*2)
    # print("Accuracy:", res)
    # res = evaluator.constr([8]*model.n_layers*2)
    # print("BOPs:", res)

    model = MLP(256, config={"Layers": [64, 64, 64, 10]})
    train_loader, test_loader = mnist(shuffle=False, resize=16)
    evaluator = MiCoEval(model, 10, train_loader, test_loader, 
                        "output/ckpt/mlp_mnist.pth", model_name="mlp")
    res = evaluator.eval_latency([8]*model.n_layers*2, 'mico')
    print("Latency:", res)

