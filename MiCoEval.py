import torch
import torch.nn as nn
import torch.nn.utils.fusion as fusion
import torch.nn.functional as F

import numpy as np

from matplotlib import pyplot as plt

import os
import copy
import json
import random
import itertools 
import warnings

from tqdm import tqdm

from MiCoUtils import (
    list_qlayers,
    list_quantize_layers,
    replace_quantize_layers,
    set_to_qforward
)

from MiCoModel import MiCoModel
from SimUtils import sim_bitfusion, sim_mico


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")

class MiCoEval:
    def __init__(self, model: MiCoModel, 
                 epochs: int, train_loader, test_loader, 
                 pretrained_model,
                 lr=0.0001, model_name = "", objective='ptq_acc'):
        
        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.pretrained_model = pretrained_model
        self.lr = lr

        self.model_name = model_name
        self.n_layers = self.model.n_layers

        # Initial Conversion and Test
        res = self.eval_ptq([8]*self.n_layers*2)
        self.layers = model.get_qlayers()
        assert self.n_layers > 0, "No QLayer found! Please convert the model first."
        
        self.layer_macs = [layer.get_mac() for layer in self.layers]
        self.layer_params = [layer.get_params() for layer in self.layers]
        
        self.set_eval(objective)

        print("Model:", model._get_name())
        print("Number of QLayers: ", self.n_layers)
        print("Total MACs: ", np.sum(self.layer_macs))
        print("Total Params: ", np.sum(self.layer_params))
        print("INT8 Model Accuracy: ", res)
        return
    
    def set_eval(self, objective: str):
        self.objective = objective
        if self.objective == 'ptq_acc':
            self.eval = self.eval_ptq
        elif self.objective == 'qat_acc':
            self.eval = self.eval_qat
        elif self.objective == 'bops':
            self.eval = self.eval_bops
        elif self.objective == 'latency_bitfusion':
            self.eval = lambda scheme: self.eval_latency(scheme, 'bitfusion')
        elif self.objective == 'latency_mico':
            self.eval = lambda scheme: self.eval_latency(scheme, 'mico')
    
    def load_pretrain(self):
        ckpt = torch.load(self.pretrained_model)
        self.model.load_state_dict(ckpt)
        return

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
    
    # Classic Bit Operation Counts
    def eval_bops(self, scheme: list):
        bops = 0
        wq = np.array(scheme[:self.n_layers])
        aq = np.array(scheme[self.n_layers:])
        bops = np.dot(wq*aq, self.layer_macs)
        return bops

    # TODO: Implement latency evaluation
    def eval_latency(self, scheme: list, target: str = 'bitfusion'):

        wq = scheme[:self.n_layers]
        aq = scheme[self.n_layers:]

        res = -1

        if target == 'bitfusion':
            res = sim_bitfusion(self.model_name, wq, aq)
        elif target == 'mico':
            res = sim_mico(wq, aq)
        else:
            raise ValueError(f"Target \"{target}\" not supported.")
        
        return res
    
# Test
if __name__ == "__main__":
    from models import LeNet, VGG
    from datasets import mnist, cifar10

    # model = LeNet(1)
    # train_loader, test_loader = mnist(shuffle=False)
    # evaluator = MiCoEval(model, 10, train_loader, test_loader, 
    #                      "output/ckpt/lenet_mnist.pth", model_name="lenet")
    
    model = VGG(3, num_class=10)
    train_loader, test_loader = cifar10(shuffle=False)
    evaluator = MiCoEval(model, 10, train_loader, test_loader, 
                         "output/ckpt/vgg_cifar10.pth", model_name="vgg")
    evaluator.set_eval("latency_bitfusion")
    res = evaluator.eval([8]*model.n_layers*2)
    print(res)
