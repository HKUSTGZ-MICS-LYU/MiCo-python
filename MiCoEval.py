import torch
from torchvision.models import get_model
from torch.utils.data.dataloader import DataLoader

import numpy as np

from matplotlib import pyplot as plt

import os
import json
import time
import warnings

from tqdm import tqdm

from MiCoModel import MiCoModel, from_torch
from MiCoCodeGen import MiCoCodeGen
from MiCoUtils import fuse_model
from SimUtils import sim_bitfusion, sim_mico

from copy import deepcopy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")

class MiCoEval:
    def __init__(self, model: MiCoModel | str, 
                 epochs: int, 
                 train_loader: DataLoader, 
                 test_loader: DataLoader, 
                 pretrained_model: str = "",
                 lr=0.0001, model_name = "", 
                 objective='ptq_acc',
                 constraint='bops',
                 linear_group_size = 1,
                 output_json='output/json/mico_eval.json') -> None:
        
        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.pretrained_model = pretrained_model
        self.group_size = linear_group_size
        self.lr = lr

        self.from_torch = isinstance(model, str)  # If model is a string, load from torchvision

        self.load_pretrain()
        print("Pretrained Model Loaded.")

        self.output_json = output_json
        self.model_name = model_name
        self.n_layers = self.model.n_layers
        self.dim = self.n_layers * 2
        
        self.data_trace = {}
        if os.path.exists(self.output_json):
            with open(self.output_json, 'r') as f:
                self.data_trace = json.load(f)
        
        self.set_eval(objective)
        self.set_constraint(constraint)
        
        self.fp_acc = self.eval_fp()
        # Initial Conversion and Test
        res = self.eval_f([8]*self.n_layers*2)
        self.baseline_acc = res
        self.layers = self.model.get_qlayers()
        assert self.n_layers > 0, "No QLayer found! Please convert the model first."
        
        self.layer_macs = [layer.get_mac() for layer in self.layers]
        self.layer_params = [layer.get_params() for layer in self.layers]

        # A Regression Model for Hardware Latency
        self.mico_target = "small"
        self.misc_latency = 0.0
        self.misc_proxy = None

        print("Model:", self.model._get_name())
        print("Number of QLayers: ", self.n_layers)
        print("Total MACs: ", np.sum(self.layer_macs))
        print("Total Params: ", np.sum(self.layer_params))
        print("INT8 Model Accuracy: ", res)
        print("FP Model Accuracy: ", self.fp_acc)
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
        if self.from_torch:
            self.model = get_model(self.model, weights="DEFAULT").to(DEVICE)
            self.model = from_torch(self.model).to(DEVICE)
        else:
            ckpt = torch.load(self.pretrained_model, weights_only=False)
            if "model" in ckpt:
                ckpt = ckpt["model"]
            self.model.load_state_dict(ckpt)
        return

    def set_proxy(self, matmul_proxy, conv2d_proxy):
        self.matmul_proxy = matmul_proxy
        self.conv2d_proxy = conv2d_proxy
        return
    
    def set_misc_proxy(self, misc_proxy):
        self.misc_proxy = misc_proxy
        return
    
    def get_misc_latency(self):
        # assert self.misc_proxy is not None, "Misc Proxy not set."
        self.misc_latency = 0.0

        ignored_misc = ["memcpy", 
                        "MiCo_flatten2d_f32", 
                        "MiCo_relu4d_f32",
                        "MiCo_relu2d_f32"]

        gen_model = deepcopy(self.model)
        gen_model = fuse_model(gen_model)
        gen_model.set_qscheme([[8]*self.n_layers, [8]*self.n_layers], group_size=self.group_size)
        codegen = MiCoCodeGen(gen_model)
        self.input_size = self.train_loader.dataset[0][0].shape
        example_input = torch.randn(1, *self.input_size).to(DEVICE)
        codegen.forward(example_input)
        for forward_pass in codegen.model_forward:
            if ("bitconv2d" in forward_pass) or ("bitlinear" in forward_pass):
                continue
            else:
                layer_name = forward_pass.split('(')[0]
                if layer_name in ignored_misc:
                    continue
                else:
                    layer_name = layer_name.strip('MiCo_')
                    print("Misc Layer Name:", layer_name)
                    layer_args = forward_pass.split('(')[1].split(')')[0]
                    layer_args = layer_args.split(', ')
                    layer_feature = []
                    for layer_arg in layer_args:
                        if layer_arg.startswith('&model->'):
                            tensor_name = layer_arg.split('->')[1]
                            # Find variable size in the model init pass
                            tensor_shape = codegen.tensors[tensor_name]["tensor"].shape
                            layer_feature += tensor_shape
                        else:
                            layer_feature.append(int(layer_arg))
                    # Post-process on feature
                    if layer_name in ["avgpool4d_f32", "maxpool4d_f32"]:
                        layer_feature = layer_feature[4:] # Skip output shape
                        layer_feature = layer_feature[1:] # Skip batch size
                    layer_feature = np.array([layer_feature])
                    print("Misc Layer Feature:", layer_feature)
                    layer_latency = self.misc_proxy(
                        self.mico_target, layer_name, layer_feature)[0]
                    print("Misc Layer Latency:", layer_latency)
                    self.misc_latency += layer_latency
        print("Misc Latency:", self.misc_latency)
        return
    
    def set_mico_target(self, mico_type: str):
        self.mico_target = mico_type

    def eval_fp(self):
        self.model.unset_qscheme()
        return self.model.test(self.test_loader)['TestAcc']

    def eval_ptq_loss(self, scheme: list):
        wq = scheme[:self.n_layers]
        aq = scheme[self.n_layers:]
        # self.load_pretrain()
        self.model.set_qscheme([wq, aq], group_size=self.group_size)
        return self.model.test(self.test_loader)['TestLoss']

    def eval_ptq(self, scheme: list):
        wq = scheme[:self.n_layers]
        aq = scheme[self.n_layers:]
        # self.load_pretrain()
        self.model.set_qscheme([wq, aq], group_size=self.group_size)
        return self.model.test(self.test_loader)['TestAcc']
    
    def eval_qat(self, scheme: list):
        wq = scheme[:self.n_layers]
        aq = scheme[self.n_layers:]
        self.load_pretrain()
        self.model.set_qscheme([wq, aq], qat=True, group_size=self.group_size)
        self.model.train_loop(self.epochs, self.train_loader, self.test_loader, 
                              verbose=True, lr=self.lr)
        return self.model.test(self.test_loader)['TestAcc']
    
    def eval_torchao(self, scheme: list):
        wq = scheme[:self.n_layers]
        aq = scheme[self.n_layers:]
        # self.load_pretrain()
        self.model.set_qscheme_torchao([wq, aq])
        return self.model.test(self.test_loader)['TestAcc']

    # Classic Bit Operation Counts
    def eval_bops(self, scheme: list):
        bops = 0
        wq = np.array(scheme[:self.n_layers])
        aq = np.array(scheme[self.n_layers:])
        bops = np.dot(wq*aq, self.layer_macs)
        return bops
    
    def eval_cbops(self, scheme: list):
        cbops = 0
        wq = np.array(scheme[:self.n_layers])
        aq = np.array(scheme[self.n_layers:])
        max_q = np.max([aq, wq], axis=0)
        load_q = aq + wq
        cbops = np.dot(max_q+load_q, self.layer_macs)
        return cbops

    def eval_pred_latency(self, scheme: list):

        pred_latency = 0
        wq = np.array(scheme[:self.n_layers])
        aq = np.array(scheme[self.n_layers:])

        for i in range(self.n_layers):
            # layer_features = self.layers[i].layer_features
            layer_macs = self.layer_macs[i]
            layer_bmacs = layer_macs * np.max([aq[i], wq[i]])
            layer_wloads = wq[i] * layer_macs
            layer_aloads = aq[i] * layer_macs
            # layer_features = (layer_bmacs, layer_wloads, layer_aloads, wq[i], aq[i])
            layer_features = (layer_bmacs, layer_wloads, layer_aloads)
            layer_features = np.array([layer_features])
            # layer_features = np.array([layer_bmacs, layer_wloads, layer_aloads]).reshape(1, -1)
            if self.layers[i].layer_type == 'Conv2D':
                pred_latency += self.conv2d_proxy.predict(layer_features)[0]
            elif self.layers[i].layer_type == 'Linear':
                pred_latency += self.matmul_proxy.predict(layer_features)[0]
        return pred_latency + self.misc_latency

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
            gen_model.set_qscheme([wq, aq], group_size=self.group_size)
            codegen = MiCoCodeGen(gen_model)
            self.input_size = self.train_loader.dataset[0][0].shape
            example_input = torch.randn(1, *self.input_size).to(DEVICE)
            codegen.forward(example_input)
            codegen.convert()
            codegen.build(target="mico_fpu")

            # if self.mico_target == "high": 
                # codegen.build(target="mico_fpu")
            # else:
            #     codegen.build(target="mico")
            
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

