import torch
from torchvision.models import get_model
from torch.utils.data.dataloader import DataLoader
from torch import nn

import numpy as np

from matplotlib import pyplot as plt

import os
import json
import csv
import time
import warnings
import tempfile
from contextlib import contextmanager

try:
    import fcntl
except ImportError:
    fcntl = None

from tqdm import tqdm

from MiCoModel import MiCoModel, from_torch
from MiCoPasses import normalize_compression_order
from MiCoQLayers import BitConv1d, BitConv2d, BitLinear
from MiCoCodeGen import MiCoCodeGen
from MiCoUtils import fuse_model
from MiCoPruning import (
    AttentionPruningSpec,
    PruningSpec,
    apply_attention_pruning,
    apply_pruning,
    build_attention_pruning_plan,
    build_pruning_plan,
    discover_attention_layers,
)
from SimUtils import gen_sim_bitfusion, sim_mico, run_host

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
                 linear_group_size = 0,
                 output_json='output/json/mico_eval.json',
                  pruning_mode="none",
                  pruning_types=None,
                  attention_pruning_mode="none",
                  attention_pruning_types=None,
                  order="prune_then_quantize") -> None:
        
        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.pretrained_model = pretrained_model
        self.group_size = linear_group_size
        self.lr = lr
        self.pruning_mode = pruning_mode
        self.pruning_types = list(pruning_types or [0.0, 0.25, 0.5, 0.75])
        self.attention_pruning_mode = attention_pruning_mode
        self.attention_pruning_types = list(
            attention_pruning_types or [0.0, 0.25, 0.5, 0.75]
        )
        self.compression_order = normalize_compression_order(order)

        self.from_torch = isinstance(model, str)  # If model is a string, load from torchvision

        self.load_pretrain()
        self.base_model = deepcopy(self.model)
        self.pruning_dim = (
            self.model.n_layers if pruning_mode != "none" else 0
        ) if hasattr(self.model, "n_layers") else 0
        self.n_attn_pruning_layers = (
            len(discover_attention_layers(self.base_model))
            if attention_pruning_mode != "none"
            else 0
        )
        self.attention_pruning_dim = self.n_attn_pruning_layers
        self.pruning_choices_by_dim = []
        if self.pruning_dim:
            pruning_choices = list(self.pruning_types)
            if 0.0 not in pruning_choices:
                pruning_choices.append(0.0)
            self.pruning_choices_by_dim = [list(pruning_choices) for _ in range(self.pruning_dim)]
            # The structured channel pruner protects the final classifier output.
            self.pruning_choices_by_dim[-1] = [0.0]
        attention_choices = list(self.attention_pruning_types)
        if 0.0 not in attention_choices:
            attention_choices.append(0.0)
        self.attention_pruning_choices_by_dim = [
            list(attention_choices)
            for _ in range(self.attention_pruning_dim)
        ]
        if self.pruning_dim and pruning_mode not in {"structured"}:
            raise ValueError("MiCoEval only supports structured pruning mode")
        if self.attention_pruning_dim and attention_pruning_mode not in {"heads"}:
            raise ValueError("MiCoEval only supports attention heads pruning mode")
        print("Pretrained Model Loaded.")

        self.output_json = output_json
        self.cache_format = 'csv' if str(self.output_json).lower().endswith('.csv') else 'json'
        self.model_name = model_name
        self.n_layers = self.model.n_layers
        self.dim = self.n_layers * 2 + self.pruning_dim + self.attention_pruning_dim
        
        self.data_trace = self._load_data_trace()
        
        self.set_eval(objective)
        self.set_constraint(constraint)

        self.fp_res = self.eval_fp()
        self.fp_acc = self.eval_fp()['TestAcc']
        self.fp_loss = self.eval_fp()['TestLoss']
        # Initial Conversion and Test
        res = self.eval_f([8]*self.n_layers*2)
        self.baseline_acc = res
        self.layers = self.model.get_qlayers()
        assert self.n_layers > 0, "No QLayer found! Please convert the model first."
        self.layer_paths = [
            path for path, layer in self.model.named_modules()
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear))
        ]
        if len(self.layer_paths) != self.n_layers:
            raise ValueError("quantizable layer path traversal does not match n_layers")
        self.layer_macs = [layer.get_mac() for layer in self.layers]
        self.layer_params = [layer.get_params() for layer in self.layers]
        self.layer_type = [type(layer) for layer in self.layers]
        self.layer_features = [layer.layer_features for layer in self.layers]

        # A Regression Model for Hardware Latency
        self.mico_target = "small"
        self.misc_latency = 0.0
        self.misc_proxy = None

        # Proxy Calibration
        self.intercept = 0.0
        self.slope = 1.0

        print("Model:", self.model._get_name())
        print("Number of QLayers: ", self.n_layers)
        print("Total MACs: ", np.sum(self.layer_macs))
        print("Total Params: ", np.sum(self.layer_params))
        print("INT8 Model Accuracy: ", res)
        print("FP Model Accuracy: ", self.fp_acc)
        print("FP Model Loss:", self.fp_loss)
        return
    
    def get_layer_info(self):
        layer_info = []
        for i in range(self.n_layers):
            info = {
                "Layer Index": i,
                "Layer Type": str(self.layer_type[i]),
                "Layer MACs": self.layer_macs[i],
                "Layer Params": self.layer_params[i],
                "Layer Features": self.layer_features[i]
            }
            layer_info.append(info)
        return layer_info

    def _scheme_to_key(self, scheme: list):
        if not (getattr(self, "pruning_dim", 0) or getattr(self, "attention_pruning_dim", 0)):
            return ",".join(str(int(x)) for x in scheme)
        return ",".join(f"{float(x):.12g}" for x in scheme)

    def _cache_lock_path(self):
        return f"{self.output_json}.lock"

    @contextmanager
    def _cache_lock(self, exclusive: bool):
        lock_path = self._cache_lock_path()
        lock_dir = os.path.dirname(lock_path)
        if lock_dir:
            os.makedirs(lock_dir, exist_ok=True)
        with open(lock_path, "a+") as lock_file:
            if fcntl is not None:
                lock_mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                fcntl.flock(lock_file.fileno(), lock_mode)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load_data_trace_unlocked(self):
        if not os.path.exists(self.output_json):
            return {}
        if self.cache_format == 'csv':
            data_trace = {}
            with open(self.output_json, 'r', newline='') as f:
                for row in csv.DictReader(f):
                    scheme = row.get("scheme")
                    objective = row.get("objective")
                    value = row.get("value")
                    if scheme is None or objective is None or value is None:
                        continue
                    data_trace.setdefault(scheme, {})[objective] = float(value)
            return data_trace
        with open(self.output_json, 'r') as f:
            return json.load(f)

    def _save_data_trace_unlocked(self, data_trace: dict):
        output_dir = os.path.dirname(self.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        temp_dir = output_dir if output_dir else "."
        fd, tmp_path = tempfile.mkstemp(prefix=".micoeval_", suffix=".tmp", dir=temp_dir)
        try:
            if self.cache_format == 'csv':
                with os.fdopen(fd, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["scheme", "objective", "value"])
                    for scheme, objectives in data_trace.items():
                        for objective, value in objectives.items():
                            writer.writerow([scheme, objective, value])
            else:
                with os.fdopen(fd, 'w') as f:
                    json.dump(data_trace, f, indent=4)
            os.replace(tmp_path, self.output_json)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _load_data_trace(self):
        with self._cache_lock(exclusive=False):
            return self._load_data_trace_unlocked()

    def _save_data_trace(self):
        with self._cache_lock(exclusive=True):
            self._save_data_trace_unlocked(self.data_trace)

    def _lookup_cached_result(self, data_trace: dict, cache_key: str, legacy_key: str):
        if cache_key in data_trace:
            return data_trace[cache_key].get(self.objective), False
        legacy_values = data_trace.get(legacy_key)
        if legacy_values is None:
            return None, False
        data_trace[cache_key] = data_trace.pop(legacy_key, {})
        return data_trace[cache_key].get(self.objective), True

    def eval(self, scheme: list, offline: bool = False, use_cache: bool = True):
        if not use_cache:
            return self.eval_f(scheme)

        cache_key = self._scheme_to_key(scheme)
        legacy_key = str(scheme)
        with self._cache_lock(exclusive=True):
            current_trace = self._load_data_trace_unlocked()
            res, migrated = self._lookup_cached_result(current_trace, cache_key, legacy_key)
            if migrated:
                self._save_data_trace_unlocked(current_trace)
            self.data_trace = current_trace

        if res is None and offline:
            raise ValueError(f"Missing cached result for scheme {scheme} (key={cache_key}) objective [{self.objective}] in {self.output_json}")
        if res is not None:
            return res

        res = self.eval_f(scheme)

        with self._cache_lock(exclusive=True):
            current_trace = self._load_data_trace_unlocked()
            cached_res, migrated = self._lookup_cached_result(current_trace, cache_key, legacy_key)
            if cached_res is not None:
                if migrated:
                    self._save_data_trace_unlocked(current_trace)
                self.data_trace = current_trace
                return cached_res

            current_trace.setdefault(cache_key, {})[self.objective] = res
            self._save_data_trace_unlocked(current_trace)
            self.data_trace = current_trace

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
            'cbops': self.eval_cbops,
            'size': self.eval_size,
            'latency_host': lambda scheme: self.eval_latency(scheme, 'host'),
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
            self.model = self.model.to(DEVICE)
            ckpt = torch.load(self.pretrained_model, weights_only=False, map_location=DEVICE)
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
            if ("bitconv2d" in forward_pass) or \
                ("bitlinear" in forward_pass) or \
                ('CONNECT' in forward_pass):
                continue
            else:
                layer_name = forward_pass.split('(')[0]
                if layer_name in ignored_misc:
                    continue
                else:
                    layer_name = layer_name.replace('MiCo_', '')
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
                        # TODO: Include Padding Size
                        layer_feature = layer_feature[:-1] # Skip padding size
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

    def _split_scheme(self, scheme: list):
        values = list(scheme)
        if len(values) == self.n_layers * 2 and self.dim != self.n_layers * 2:
            values.extend([0.0] * (self.dim - self.n_layers * 2))
        if len(values) != self.dim:
            raise ValueError(
                f"expected compression scheme length {self.dim}, got {len(values)}"
            )
        weight = values[:self.n_layers]
        activation = values[self.n_layers:self.n_layers * 2]
        cursor = self.n_layers * 2
        pruning = values[cursor:cursor + self.pruning_dim]
        cursor += self.pruning_dim
        attention_pruning = values[cursor:cursor + self.attention_pruning_dim]
        return weight, activation, pruning, attention_pruning

    def _candidate_from_scheme(self, scheme: list, *, qat=False, quantize=True):
        weight, activation, pruning, attention_pruning = self._split_scheme(scheme)
        candidate = deepcopy(self.base_model)
        pruning_result = None
        attention_result = None
        already_quantized = False

        if quantize and (self.pruning_dim or self.attention_pruning_dim):
            result = candidate.set_compression_scheme(
                qscheme=[weight, activation],
                pscheme=pruning if self.pruning_dim else None,
                pruning_mode=self.pruning_mode,
                attention_pscheme=attention_pruning if self.attention_pruning_dim else None,
                attention_pruning_mode=self.attention_pruning_mode,
                order=self.compression_order,
                qat=qat,
                group_size=self.group_size,
            )
            already_quantized = True
            if isinstance(result, dict):
                pruning_result = result.get("pruning")
                attention_result = result.get("attention_pruning")
            elif self.pruning_dim:
                pruning_result = result
            else:
                attention_result = result
        else:
            if self.pruning_dim:
                candidate, pruning_result = apply_pruning(
                    candidate,
                    PruningSpec(ratios=pruning, mode=self.pruning_mode),
                )
            if self.attention_pruning_dim:
                candidate, attention_result = apply_attention_pruning(
                    candidate,
                    AttentionPruningSpec(
                        ratios=attention_pruning,
                        mode=self.attention_pruning_mode,
                    ),
                )
        return (
            candidate,
            weight,
            activation,
            pruning_result,
            attention_result,
            already_quantized,
        )

    def eval_fp(self):
        if self.pruning_dim or self.attention_pruning_dim:
            self.model = deepcopy(self.base_model)
        else:
            self.model.unset_qscheme()
        return self.model.test(self.test_loader)

    def eval_ptq_loss(self, scheme: list):
        candidate, wq, aq, _, _, already_quantized = self._candidate_from_scheme(scheme)
        if not already_quantized:
            candidate.set_qscheme([wq, aq], group_size=self.group_size)
        self.model = candidate
        return self.model.test(self.test_loader)['TestLoss']

    def eval_ptq(self, scheme: list):
        candidate, wq, aq, _, _, already_quantized = self._candidate_from_scheme(scheme)
        if not already_quantized:
            candidate.set_qscheme([wq, aq], group_size=self.group_size)
        self.model = candidate
        return self.model.test(self.test_loader)['TestAcc']
    
    def eval_qat(self, scheme: list, epochs: int = None):
        if epochs is None:
            epochs = self.epochs
        candidate, wq, aq, _, _, already_quantized = self._candidate_from_scheme(
            scheme, qat=True
        )
        if not already_quantized:
            candidate.set_qscheme([wq, aq], qat=True, group_size=self.group_size)
        self.model = candidate
        self.model.train_loop(epochs, self.train_loader, self.test_loader,
                              verbose=True, lr=self.lr)
        return self.model.test(self.test_loader)['TestAcc']
    
    def eval_torchao(self, scheme: list):
        if self.compression_order == "quantize_then_prune" and (
            self.pruning_dim or self.attention_pruning_dim
        ):
            raise ValueError(
                "TorchAO does not expose MiCo Qweight for quantize_then_prune"
            )
        candidate, wq, aq, _, _, _ = self._candidate_from_scheme(
            scheme, quantize=False
        )
        candidate.set_qscheme_torchao([wq, aq])
        self.model = candidate
        return self.model.test(self.test_loader)['TestAcc']

    def _candidate_costs(self, scheme: list):
        """Estimate costs after optional channel/head materialization.

        Cost evaluation is kept analytical so near-constraint sampling does not
        deep-copy, quantize, and run hundreds of temporary models.  The same
        shape ratios are used by the materializers and are exact for dense
        Conv/Linear MAC and weight counts.
        """
        if not (self.pruning_dim or self.attention_pruning_dim):
            return np.array(self.layer_macs, dtype=float), np.array(self.layer_params, dtype=float), self.layers
        cache_key = self._scheme_to_key(scheme)
        cached = getattr(self, "_cost_cache", {}).get(cache_key)
        if cached is not None:
            return cached

        macs = np.array(self.layer_macs, dtype=float)
        params = np.array(self.layer_params, dtype=float)
        path_to_index = {path: index for index, path in enumerate(self.layer_paths)}
        base_modules = dict(self.base_model.named_modules())

        if self.pruning_dim:
            pruning = self._split_scheme(scheme)[2]
            plan = build_pruning_plan(
                self.base_model,
                PruningSpec(ratios=pruning, mode=self.pruning_mode),
            )
            for layer in plan.layers:
                index = path_to_index[layer.path]
                input_size, output_size = layer.module.weight.shape[1], layer.module.weight.shape[0]
                input_ratio = len(layer.input_indices) / input_size
                output_ratio = len(layer.output_indices) / output_size
                macs[index] *= input_ratio * output_ratio
                params[index] = layer.module.weight.numel() * input_ratio * output_ratio

        if self.attention_pruning_dim:
            attention = self._split_scheme(scheme)[3]
            plan = build_attention_pruning_plan(
                self.base_model,
                AttentionPruningSpec(ratios=attention, mode=self.attention_pruning_mode),
            )
            for layer in plan.layers:
                layout_modules = {
                    "q": base_modules[layer.q_path],
                    "k": base_modules[layer.k_path],
                    "v": base_modules[layer.v_path],
                    "output": base_modules[layer.output_path],
                }
                projection_data = (
                    (layer.q_path, layout_modules["q"], len(layer.query_heads) * layer.head_dim,
                     layout_modules["q"].out_features),
                    (layer.k_path, layout_modules["k"], len(layer.kv_heads) * layer.head_dim,
                     layout_modules["k"].out_features),
                    (layer.v_path, layout_modules["v"], len(layer.kv_heads) * layer.head_dim,
                     layout_modules["v"].out_features),
                    (layer.output_path, layout_modules["output"], len(layer.query_heads) * layer.head_dim,
                     layout_modules["output"].in_features),
                )
                for path, module, kept_width, original_width in projection_data:
                    index = path_to_index[path]
                    ratio = kept_width / original_width
                    macs[index] *= ratio
                    params[index] = module.weight.numel() * ratio

        result = (macs, params, self.layers)
        if not hasattr(self, "_cost_cache"):
            self._cost_cache = {}
        self._cost_cache[cache_key] = result
        return result

    # Classic Bit Operation Counts
    def eval_bops(self, scheme: list):
        wq, aq, _, _ = self._split_scheme(scheme)
        macs, _, _ = self._candidate_costs(scheme)
        return float(np.dot(np.array(wq) * np.array(aq), macs))
    
    def eval_cbops(self, scheme: list):
        return float(np.sum(self.eval_cbops_feat(scheme)))

    def eval_cbops_feat(self, scheme: list, layer_type=None):
        """Return aggregated 3D CBOPs features: [BMACS, W_LOADS, A_LOADS]."""
        wq, aq, _, _ = self._split_scheme(scheme)
        wq = np.array(wq)
        aq = np.array(aq)
        macs, _, layers = self._candidate_costs(scheme)

        if layer_type is not None:
            mask = np.array([isinstance(layer, layer_type) for layer in layers])
            wq = wq[mask]
            aq = aq[mask]
            macs = macs[mask]

        q_max = np.maximum(aq, wq)
        bmacs = np.dot(macs, q_max)
        w_loads = np.dot(macs, wq)
        a_loads = np.dot(macs, aq)
        return np.array([bmacs, w_loads, a_loads], dtype=float)

    def eval_pred_latency(self, scheme: list):

        wq, aq, _, _ = self._split_scheme(scheme)
        wq = np.array(wq)
        aq = np.array(aq)
        macs, _, layers = self._candidate_costs(scheme)

        # Batch layers by type for vectorized prediction
        conv2d_features = []
        linear_features = []
        for i in range(self.n_layers):
            if isinstance(layers[i], BitConv2d):
                feat = self._proxy_layer_feature(
                    self.conv2d_proxy, i, aq[i], wq[i], layers=layers, macs=macs
                )
                conv2d_features.append(feat)
            elif isinstance(layers[i], BitLinear):
                feat = self._proxy_layer_feature(
                    self.matmul_proxy, i, aq[i], wq[i], layers=layers, macs=macs
                )
                linear_features.append(feat)

        pred_latency = 0.0
        if conv2d_features:
            pred_latency += np.sum(self.conv2d_proxy.predict(np.array(conv2d_features)))
        if linear_features:
            pred_latency += np.sum(self.matmul_proxy.predict(np.array(linear_features)))
        pred_latency += self.misc_latency
        pred_latency = self.slope * pred_latency + self.intercept
        return pred_latency

    def _proxy_layer_feature(self, proxy, layer_idx: int, aq, wq, layers=None, macs=None):
        source_layers = self.layers if layers is None else layers
        layer_macs = source_layers[layer_idx].get_mac() if macs is None else macs[layer_idx]
        if self._proxy_accepts_macs_bitwidth_only(proxy):
            return [layer_macs, aq, wq]
        return [layer_macs] + source_layers[layer_idx].layer_features + [aq, wq]

    def _proxy_accepts_macs_bitwidth_only(self, proxy) -> bool:
        preprocess_name = getattr(proxy, "preprocess_name", None)
        return preprocess_name in {"macs+", "bops", "sumq", "maxq", "cbops"}

    def calibrate_proxy(self, min_q: int = 2, max_q: int = 8, target: str = 'bitfusion'):
        
        print("Calibrating Proxy...")

        min_pred = self.eval_pred_latency([min_q] * self.n_layers * 2)
        min_real = self.eval_latency([min_q] * self.n_layers * 2, target=target)
        print("Min Predicted Latency:", min_pred)
        print("Min Real Latency:", min_real)

        max_pred = self.eval_pred_latency([max_q] * self.n_layers * 2)
        max_real = self.eval_latency([max_q] * self.n_layers * 2, target=target)
        print("Max Predicted Latency:", max_pred)
        print("Max Real Latency:", max_real)

        # Linear Calibration
        self.slope = (max_real - min_real) / (max_pred - min_pred)
        self.intercept = min_real - self.slope * min_pred
        
        return min_real, max_real

    def eval_size(self, scheme:list):
        wq, _, _, _ = self._split_scheme(scheme)
        _, params, _ = self._candidate_costs(scheme)
        return float(np.dot(np.array(wq), params))

    def eval_latency(self, scheme: list, target: str = 'bitfusion'):

        if target == "torchao":
            if self.compression_order == "quantize_then_prune" and (
                self.pruning_dim or self.attention_pruning_dim
            ):
                raise ValueError(
                    "TorchAO does not expose MiCo Qweight for quantize_then_prune"
                )
            candidate, wq, aq, _, _, already_quantized = self._candidate_from_scheme(
                scheme, quantize=False
            )
        else:
            candidate, wq, aq, _, _, already_quantized = self._candidate_from_scheme(scheme)
        gen_model = candidate
        res = -1
        if target == 'bitfusion':
            if not already_quantized:
                gen_model.set_qscheme([wq, aq], group_size=self.group_size)
            self.input_size = self.train_loader.dataset[0][0].shape
            example_input = torch.randn(1, *self.input_size).to(DEVICE)
            res = gen_sim_bitfusion(gen_model, 16, example_input) # Default Batch Size = 16
        elif target == 'mico':
            assert self.mico_target is not None, "MiCo Target not set."
            gen_model = fuse_model(gen_model)
            if not already_quantized:
                gen_model.set_qscheme([wq, aq], group_size=self.group_size)
            codegen = MiCoCodeGen(gen_model)
            self.input_size = self.train_loader.dataset[0][0].shape
            example_input = torch.randn(1, *self.input_size).to(DEVICE)
            codegen.forward(example_input)
            codegen.convert()
            codegen.build(target="mico_fpu")
            
            res = sim_mico(self.mico_target)

        elif target == 'host':
            gen_model = fuse_model(gen_model)
            if not already_quantized:
                gen_model.set_qscheme([wq, aq], group_size=self.group_size)
            codegen = MiCoCodeGen(gen_model)
            self.input_size = self.train_loader.dataset[0][0].shape
            example_input = torch.randn(1, *self.input_size).to(DEVICE)
            codegen.forward(example_input)
            codegen.convert()
            codegen.build(target="host", options="OPT=lut")
            res = run_host()
            
        elif target == 'proxy':
            res = self.eval_pred_latency(scheme)
        elif target == 'torchao':
            gen_model.set_qscheme_torchao([wq, aq])
            start_time = time.time()
            gen_model.test(self.test_loader)
            end_time = time.time()
            res = end_time - start_time
        else:
            raise ValueError(f"Target \"{target}\" not supported.")
        if res is None:
            raise ValueError(f"Error in simulating the model on {target}.")
        return res
    
# Test
if __name__ == "__main__":
    from models import MLP, LeNet, VGG, resnet_alt_8
    from MiCoDatasets import mnist, cifar10

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

    # model = MLP(256, config={"Layers": [64, 64, 64, 10]})
    # train_loader, test_loader = mnist(shuffle=False, resize=16)
    # evaluator = MiCoEval(model, 10, train_loader, test_loader, 
    #                     "output/ckpt/mlp_mnist.pth", model_name="mlp")

    model = resnet_alt_8(10)
    train_loader, test_loader = cifar10(shuffle=False)
    model.default_dataset = "CIFAR10"

    evaluator = MiCoEval(model, 10, train_loader, test_loader, 
                         "output/ckpt/resnet8_cifar10.pth", model_name="resnet8")
    
    res = evaluator.eval_latency([8]*model.n_layers*2, 'host')
    print("Latency:", res)
