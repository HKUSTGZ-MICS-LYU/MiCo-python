import json
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from gekko import GEKKO
from tqdm import tqdm

from MiCoEval import MiCoEval
from MiCoUtils import list_qlayers
from searchers.QSearcher import QSearcher


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HAWQSearcher(QSearcher):
    """HAWQ-style Hessian trace weighted ILP searcher for MiCo.

    This ports the HAWQ-V3 ILP selection idea into MiCo's Searcher API.
    The v1 scope is tied W/A precision per layer, matching HAWQ's INT4/8
    mixed-precision setting while using MiCo's own quantization semantics.
    """

    def __init__(
        self,
        evaluator: MiCoEval,
        n_inits: int = 10,
        qtypes: list = None,
        trace_samples: int = 8,
        max_batches: int = 4,
        cache_path: str = None,
        force_refresh: bool = False,
        fix_first_last: bool = True,
        first_last_bit: int = None,
    ) -> None:
        qtypes = qtypes or [4, 5, 6, 7, 8]
        super().__init__(evaluator, n_inits, qtypes)
        self.qbits = sorted({int(q) for q in qtypes})
        if min(self.qbits) < 1:
            raise ValueError("HAWQSearcher only supports integer qtypes >= 1")
        self.trace_samples = max(1, int(trace_samples))
        self.max_batches = None if max_batches is None else max(1, int(max_batches))
        self.cache_path = cache_path
        self.force_refresh = force_refresh
        self.fix_first_last = bool(fix_first_last)
        self.first_last_bit = max(self.qbits) if first_last_bit is None else int(first_last_bit)
        if self.first_last_bit not in self.qbits:
            raise ValueError(f"first_last_bit {self.first_last_bit} must be in qtypes {self.qbits}")
        self.device = self._select_device()
        self.trace = None
        self.delta_table = None
        self.sensitivity_table = None
        self.cost_table = None
        self._using_shared_model = False

    def _select_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(self.evaluator.model, "current_device"):
            return self.evaluator.model.current_device()
        return DEVICE

    def _cache_key(self):
        return {
            "n_layers": int(self.n_layers),
            "qbits": [int(q) for q in self.qbits],
            "trace_samples": int(self.trace_samples),
            "max_batches": self.max_batches,
            "group_size": int(getattr(self.evaluator, "group_size", 1)),
            "pretrained_model": str(getattr(self.evaluator, "pretrained_model", "")),
        }

    def _load_cache(self):
        if self.force_refresh or not self.cache_path or not os.path.exists(self.cache_path):
            return False
        with open(self.cache_path, "r") as f:
            data = json.load(f)
        if data.get("cache_key") != self._cache_key():
            print("HAWQ cache key mismatch. Recomputing Hessian trace and delta table.")
            return False
        self.trace = np.array(data["trace"], dtype=float)
        self.delta_table = np.array(data["delta_table"], dtype=float)
        self.sensitivity_table = np.array(data["sensitivity_table"], dtype=float)
        print(f"Loaded HAWQ cache: {self.cache_path}")
        return True

    def _save_cache(self):
        if not self.cache_path:
            return
        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        data = {
            "cache_key": self._cache_key(),
            "trace": self.trace.tolist(),
            "delta_table": self.delta_table.tolist(),
            "sensitivity_table": self.sensitivity_table.tolist(),
        }
        with open(self.cache_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved HAWQ cache: {self.cache_path}")

    def _clone_evaluator_model(self):
        runtime_attrs = {}
        for attr in ("train_loader", "test_loader"):
            if hasattr(self.evaluator.model, attr):
                runtime_attrs[attr] = getattr(self.evaluator.model, attr)
                setattr(self.evaluator.model, attr, None)
        try:
            return deepcopy(self.evaluator.model)
        except TypeError as exc:
            if "pickle" not in str(exc):
                raise
            print(
                "HAWQ warning: model deepcopy failed due to a non-picklable "
                "runtime object; using evaluator.model for sensitivity extraction."
            )
            self._using_shared_model = True
            return self.evaluator.model
        finally:
            for attr, value in runtime_attrs.items():
                setattr(self.evaluator.model, attr, value)

    def _disable_flash_attention_for_hessian(self, model):
        original_flash = []
        for module in model.modules():
            if hasattr(module, "flash") and isinstance(module.flash, bool) and module.flash:
                original_flash.append((module, module.flash))
                module.flash = False

        def restore_flash():
            for module, value in original_flash:
                module.flash = value

        return restore_flash

    def _prepare_model(self):
        self.evaluator.load_pretrain()
        self._using_shared_model = False
        model = self._clone_evaluator_model().to(self.device)
        if hasattr(model, "unset_qscheme"):
            model.unset_qscheme()
        restore_flash = self._disable_flash_attention_for_hessian(model)
        layers = list_qlayers(model)
        if len(layers) != self.n_layers:
            raise ValueError(f"Expected {self.n_layers} qlayers, got {len(layers)}")
        original_requires_grad = [param.requires_grad for param in model.parameters()]
        for param in model.parameters():
            param.requires_grad = False
        for layer in layers:
            layer.weight.requires_grad = True

        def restore_requires_grad():
            if not self._using_shared_model:
                restore_flash()
                return
            for param, requires_grad in zip(model.parameters(), original_requires_grad):
                param.requires_grad = requires_grad
            restore_flash()

        return model, layers, restore_requires_grad

    def _batch_loss(self, model, x, y):
        try:
            output = model(x, y)
        except TypeError:
            output = model(x)

        loss = getattr(model, "last_loss", None)
        if loss is not None:
            return loss

        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.dim() >= 3 and y.dim() >= 2 and tuple(output.shape[:-1]) == tuple(y.shape):
            return F.cross_entropy(
                output.reshape(-1, output.size(-1)),
                y.reshape(-1),
                ignore_index=-1,
            )
        return F.cross_entropy(output, y)

    def _compute_hessian_trace(self, model, layers):
        weights = [layer.weight for layer in layers]
        trace = torch.zeros(len(weights), device=self.device)
        data_iter = iter(self.evaluator.train_loader)
        used = 0

        model.train()
        pbar = tqdm(range(self.trace_samples), desc="HAWQ Hessian trace", leave=False)
        for _ in pbar:
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(self.evaluator.train_loader)
                images, labels = next(data_iter)

            if self.max_batches is not None and used >= self.max_batches:
                data_iter = iter(self.evaluator.train_loader)
                used = 0
                images, labels = next(data_iter)

            x = images.to(self.device)
            y = labels.to(self.device)
            model.zero_grad(set_to_none=True)
            loss = self._batch_loss(model, x, y)
            grads = torch.autograd.grad(loss, weights, create_graph=True, allow_unused=True)
            vectors = [
                torch.randint_like(w, low=0, high=2, dtype=w.dtype).mul_(2).sub_(1)
                for w in weights
            ]
            grad_vector = torch.tensor(0.0, device=self.device)
            for grad, vector in zip(grads, vectors):
                if grad is not None:
                    grad_vector = grad_vector + torch.sum(grad * vector)
            hvps = torch.autograd.grad(grad_vector, weights, retain_graph=False, allow_unused=True)
            for idx, (hvp, vector, weight) in enumerate(zip(hvps, vectors, weights)):
                if hvp is not None:
                    trace[idx] += torch.sum(hvp * vector).detach() / max(weight.numel(), 1)
            used += 1
            pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

        trace = (trace / float(self.trace_samples)).abs().detach().cpu().numpy()
        return trace

    def _compute_delta_table(self, layers):
        delta = np.zeros((len(layers), len(self.qbits)), dtype=float)
        original_qtypes = [layer.qtype for layer in layers]
        try:
            with torch.no_grad():
                for layer_idx, layer in enumerate(layers):
                    weight = layer.weight.detach()
                    for bit_idx, bit in enumerate(self.qbits):
                        layer.qtype = bit
                        qweight = layer.weight_quant(weight)
                        delta[layer_idx, bit_idx] = float(torch.sum((weight - qweight) ** 2).cpu())
        finally:
            for layer, qtype in zip(layers, original_qtypes):
                layer.qtype = qtype
        return delta

    def _ensure_sensitivity(self):
        if self._load_cache():
            return
        model, layers, restore_requires_grad = self._prepare_model()
        print(
            f"Computing HAWQ sensitivity: trace_samples={self.trace_samples}, "
            f"max_batches={self.max_batches}, qbits={self.qbits}, device={self.device}"
        )
        try:
            self.trace = self._compute_hessian_trace(model, layers)
            self.delta_table = self._compute_delta_table(layers)
            self.sensitivity_table = self.trace[:, None] * self.delta_table
            print("HAWQ Hessian trace:", self.trace.tolist())
            print("HAWQ sensitivity table shape:", self.sensitivity_table.shape)
            self._save_cache()
        finally:
            restore_requires_grad()

    def _scheme_for_layer_bits(self, layer_bits):
        bits = [int(q) for q in layer_bits]
        return bits + bits.copy()

    def _build_cost_table(self, constr):
        table = np.zeros((self.n_layers, len(self.qbits)), dtype=float)
        for layer_idx in range(self.n_layers):
            for bit_idx, bit in enumerate(self.qbits):
                if constr == "bops":
                    table[layer_idx, bit_idx] = float(self.evaluator.layer_macs[layer_idx]) * bit * bit
                elif constr == "size":
                    table[layer_idx, bit_idx] = float(self.evaluator.layer_params[layer_idx]) * bit
                else:
                    scheme = self._scheme_for_layer_bits(
                        [bit if i == layer_idx else min(self.qbits) for i in range(self.n_layers)]
                    )
                    base = self._scheme_for_layer_bits([min(self.qbits)] * self.n_layers)
                    table[layer_idx, bit_idx] = float(self.evaluator.constr(scheme) - self.evaluator.constr(base))
        if constr not in ["bops", "size"]:
            min_scheme = self._scheme_for_layer_bits([min(self.qbits)] * self.n_layers)
            min_cost = float(self.evaluator.constr(min_scheme))
            table += min_cost / max(self.n_layers, 1)
        return table

    def _solve_ilp(self, constr_value):
        if constr_value is None:
            raise ValueError("constr_value must be provided for HAWQ search.")
        min_total_cost = float(np.sum(np.min(self.cost_table, axis=1)))
        if min_total_cost > constr_value:
            raise ValueError("Constraint is tighter than the minimum HAWQ qtype scheme.")

        m = GEKKO(remote=False)
        picks = []
        for _ in range(self.n_layers):
            layer_picks = [m.Var(lb=0, ub=1, integer=True) for _ in self.qbits]
            m.Equation(m.sum(layer_picks) == 1)
            picks.append(layer_picks)
        if self.fix_first_last and self.n_layers >= 1:
            fixed_idx = self.qbits.index(self.first_last_bit)
            fixed_layers = [0]
            if self.n_layers > 1:
                fixed_layers.append(self.n_layers - 1)
            for layer_idx in fixed_layers:
                for bit_idx, pick in enumerate(picks[layer_idx]):
                    m.Equation(pick == (1 if bit_idx == fixed_idx else 0))

        m.Equation(
            m.sum([
                picks[layer_idx][bit_idx] * float(self.cost_table[layer_idx, bit_idx])
                for layer_idx in range(self.n_layers)
                for bit_idx in range(len(self.qbits))
            ]) <= float(constr_value)
        )

        max_sensitivity = float(np.max(np.abs(self.sensitivity_table)))
        if max_sensitivity <= 0:
            max_sensitivity = 1.0
        objective = m.sum([
            picks[layer_idx][bit_idx] * float(self.sensitivity_table[layer_idx, bit_idx] / max_sensitivity)
            for layer_idx in range(self.n_layers)
            for bit_idx in range(len(self.qbits))
        ])
        m.Obj(objective)
        m.options.IMODE = 3
        m.options.SOLVER = 1

        print("Solving HAWQ ILP with constraint:", constr_value)
        m.solve(disp=True)

        layer_bits = []
        for layer_picks in picks:
            values = [float(p.value[0]) for p in layer_picks]
            idx = max(range(len(values)), key=lambda i: values[i])
            if values[idx] < 0.5:
                raise ValueError(f"Invalid HAWQ one-hot solution: {values}")
            layer_bits.append(self.qbits[idx])
        return layer_bits

    def search(self, n_iter: int, target: str, constr: str = None, constr_value=None):
        self.evaluator.set_eval(target)
        if constr:
            self.evaluator.set_constraint(constr)
        if constr not in ["bops", "size", "latency_proxy"]:
            raise ValueError("HAWQSearcher supports bops, size, and latency_proxy constraints.")

        self._ensure_sensitivity()
        self.cost_table = self._build_cost_table(constr)
        if self.fix_first_last:
            print(f"HAWQ fixed first/last layers to {self.first_last_bit}-bit")
        layer_bits = self._solve_ilp(constr_value)
        scheme = self._scheme_for_layer_bits(layer_bits)
        real_constr = self.evaluator.constr(scheme)
        if real_constr > constr_value:
            raise ValueError(
                f"HAWQ ILP decoded scheme violates constraint: {real_constr} > {constr_value}"
            )
        result = self.evaluator.eval(scheme)
        print("HAWQ Layer Bits:", layer_bits)
        print("HAWQ Best Scheme:", scheme)
        print("HAWQ Result:", result)
        print(f"HAWQ Constraint Value: {real_constr} ({real_constr / constr_value:.5f})")
        self.best_trace = [result] * max(1, int(n_iter))
        return scheme, result
