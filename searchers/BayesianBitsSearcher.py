import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from MiCoEval import MiCoEval
from MiCoUtils import list_qlayers
from searchers.QSearcher import QSearcher


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def round_ste(x: torch.Tensor):
    return x + (torch.round(x) - x).detach()


def hard_concrete_sample(log_alpha, beta=2.0 / 3.0, zeta=1.1, gamma=-0.1):
    u = torch.rand_like(log_alpha)
    s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + log_alpha) / beta)
    sbar = s * (zeta - gamma) + gamma
    return torch.clamp(sbar, 0, 1)


def hard_concrete_prob_on(log_alpha, beta=2.0 / 3.0, zeta=1.1, gamma=-0.1):
    return torch.sigmoid(log_alpha - beta * math.log(-gamma / zeta))


def hard_concrete_eval_gate(log_alpha, beta=2.0 / 3.0, zeta=1.1, gamma=-0.1, threshold=0.34):
    offset = math.log(-gamma / zeta) * beta
    return (torch.sigmoid(offset - log_alpha) < threshold).float()


class OrderedGateQuantizer(nn.Module):
    """BayesianBits-style ordered gates over an arbitrary MiCo bitwidth set."""

    def __init__(self, qtypes, is_activation=False, gamma_init=6.0, stochastic=True):
        super().__init__()
        self.qtypes = sorted({int(q) for q in qtypes})
        if len(self.qtypes) < 1:
            raise ValueError("qtypes must contain at least one bitwidth")
        if min(self.qtypes) < 1:
            raise ValueError("BayesianBitsSearcher only supports qtypes >= 1")
        self.gate_transitions = list(zip(self.qtypes[:-1], self.qtypes[1:]))
        self.is_activation = is_activation
        self.stochastic = stochastic
        n_gates = len(self.gate_transitions)
        if isinstance(gamma_init, (list, tuple)):
            if len(gamma_init) != n_gates:
                raise ValueError(
                    f"gamma_init length {len(gamma_init)} must match number of gates {n_gates} "
                    f"for qtype transitions {self.gate_transitions}"
                )
            gamma_tensor = torch.tensor([float(x) for x in gamma_init])
        else:
            gamma_tensor = torch.full((n_gates,), float(gamma_init))
        self.gamma = nn.Parameter(gamma_tensor)

    def _activation_reduce_dim(self, x):
        if x.dim() == 4:
            return (-2, -1)
        return -1

    def _quantize(self, x, qbit):
        qbit = int(qbit)
        round_f = round_ste if self.training else torch.round
        if self.is_activation:
            reduce_dim = self._activation_reduce_dim(x)
            if qbit == 1:
                denom = x.abs().mean(dim=reduce_dim, keepdim=True).clamp(min=1e-5)
                y = x.sign() * denom
                q = torch.where(y == 0.0, -denom, y)
                return x + (q - x).detach() if self.training else q
            if qbit == 2:
                denom = x.abs().mean(dim=reduce_dim, keepdim=True).clamp(min=1e-5)
                scale = 1.0 / denom
                return round_f(x * scale).clamp(-2, 1) / scale
            denom = x.abs().amax(dim=reduce_dim, keepdim=True).clamp(min=1e-5)
            scale = (2 ** (qbit - 1) - 1) / denom
            return round_f(x * scale).clamp(-(2 ** (qbit - 1)), 2 ** (qbit - 1) - 1) / scale

        if qbit == 1:
            denom = x.abs().mean().clamp(min=1e-5)
            y = x.sign() * denom
            q = torch.where(y == 0.0, -denom, y)
            return x + (q - x).detach() if self.training else q
        if qbit == 2:
            denom = x.abs().mean().clamp(min=1e-5)
            scale = (2 ** (qbit - 1) - 1) / denom
            return round_f(x * scale).clamp(-2, 1) / scale
        denom = x.abs().max().clamp(min=1e-5)
        scale = (2 ** (qbit - 1) - 1) / denom
        return round_f(x * scale).clamp(-(2 ** (qbit - 1)), 2 ** (qbit - 1) - 1) / scale

    def _gates(self):
        if self.gamma.numel() == 0:
            return []
        if self.training and self.stochastic:
            return list(hard_concrete_sample(self.gamma))
        return list(hard_concrete_eval_gate(self.gamma))

    def expected_bit(self):
        if len(self.qtypes) == 1:
            return torch.tensor(float(self.qtypes[0]), device=self.gamma.device)

        probs = list(hard_concrete_prob_on(self.gamma))
        value = torch.tensor(float(self.qtypes[-1] - self.qtypes[-2]), device=self.gamma.device)
        for idx in range(len(probs) - 2, -1, -1):
            delta = float(self.qtypes[idx + 1] - self.qtypes[idx])
            value = delta + probs[idx + 1] * value if idx + 1 < len(probs) else value
        return float(self.qtypes[0]) + probs[0] * value

    def regularizer(self, mac_count, max_macs):
        if len(self.qtypes) == 1:
            return torch.tensor(0.0, device=self.gamma.device)

        reg = torch.tensor(0.0, device=self.gamma.device)
        prev_on = torch.tensor(1.0, device=self.gamma.device)
        mac_weight = float(mac_count) / max(float(max_macs), 1.0)
        for prob_on, bitwidth in zip(hard_concrete_prob_on(self.gamma), self.qtypes[1:]):
            gate_on = prev_on * prob_on
            reg = reg + gate_on * float(bitwidth) * mac_weight
            prev_on = gate_on
        return reg

    def decode_bit(self):
        bit = self.qtypes[0]
        for gate, next_bit in zip(hard_concrete_eval_gate(self.gamma), self.qtypes[1:]):
            if float(gate.item()) < 0.5:
                break
            bit = next_bit
        return int(bit)

    def forward(self, x):
        if len(self.qtypes) == 1:
            return self._quantize(x, self.qtypes[0])

        qvals = [self._quantize(x, q) for q in self.qtypes]
        out = qvals[-1] - qvals[-2]
        gates = self._gates()
        for idx in range(len(gates) - 2, -1, -1):
            out = qvals[idx + 1] - qvals[idx] + gates[idx + 1] * out
        out = qvals[0] + gates[0] * out
        return out


class BayesianBitsSearcher(QSearcher):
    def __init__(
        self,
        evaluator: MiCoEval,
        n_inits: int = 10,
        qtypes: list = None,
        mu_grid=None,
        lr: float = 1e-3,
        gamma_init: float = 6.0,
        max_batches: int = None,
        train_weights: str = "auto",
        stochastic: bool = True,
        objective: str = "constrained",
        constraint_lambda: float = 10.0,
        constraint_margin: float = 1.0,
    ) -> None:
        qtypes = qtypes or [1, 2, 4, 8]
        super().__init__(evaluator, n_inits, qtypes)
        self.qtypes = sorted({int(q) for q in qtypes})
        self.gate_transitions = list(zip(self.qtypes[:-1], self.qtypes[1:]))
        if isinstance(gamma_init, (list, tuple)) and len(gamma_init) != len(self.gate_transitions):
            raise ValueError(
                f"gamma_init length {len(gamma_init)} must match number of gates "
                f"{len(self.gate_transitions)} for qtype transitions {self.gate_transitions}"
            )
        self.mu_grid = mu_grid or [0.01, 0.03, 0.05, 0.07, 0.1, 0.2]
        self.lr = lr
        self.gamma_init = gamma_init
        self.max_batches = max_batches
        self.train_weights = train_weights
        self.stochastic = stochastic
        if objective not in ["constrained", "pareto"]:
            raise ValueError(f"Unknown BayesianBits objective: {objective}")
        self.objective = objective
        self.constraint_lambda = constraint_lambda
        self.constraint_margin = constraint_margin
        self.candidates = []
        self.device = self._select_device()

    def _select_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(self.evaluator.model, "current_device"):
            return self.evaluator.model.current_device()
        return DEVICE

    def _prepare_model(self, target):
        self.evaluator.load_pretrain()
        model = deepcopy(self.evaluator.model).to(self.device)
        max_q = max(self.qtypes)
        model.set_qscheme(
            [[max_q] * self.n_layers, [max_q] * self.n_layers],
            qat=True,
            group_size=self.evaluator.group_size,
            device=self.device,
        )
        layers = list_qlayers(model)
        if len(layers) != self.n_layers:
            raise ValueError(f"Expected {self.n_layers} qlayers, got {len(layers)}")

        learn_weights = self.train_weights == "yes" or (
            self.train_weights == "auto" and target == "qat_acc"
        )
        for layer in layers:
            layer.bb_weight_quantizer = OrderedGateQuantizer(
                self.qtypes, is_activation=False, gamma_init=self.gamma_init,
                stochastic=self.stochastic,
            ).to(self.device)
            layer.bb_act_quantizer = OrderedGateQuantizer(
                self.qtypes, is_activation=True, gamma_init=self.gamma_init,
                stochastic=self.stochastic,
            ).to(self.device)

        for name, param in model.named_parameters():
            is_bb_param = ".bb_weight_quantizer." in name or ".bb_act_quantizer." in name
            param.requires_grad = is_bb_param or learn_weights
        return model, layers

    def _expected_mico_bops(self, layers):
        total = torch.tensor(0.0, device=self.device)
        for layer in layers:
            macs = float(layer.get_mac())
            w_bit = layer.bb_weight_quantizer.expected_bit()
            a_bit = layer.bb_act_quantizer.expected_bit()
            total = total + macs * w_bit * a_bit
        return total

    def _gate_regularizer(self, layers):
        max_macs = max(float(macs) for macs in self.evaluator.layer_macs)
        total = torch.tensor(0.0, device=self.device)
        for layer in layers:
            macs = float(layer.get_mac())
            total = total + layer.bb_weight_quantizer.regularizer(macs, max_macs)
            total = total + layer.bb_act_quantizer.regularizer(macs, max_macs)
        return total

    def _decode_scheme(self, layers):
        wq = [layer.bb_weight_quantizer.decode_bit() for layer in layers]
        aq = [layer.bb_act_quantizer.decode_bit() for layer in layers]
        return wq + aq

    def _loss_complexity(self, layers, mu, target_constr):
        bb_reg = self._gate_regularizer(layers)
        expected_bops = self._expected_mico_bops(layers)
        constraint_target = max(float(target_constr) * float(self.constraint_margin), 1.0)
        violation = F.relu(expected_bops / constraint_target - 1.0)
        if self.objective == "pareto":
            return float(mu) * bb_reg, bb_reg, expected_bops, violation

        penalty = float(self.constraint_lambda) * violation.pow(2)
        return penalty, bb_reg, expected_bops, violation

    def _train_one_mu(self, target, mu, epochs, max_constr, target_constr):
        model, layers = self._prepare_model(target)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.lr,
        )
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            total_batches = len(self.evaluator.train_loader)
            if self.max_batches is not None:
                total_batches = min(total_batches, self.max_batches)
            batch_iter = tqdm(
                enumerate(self.evaluator.train_loader),
                total=total_batches,
                desc=f"BB mu={mu} epoch {epoch + 1}/{epochs}",
                leave=False,
            )
            last_loss = None
            for batch_idx, (images, labels) in batch_iter:
                if self.max_batches is not None and batch_idx >= self.max_batches:
                    break
                x = images.to(self.device)
                y = labels.to(self.device)
                optimizer.zero_grad()
                output = model(x)
                ce_loss = criterion(output, y)
                complexity_loss, bb_reg, expected_bops, violation = self._loss_complexity(
                    layers, mu, target_constr
                )
                loss = ce_loss + complexity_loss
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())
                expected_ratio = float((expected_bops / max(max_constr, 1.0)).detach().cpu())
                target_ratio = float((expected_bops / max(float(target_constr), 1.0)).detach().cpu())
                batch_iter.set_postfix(
                    loss=f"{last_loss:.4f}",
                    bb_reg=f"{float(bb_reg.item()):.4f}",
                    target=f"{target_ratio:.3f}",
                    vio=f"{float(violation.item()):.3f}",
                    ebops=f"{expected_ratio:.3f}",
                )
            scheme = self._decode_scheme(layers)
            scheme_constr = self.evaluator.constr(scheme)
            bb_reg = float(self._gate_regularizer(layers).detach().cpu())
            expected_bops = self._expected_mico_bops(layers)
            expected_ratio = float((expected_bops / max(max_constr, 1.0)).detach().cpu())
            target_ratio = float((expected_bops / max(float(target_constr), 1.0)).detach().cpu())
            margin_target = max(float(target_constr) * float(self.constraint_margin), 1.0)
            violation = max(float((expected_bops / margin_target).detach().cpu()) - 1.0, 0.0)
            loss_text = "nan" if last_loss is None else f"{last_loss:.4f}"
            print(
                f"BB mu={mu} epoch={epoch + 1}/{epochs} loss={loss_text} "
                f"bb_reg={bb_reg:.5f} expected_mico_bops_ratio={expected_ratio:.5f} "
                f"expected_target_ratio={target_ratio:.5f} violation={violation:.5f}"
            )
            print(
                f"BB Epoch Decoded Scheme: {scheme} "
                f"constraint={scheme_constr} ({scheme_constr / max(max_constr, 1.0):.5f} of max)"
            )

        scheme = self._decode_scheme(layers)
        return scheme

    def search(self, n_iter: int, target: str, constr: str = None, constr_value=None):
        self.evaluator.set_eval(target)
        if constr:
            self.evaluator.set_constraint(constr)

        epochs = max(1, int(n_iter))
        if self.objective == "constrained" and constr != "bops":
            raise ValueError("BayesianBits constrained objective currently supports only bops constraints.")
        max_scheme = [max(self.qtypes)] * self.dim
        max_constr = self.evaluator.constr(max_scheme)
        target_constr = constr_value if constr_value is not None else max_constr
        final_x = None
        final_y = None
        closest_x = None
        closest_y = None
        closest_gap = None

        for mu in self.mu_grid:
            print(
                f"BayesianBits Training: mu={mu}, epochs={epochs}, qtypes={self.qtypes}, "
                f"gate_transitions={self.gate_transitions}, objective={self.objective}, device={self.device}"
            )
            scheme = self._train_one_mu(target, mu, epochs, max_constr, target_constr)
            real_constr = self.evaluator.constr(scheme) if constr else None
            eval_y = self.evaluator.eval(scheme)
            feasible = (not constr) or (real_constr <= constr_value)
            print("BB Decoded Scheme:", scheme)
            print("BB Result:", eval_y)
            if constr:
                print(f"BB Constraint Value: {real_constr} ({real_constr / constr_value:.5f})")

            self.candidates.append({
                "mu": float(mu),
                "scheme": scheme,
                "result": float(eval_y),
                "constraint": None if real_constr is None else float(real_constr),
                "feasible": bool(feasible),
            })

            if feasible and ((final_y is None) or (eval_y > final_y)):
                final_x = scheme
                final_y = eval_y
            if constr:
                gap = abs(real_constr - constr_value)
                if closest_gap is None or gap < closest_gap:
                    closest_gap = gap
                    closest_x = scheme
                    closest_y = eval_y
            self.best_trace.append(final_y if final_y is not None else eval_y)

        if final_x is None:
            if closest_x is None:
                raise ValueError("BayesianBits did not produce any candidate scheme.")
            print("Warning: no feasible BayesianBits candidate found. Returning closest constraint candidate.")
            final_x = closest_x
            final_y = closest_y

        print("Optimization Ends...")
        print("Best Scheme:", final_x)
        print("Best Result:", final_y)
        if constr:
            constr_x = self.evaluator.constr(final_x)
            print(f"Constraint Value: {constr_x} ({constr_x / constr_value:.5f})")
        return final_x, final_y
