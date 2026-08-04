from typing import List

from MiCoEval import MiCoEval
from searchers.QSearcher import QSearcher


class SimpleSearcher(QSearcher):
    """Constraint-aware uniform quantization baseline.

    The baseline picks the highest uniform bitwidth from qtypes that satisfies
    the requested constraint. Optionally, it keeps the first and last layers at
    the maximum qtype while using a uniform bitwidth for all middle layers.
    """

    def __init__(
        self,
        evaluator: MiCoEval,
        n_inits: int = 10,
        qtypes: list = None,
        fix_first_last: bool = False,
    ) -> None:
        qtypes = qtypes or [4, 5, 6, 7, 8]
        super().__init__(evaluator, n_inits, qtypes)
        self.qbits = sorted({int(q) for q in qtypes})
        if not self.qbits:
            raise ValueError("SimpleSearcher requires at least one qtype.")
        if min(self.qbits) < 1:
            raise ValueError("SimpleSearcher only supports integer qtypes >= 1.")
        self.fix_first_last = bool(fix_first_last)
        self.max_bit = max(self.qbits)

    def _scheme_for_bit(self, bit: int) -> List[int]:
        wq = [int(bit)] * self.n_layers
        aq = [int(bit)] * self.n_layers
        if self.fix_first_last and self.n_layers >= 1:
            fixed_layers = [0]
            if self.n_layers > 1:
                fixed_layers.append(self.n_layers - 1)
            for layer_idx in fixed_layers:
                wq[layer_idx] = self.max_bit
                aq[layer_idx] = self.max_bit
        return wq + aq

    def _select_scheme(self, constr_value):
        candidates = []
        for bit in sorted(self.qbits, reverse=True):
            scheme = self._scheme_for_bit(bit)
            constr = self.evaluator.constr(scheme)
            candidates.append((bit, scheme, constr))
            if constr <= constr_value:
                return bit, scheme, constr

        min_bit, min_scheme, min_constr = candidates[-1]
        raise ValueError(
            "No SimpleSearcher scheme satisfies the constraint: "
            f"minimum candidate bit={min_bit}, constraint={min_constr}, "
            f"target={constr_value}"
        )

    def search(self, n_iter: int, target: str, constr: str = None, constr_value=None):
        self.evaluator.set_eval(target)
        if constr:
            self.evaluator.set_constraint(constr)

        if constr is None:
            bit = self.max_bit
            scheme = self._scheme_for_bit(bit)
            real_constr = None
        else:
            if constr_value is None:
                raise ValueError("constr_value must be provided for constrained SimpleSearcher search.")
            bit, scheme, real_constr = self._select_scheme(constr_value)

        result = self.evaluator.eval(scheme)
        mode = "first_last_max_uniform_middle" if self.fix_first_last else "uniform"
        print(f"SimpleSearcher Mode: {mode}")
        print(f"SimpleSearcher Uniform Bit: {bit}")
        print(f"SimpleSearcher Best Scheme: {scheme}")
        print(f"SimpleSearcher Result: {result}")
        if real_constr is not None:
            print(f"SimpleSearcher Constraint Value: {real_constr} ({real_constr / constr_value:.5f})")

        self.best_trace = [result] * max(1, int(n_iter))
        return scheme, result
