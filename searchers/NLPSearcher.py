from MiCoEval import MiCoEval
from searchers.QSearcher import QSearcher

from gekko import GEKKO

# Edge-MPQ Baseline NLP Searcher
class NLPSearcher(QSearcher):
    def __init__(self, evaluator: MiCoEval, 
                 n_inits: int = 10, 
                 qtypes: list = [4,5,6,7,8],
                 use_sos = False) -> None:
        super().__init__(evaluator, n_inits, qtypes)

        self.qbits = sorted(set(qtypes))
        self.layer_macs = self.evaluator.layer_macs
        self.layer_params = self.evaluator.layer_params
        self.layer_w = []
        self.best_trace = []
        self.use_sos = use_sos
        return
    
    def calculate_w(self, qbits = 4):
        BOPS_UNIT = 1e9
        self.layer_w = []
        # Evaluate INT8 model
        scheme = [8] * self.evaluator.n_layers * 2
        int8_acc = self.evaluator.eval(scheme)
        int8_bops = self.evaluator.eval_bops(scheme)
        print("INT8 Acc: ", int8_acc)
        print("INT8 BOPS: ", int8_bops)
        eps = 1e-5
        # Evaluate Layer-wise Quantized Model
        for i in range(self.n_layers):
            scheme = [8] * self.n_layers * 2
            scheme[i] = qbits
            scheme[self.n_layers + i] = qbits
            acc = self.evaluator.eval(scheme)
            bops = self.evaluator.constr(scheme)
            print(f"Layer {i} Quantized Acc: {acc:.3f}, BOPS: {bops}")
            delta_ops = (int8_bops - bops) / BOPS_UNIT
            delta_acc = int8_acc - acc
            delta_acc = eps if delta_acc < eps else delta_acc
            w_value = delta_ops / delta_acc
            s_value = 1.0 / w_value  # Accuracy loss per BOP unit
            self.layer_w.append(float(-s_value))
        print("Layer-wise W: ", self.layer_w)
        return

    def _build_q_vars(self, m: GEKKO):
        # Use explicit one-hot binaries for robust discrete selection.
        # GEKKO SOS1 can be numerically fragile for some constrained MINLPs.
        # Also model q^2 directly to keep constraints/objective linear.
        pick_vars = []
        q2_vars = []
        for _ in range(self.n_layers):
            picks = [m.Var(lb=0, ub=1, integer=True) for _ in self.qbits]
            m.Equation(m.sum(picks) == 1)
            q2_var = m.Intermediate(
                m.sum([picks[i] * (self.qbits[i] ** 2) for i in range(len(self.qbits))])
            )
            pick_vars.append(picks)
            q2_vars.append(q2_var)
        return pick_vars, q2_vars

    def _decode_pick(self, pick_values: list, tol: float = 1e-3):
        idx = max(range(len(pick_values)), key=lambda i: pick_values[i])
        if pick_values[idx] < (1.0 - tol):
            raise ValueError(f"Invalid one-hot assignment returned by GEKKO: {pick_values}")
        return self.qbits[idx]

    def search(self, n_iter: int, target: str, 
               constr: str = None, 
               constr_value = None):
        
        self.evaluator.set_eval(target)
        assert constr == 'bops', "Only BOPS constraint is supported"
        if constr_value is None:
            raise ValueError("constr_value must be provided for bops-constrained NLP search.")
        self.evaluator.set_constraint(constr)

        print("Calculating Layer-wise W...")
        self.calculate_w(qbits=min(self.qbits))
        m = GEKKO(remote=False)

        pick_vars, q2_vars = self._build_q_vars(m)
        
        m.Equation(m.sum([
            q2_vars[i] * self.layer_macs[i] for i in range(self.n_layers)
        ]) <= constr_value)

        # layer_w is "accuracy loss per BOP unit"; multiply by layer MACs to
        # get per-layer impact under q^2 MAC scaling.
        weighted_terms = [self.layer_w[i] * self.layer_macs[i] for i in range(self.n_layers)]
        max_abs_weight = max(abs(v) for v in weighted_terms) if weighted_terms else 1.0
        if max_abs_weight <= 0:
            max_abs_weight = 1.0

        total_w = m.sum([
            q2_vars[i] * (weighted_terms[i] / max_abs_weight)
            for i in range(self.n_layers)
        ])
        m.Obj(total_w)

        if self.use_sos:
            print("Using robust one-hot constraints for quantization bitwidths")
        m.options.IMODE = 3
        m.options.SOLVER = 1
        
        print("Searching with BOPS Constraint:", constr_value)

        try:
            m.solve(disp=True)
        except Exception as e:
            print("Gekko Solver Error:", e)
            return None, None

        try:
            wq_vars = []
            for picks in pick_vars:
                pick_values = [p.value[0] for p in picks]
                wq_vars.append(self._decode_pick(pick_values))
            aq_vars = wq_vars.copy()
        except ValueError as e:
            print("Warning:", e)
            return None, None

        qscheme = wq_vars + aq_vars
        print("Best QS: ", qscheme)

        real_bops = self.evaluator.constr(qscheme)
        if real_bops > constr_value:
            print("Warning: Decoded solution violates BOPS constraint:", real_bops)
            return None, None

        res = self.evaluator.eval(qscheme)
        print("Best Result: ", res)

        # Dummy Trace
        self.best_trace = [res] * n_iter
        
        return qscheme, res
    
