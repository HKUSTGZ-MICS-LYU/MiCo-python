from MiCoEval import MiCoEval
from searchers.QSearcher import QSearcher

from gekko import GEKKO
from copy import deepcopy

# Edge-MPQ Baseline NLP Searcher
class NLPSearcher(QSearcher):
    def __init__(self, evaluator: MiCoEval, 
                 n_inits: int = 10, 
                 qtypes: list = [4,5,6,7,8],
                 use_sos = False) -> None:
        super().__init__(evaluator, n_inits, qtypes)

        self.qbits = qtypes
        self.layer_macs = self.evaluator.layer_macs
        self.layer_params = self.evaluator.layer_macs
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
            s_value = 1.0 / w_value
            self.layer_w.append(float(-s_value))
        print("Layer-wise W: ", self.layer_w)
        return


    def search(self, n_iter: int, target: str, 
               constr: str = None, 
               constr_value = None):
        
        self.evaluator.set_eval(target)
        assert constr == 'bops', "Only BOPS constraint is supported"
        self.evaluator.set_constraint(constr)

        print("Calculating Layer-wise W...")
        self.calculate_w(qbits=min(self.qbits))
        m = GEKKO(remote=False)

        if self.use_sos:
            # Use binary variables to select exactly one value from qbits
            q_vars = []
            for i in range(self.n_layers):
                b = [m.Var(lb=0, ub=1, integer=True) for _ in self.qbits]
                b[0].value = 1
                m.Equation(m.sum(b) == 1)
                q_var = m.Intermediate(
                    m.sum([b[j] * self.qbits[j] for j in range(len(self.qbits))])
                )
                q_vars.append(q_var)
        else:
            # Integer Method
            q_vars = [m.CV(lb=min(self.qbits), ub=max(self.qbits), integer=True) \
                      for i in range(self.n_layers)]
        
        m.Equation(m.sum([(q_vars[i]**2) * self.layer_macs[i] 
                        for i in range(self.n_layers)]) <= constr_value)
        
        total_w = m.sum([(q_vars[i]*q_vars[i]) * self.layer_w[i] for i in range(self.n_layers)])
        m.Obj(total_w)

        if self.use_sos:
            print("Using SOS1 Constraints for Quantization Bitwidths")
            m.options.IMODE = 3
            m.options.SOLVER = 1
        else:
            m.options.IMODE = 3
            m.options.SOLVER = 3
        
        print("Searching with BOPS Constraint:", constr_value)

        try:
            m.solve(disp=True)
        except Exception as e:
            print("Gekko Solver Error:", e)
            return None, None

        def convert_q(q):
            if 1.58 <= q < 2:
                return 1.58
            else :
                return int(q)

        wq_vars = [convert_q(v.value[0]) for v in q_vars]
        aq_vars = [convert_q(v.value[0]) for v in q_vars]

        # Assert all variables are in the allowed quantization bitwidths
        for v in wq_vars + aq_vars:
            if v not in self.qbits:
                print("Warning: Gekko returned a bitwidth that is not in the allowed set:", v)
                return None, None

        qscheme = wq_vars + aq_vars
        print("Best QS: ", qscheme)
        res = self.evaluator.eval(qscheme)
        print("Best Result: ", res)

        # Dummy Trace
        self.best_trace = [res] * n_iter
        
        return qscheme, res
    
