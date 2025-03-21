from MiCoEval import MiCoEval

from gekko import GEKKO

# Edge-MPQ Baseline NLP Searcher
class NLPSearcher:
    def __init__(self, evaluator: MiCoEval, 
                 n_inits: int = 10, 
                 qtypes: list = [4,5,6,7,8]) -> None:
        
        self.evaluator = evaluator
        self.qbits = qtypes
        self.layer_macs = self.evaluator.layer_macs
        self.layer_params = self.evaluator.layer_macs
        self.n_layers = self.evaluator.n_layers
        self.layer_w = []
        self.best_trace = []
        return
    
    def calculate_w(self, qbits = 4):
        BOPS_UNIT = 1e9
        self.layer_w = []
        # Evaluate INT8 model
        scheme = [8] * self.evaluator.n_layers * 2
        int8_acc = self.evaluator.eval_ptq(scheme)
        int8_bops = self.evaluator.eval_bops(scheme)
        print("INT8 Acc: ", int8_acc)
        print("INT8 BOPS: ", int8_bops)
        eps = 1e-5
        # Evaluate Layer-wise Quantized Model
        for i in range(self.n_layers):
            scheme = [8] * self.n_layers * 2
            scheme[i] = qbits
            scheme[self.n_layers + i] = qbits
            acc = self.evaluator.eval_ptq(scheme)
            bops = self.evaluator.eval_bops(scheme)
            print(f"Layer {i} Quantized Acc: {acc:.3f}, BOPS: {bops}")
            delta_ops = (int8_bops - bops) / BOPS_UNIT
            delta_acc = int8_acc - acc
            delta_acc = eps if delta_acc < eps else delta_acc
            w_value = delta_ops / delta_acc 
            s_value = 1.0 / w_value
            self.layer_w.append(-s_value)
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
        # FIXME: It's not working somehow, weird
        q_vars = [m.sos1(self.qbits) for i in range(self.n_layers)]

        total_bops = m.sum([(q_vars[i] * q_vars[i]) * self.layer_macs[i] 
                    for i in range(self.n_layers)])
        
        m.Equation(total_bops <= constr_value)
        
        total_w = m.sum([q_vars[i] * self.layer_w[i] for i in range(self.n_layers)])

        m.Minimize(total_w)
        
        print("Searching with BOPS Constraint:", constr_value)

        # m.options.IMODE = 3
        m.solve(disp=True)

        def convert_q(q):
            if 1.58 <= q < 2:
                return 1.58
            else :
                return int(q)

        wq_vars = [convert_q(v.value[0]) for v in q_vars]
        aq_vars = [convert_q(v.value[0]) for v in q_vars]

        qscheme = wq_vars + aq_vars
        print("Best QS: ", qscheme)
        res = self.evaluator.eval(qscheme)
        print("Best Result: ", res)

        # Dummy Trace
        self.best_trace = [res] * n_iter
        
        return qscheme, res
    
