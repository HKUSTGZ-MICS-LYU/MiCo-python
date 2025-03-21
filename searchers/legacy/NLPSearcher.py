from MiCoSearch import MiCoSearch

from gekko import GEKKO

# Edge-MPQ Baseline NLP Searcher
class NLPSearcher:
    def __init__(self, search: MiCoSearch, qbits: list) -> None:
        self.mpq = search
        self.qbits = qbits
        self.layer_macs = self.mpq.layer_macs
        self.layer_params = self.mpq.layer_params
        self.n_layers = self.mpq.n_layers
        self.layer_w = []
        return
    
    def calculate_w(self, qbits = 4, max_bops = False):
        BOPS_UNIT = 1e9
        self.layer_w = []
        # Evaluate INT8 model
        scheme = [[8] * self.n_layers, [8] * self.n_layers]
        res = self.mpq.eval_scheme(scheme, ptq=True)
        int8_acc = res["Accuracy"]
        int8_bops = res["MaxBOPs"] if max_bops else res["BOPs"]
        eps = 1e-5
        # Evaluate Layer-wise Quantized Model
        for i in range(self.n_layers):
            scheme = [[8] * self.n_layers, [8] * self.n_layers]
            scheme[0][i] = qbits
            scheme[1][i] = qbits
            res = self.mpq.eval_scheme(scheme, ptq=True)
            acc = res["Accuracy"]
            bops = res["MaxBOPs"] if max_bops else res["BOPs"]
            delta_ops = (int8_bops - bops) / BOPS_UNIT
            delta_acc = int8_acc - acc
            delta_acc = eps if delta_acc < eps else delta_acc
            w_value = delta_ops / delta_acc 
            self.layer_w.append(-1.0 / w_value)
        print("Layer-wise W: ", self.layer_w)
        return


    def search(self, constr_bops=None, constr_size=None, ptq=False, use_max_q=False):
        self.calculate_w(qbits=min(self.qbits), max_bops=use_max_q)
        m = GEKKO(remote=False)
        print(self.n_layers)
        q_vars = [m.sos1(self.qbits) for i in range(self.n_layers)]

        if constr_bops is not None:
            m.Equation(m.sum([(q_vars[i] * q_vars[i] + q_vars[i] + q_vars[i]) * 
                          self.layer_macs[i] 
                          for i in range(self.n_layers)]) <= constr_bops)
        if constr_size is not None:
            m.Equation(m.sum([q_vars[i] * self.layer_params[i] 
                              for i in range(self.n_layers)]) <= constr_size)

        m.Minimize(m.sum([(q_vars[i] * q_vars[i]) * self.layer_w[i]
                     for i in range(self.n_layers)]))

        m.options.IMODE = 3
        m.solve(disp=True)

        def convert_q(q):
            if 1.58 <= q < 2:
                return 1.58
            else :
                return int(q)

        wq_vars = [convert_q(v.value[0]) for v in q_vars]
        aq_vars = [convert_q(v.value[0]) for v in q_vars]

        qscheme = [wq_vars, aq_vars]
        name = self.mpq.get_scheme_str(qscheme)
        print("Scheme: ", name)
        res = self.mpq.eval_scheme(qscheme, verbose=True, ptq=ptq)
        return {name:res}
    
