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
        return
    
    def search(self, layer_w, constr_bops=None, constr_size=None, ptq=False):
        m = GEKKO()
        q_vars = [m.sos1(self.qbits) for i in range(self.n_layers)]


        if constr_bops is not None:
            m.Equation(m.sum([(q_vars[i] * q_vars[i] + q_vars[i] + q_vars[i]) * 
                          self.layer_macs[i] 
                          for i in range(self.n_layers)]) <= constr_bops)
        if constr_size is not None:
            m.Equation(m.sum([q_vars[i] * self.layer_params[i] 
                              for i in range(self.n_layers)]) <= constr_size)

        m.Minimize(m.sum([(q_vars[i] * q_vars[i]) * layer_w[i]
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
        return res
