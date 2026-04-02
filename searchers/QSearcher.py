from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Optional

from MiCoEval import MiCoEval

class QSearcher(ABC):

    evaluator: MiCoEval
    n_inits: int
    qtypes: list
    best_trace: list
    best_scheme_trace: list
    record_hook: Optional[Callable]
    target: Optional[str]
    constr_name: Optional[str]
    constr_value: Optional[float]

    def __init__(self, evaluator: MiCoEval,
                 n_inits: int = 10, 
                 qtypes: list = [4,5,6,7,8]) -> None:
        self.evaluator = evaluator
        self.n_inits = n_inits
        self.qtypes = qtypes
        self.best_trace = []
        self.best_scheme_trace = []
        self.record_hook = None
        self.target = None
        self.constr_name = None
        self.constr_value = None
        self.n_layers = evaluator.n_layers
        self.dim = evaluator.dim
        return

    def start_search(self, target: str, constr: str = None, constr_value=None):
        self.target = target
        self.constr_name = constr
        self.constr_value = constr_value
        self.best_trace = []
        self.best_scheme_trace = []
        self.evaluator.set_eval(target)
        if constr:
            self.evaluator.set_constraint(constr)

    def record_best(self, best_scheme, best_value):
        self.best_trace.append(best_value)
        self.best_scheme_trace.append(
            list(best_scheme) if best_scheme is not None else None
        )
        if (self.record_hook is not None) and callable(self.record_hook):
            self.record_hook(self, best_scheme, best_value)

    def set_record_hook(self, hook):
        self.record_hook = hook


    @abstractmethod
    def search(self, n_iter: int, target: str, 
               constr: str = None, 
               constr_value = None) -> Tuple[List, float]:
        pass
