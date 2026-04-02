from abc import ABC, abstractmethod
from typing import List, Tuple

from MiCoEval import MiCoEval

class QSearcher(ABC):

    evaluator: MiCoEval
    n_inits: int
    qtypes: list
    best_trace: list
    best_scheme_trace: list
    target: str | None
    constr_name: str | None
    constr_value: float | None

    def __init__(self, evaluator: MiCoEval,
                 n_inits: int = 10, 
                 qtypes: list = [4,5,6,7,8]) -> None:
        self.evaluator = evaluator
        self.n_inits = n_inits
        self.qtypes = qtypes
        self.best_trace = []
        self.best_scheme_trace = []
        self.target = None
        self.constr_name = None
        self.constr_value = None
        self.n_layers = evaluator.n_layers
        self.dim = evaluator.dim
        return


    @abstractmethod
    def search(self, n_iter: int, target: str, 
               constr: str = None, 
               constr_value = None) -> Tuple[List, float]:
        pass
