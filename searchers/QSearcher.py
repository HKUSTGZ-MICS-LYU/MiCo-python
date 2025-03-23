from abc import ABC, abstractmethod
from typing import List, Tuple

from MiCoEval import MiCoEval

class QSearcher(ABC):

    evaluator: MiCoEval
    n_inits: int
    qtypes: list
    best_trace: list

    def __init__(self, evaluator: MiCoEval,
                 n_inits: int = 10, 
                 qtypes: list = [4,5,6,7,8]) -> None:
        self.evaluator = evaluator
        self.n_inits = n_inits
        self.qtypes = qtypes
        self.best_trace = []
        return


    @abstractmethod
    def search(self, n_iter: int, target: str, 
               constr: str = None, 
               constr_value = None) -> Tuple[List, float]:
        pass