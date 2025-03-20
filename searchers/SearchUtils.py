import random
import itertools
import numpy as np


def random_sample(n_samples: int, qtypes:list, dims:int):
    X = np.random.choice(qtypes, (n_samples, dims)).tolist()
    return X

def grid_sample(n_samples: int, qtypes: list, dims: int):
    
    X = []
    # Max Bitwidth Result
    max_bw = max(qtypes)
    X.append([max_bw] * dims)

    for _ in range(n_samples-1):
        x = [max_bw] * dims
        # Random Mutate
        for i in range(dims):
            if random.random() < 0.5:
                x[i] = random.choice(qtypes)
        X.append(x)
    return X