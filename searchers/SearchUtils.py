import random
import itertools
import numpy as np


def random_sample(n_samples: int, qtypes:list, dims:int):
    X = np.random.choice(qtypes, (n_samples, dims)).tolist()
    return X

def random_sample_min_max(n_samples: int, qtypes: list, dims: int):
    X = []
    # Max Bitwidth Result
    max_bw = max(qtypes)
    X.append([max_bw] * dims)
    if n_samples - 1 > 0:
        min_bw = min(qtypes)
        X.append([min_bw] * dims)
    if n_samples - 2 > 0:
        X += np.random.choice(qtypes, (n_samples-2, dims)).tolist()
    return X

def grid_sample(n_samples: int, qtypes: list, dims: int):
    
    X = []
    # Max Bitwidth Result
    max_bw = max(qtypes)
    X.append([max_bw] * dims)
    if n_samples - 1 > 0:
        min_bw = min(qtypes)
        X.append([min_bw] * dims)

    # Random Bitwidth Results
    while len(X) < n_samples:
        x = [max_bw] * dims
        # Random Mutate
        for i in range(dims):
            if random.random() < 0.5:
                x[i] = random.choice(qtypes)
        if x not in X:
            X.append(x)
    return X