import random
import itertools
import numpy as np

import copy

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

def dist_to_roi(x, constr_func, constr_value, roi):
    constr = constr_func(x)
    lb = constr_value * (1-roi)
    ub = constr_value
    if (constr <= ub) and (constr >= lb):
        return 0.0
    elif constr < lb:
        return lb - constr
    else:
        return constr - ub

def near_constr_sample(n_samples: int, qtypes: list, dims: int,
                       constr_func=None, constr_value=None,
                       roi=0.2):
    if constr_func is None:
        return random_sample(n_samples, qtypes)
    
    pop = []
    for q in qtypes:
        pop.append([q] * dims)
    
    gen = 0

    while True:
        # Generate Next Generation
        while len(pop) < n_samples*2:
            pair = random.sample(pop, 2)
            sample = copy.deepcopy(pair[0])
            for i in range(dims):
                if random.random() < 0.1:
                    sample[i] = pair[1][i]
                    
            for i in range(dims):
                if random.random() < 0.1:
                    q = random.choice(qtypes)
                    sample[i] = q

            if sample in pop:
                continue

            pop.append(sample)

        # rank to get near to constraint
        pop = sorted(
            pop, 
            key=lambda x: dist_to_roi(x, constr_func, constr_value, roi),
        )

        pop = pop[:n_samples]

        if dist_to_roi(pop[-1], constr_func, constr_value, roi) == 0.0:
            break
        
        gen += 1

        if gen > 100:
            print("Warning: Near Constraint Sample Timeout")
            for i in range(n_samples):
                if dist_to_roi(pop[i], constr_func, constr_value, roi) > 0:
                    end_idx = i
                    break
            pop = pop[:end_idx]
            break

    return pop
