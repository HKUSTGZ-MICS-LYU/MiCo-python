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

    remain = n_samples - 2
    n_layers = dims // 2

    # layers to be covered per initial sample
    layers_per_sample = n_layers // remain
    if layers_per_sample == 0:
        layers_per_sample = 1
    
    layer_idxs = []

    qtypes_l_max = [q for q in qtypes if q != max_bw]

    # Random Bitwidth Results
    while len(X) < n_samples:
        x = [max_bw] * dims
        layer_sel = []
        if len(layer_idxs) == 0:
            layer_idxs = list(range(n_layers))
            random.shuffle(layer_idxs)
        for i in range(layers_per_sample):
            if len(layer_idxs) == 0:
                break
            idx = layer_idxs.pop()
            layer_sel.append(idx)

        for sel in layer_sel:

            x[sel] = random.choice(qtypes_l_max)
            x[sel+n_layers] = random.choice(qtypes_l_max)

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
                       roi=0.2, layer_macs:list=None):
    if constr_func is None:
        return random_sample(n_samples, qtypes)
    
    pop = []
    for q in qtypes:
        pop.append([q] * dims)
    
    gen = 0
    n_layers = dims // 2

    min_scheme = [min(qtypes)] * dims
    min_scheme[0] = max(qtypes)
    min_scheme[n_layers] = max(qtypes)
    min_scheme[n_layers - 1] = max(qtypes)
    min_scheme[-1] = max(qtypes)

    if constr_func is not None:
        keep_first_last = constr_func(min_scheme) < constr_value
        min_scheme[n_layers - 1] = min(qtypes)
        min_scheme[-1] = min(qtypes)
        keep_first = constr_func(min_scheme) < constr_value
    else:
        keep_first_last = True
        keep_first = True

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

            # Heuristic: Keep First and Last Layer at High Bitwidth
            if keep_first_last:
                if random.random() < 0.5:
                    sample[0] = max(qtypes)
                    sample[n_layers] = max(qtypes)
                if random.random() < 0.5:
                    sample[n_layers - 1] = max(qtypes)
                    sample[-1] = max(qtypes)
            elif keep_first:
                if random.random() < 0.5:
                    sample[0] = max(qtypes)
                    sample[n_layers] = max(qtypes)

            # Heuristic: Swap Activation and Weight Bitwidth
            # for i in range(n_layers):
            #     # if Weight Bitwidth is more than Activation Bitwidth
            #     if sample[i] > sample[i+n_layers]:
            #         if random.random() < (1.0 - sample[i+n_layers]/ sample[i]):
            #             bitwidth = sample[i]
            #             sample[i] = sample[i+n_layers]
            #             sample[i+n_layers] = bitwidth
            
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
