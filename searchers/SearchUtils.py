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
    layers_per_sample = 0
    if remain > 0:
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

def dist_to_roi(x, constr_func, roi_lb, roi_ub):
    constr = constr_func(x)
    lb = roi_lb
    ub = roi_ub
    
    # Strictly prioritize valid samples (<= ub)
    if constr > ub:
        # Penalty for invalid samples: huge value + distance
        return 1e6 + (constr - ub)
    
    # For valid samples
    if constr >= lb:
        return 0.0 # Inside ROI
    else:
        return lb - constr # Valid but too efficient (too small), penalize slightly

def near_constr_sample(n_samples: int, qtypes: list, dims: int,
                       constr_func=None, constr_value=None, roi=0.2, initial_pop=None):
    if constr_func is None:
        return random_sample(n_samples, qtypes)
    
    pop = []
    score = []
    for q in qtypes:
        pop.append([q] * dims)
    if initial_pop is not None:
        pop += initial_pop
    max_value = constr_func([max(qtypes)] * dims)
    cur_ratio = constr_value / max_value
    assert cur_ratio > roi, f"ROI is too large! Current Ratio: {cur_ratio}, ROI: {roi}"
    roi_lb = (cur_ratio - roi) * max_value
    roi_ub = constr_value

    # Get initial score
    for x in pop:
        score.append(dist_to_roi(x, constr_func, roi_lb, roi_ub))

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

    crossover_ratio = 0.25
    mutation_ratio = 0.1
    keep_ratio = 0.75

    while True:
        # Generate Next Generation
        while len(pop) < n_samples*2:
            pair = random.sample(pop, 2)
            sample = copy.deepcopy(pair[0])
            for i in range(dims):
                if random.random() < crossover_ratio:
                    sample[i] = pair[1][i]
                    
            for i in range(dims):
                if random.random() < mutation_ratio:
                    q = random.choice(qtypes)
                    sample[i] = q

            # Heuristic: Keep First and Last Layer at High Bitwidth
            if keep_first_last:
                if random.random() < keep_ratio:
                    sample[0] = max(qtypes)
                if random.random() < keep_ratio:
                    sample[n_layers] = max(qtypes)
                if random.random() < keep_ratio:
                    sample[n_layers - 1] = max(qtypes)
                if random.random() < keep_ratio:
                    sample[-1] = max(qtypes)
            elif keep_first:
                if random.random() < keep_ratio:
                    sample[0] = max(qtypes)
                if random.random() < keep_ratio:
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
            score.append(dist_to_roi(sample, constr_func, roi_lb, roi_ub))

        # rank to get near to constraint
        sorted_pairs = sorted(zip(score, pop), key=lambda pair: pair[0])
        # sort by score and eliminate population to keep only near constraint samples
        score = [pair[0] for pair in sorted_pairs[:n_samples]]
        pop = [pair[1] for pair in sorted_pairs[:n_samples]]

        if dist_to_roi(pop[-1], constr_func, roi_lb, roi_ub) == 0.0:
            break
        
        gen += 1

        if gen > 100:
            print("Warning: Near Constraint Sample Timeout")
            # Only discard truly invalid samples (those with huge distance)
            end_idx = len(pop)
            for i in range(len(pop)):
                if dist_to_roi(pop[i], constr_func, roi_lb, roi_ub) > 1e5:
                    end_idx = i
                    break
            pop = pop[:end_idx]
            break

    return pop
