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
    X_set = set()  # Use set for O(1) membership testing
    
    # Max Bitwidth Result
    max_bw = max(qtypes)
    x_max = [max_bw] * dims
    X.append(x_max)
    X_set.add(tuple(x_max))
    
    if n_samples - 1 > 0:
        min_bw = min(qtypes)
        x_min = [min_bw] * dims
        X.append(x_min)
        X_set.add(tuple(x_min))

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

        x_tuple = tuple(x)
        if x_tuple not in X_set:
            X.append(x)
            X_set.add(x_tuple)
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

    # Cache min/max to avoid repeated computation
    min_q = min(qtypes)
    max_q = max(qtypes)

    min_scheme = [min_q] * dims
    min_scheme[0] = max_q
    min_scheme[n_layers] = max_q
    min_scheme[n_layers - 1] = max_q
    min_scheme[-1] = max_q

    if constr_func is not None:
        keep_first_last = constr_func(min_scheme) < constr_value
        min_scheme[n_layers - 1] = min_q
        min_scheme[-1] = min_q
        keep_first = constr_func(min_scheme) < constr_value
    else:
        keep_first_last = True
        keep_first = True

    # Use a set for O(1) membership testing (convert lists to tuples for hashing)
    pop_set = {tuple(p) for p in pop}

    while True:
        # Generate Next Generation
        while len(pop) < n_samples*2:
            pair = random.sample(pop, 2)
            # Use list slicing instead of deepcopy (faster for simple lists)
            sample = pair[0][:]
            for i in range(dims):
                if random.random() < 0.1:
                    sample[i] = pair[1][i]
                    
            for i in range(dims):
                if random.random() < 0.1:
                    sample[i] = random.choice(qtypes)

            # Heuristic: Keep First and Last Layer at High Bitwidth
            if keep_first_last:
                if random.random() < 0.5:
                    sample[0] = max_q
                    sample[n_layers] = max_q
                if random.random() < 0.5:
                    sample[n_layers - 1] = max_q
                    sample[-1] = max_q
            elif keep_first:
                if random.random() < 0.5:
                    sample[0] = max_q
                    sample[n_layers] = max_q

            # Use set for O(1) membership check
            sample_tuple = tuple(sample)
            if sample_tuple in pop_set:
                continue

            pop.append(sample)
            pop_set.add(sample_tuple)

        # rank to get near to constraint
        pop = sorted(
            pop, 
            key=lambda x: dist_to_roi(x, constr_func, constr_value, roi),
        )

        pop = pop[:n_samples]
        # Update set to match trimmed population
        pop_set = {tuple(p) for p in pop}

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
