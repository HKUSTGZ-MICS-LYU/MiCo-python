# MiCo Search Algorithm Documentation

## Overview
The `MiCoSearcher` is an iterative search algorithm designed to find optimal mixed-precision quantization schemes for neural networks. It navigates the search space of layer-wise bitwidth configurations (e.g., 4-8 bits) to maximize a target objective (like accuracy) under strict hardware or efficiency constraints (like BOPs, model size, or latency).

It employs a **Regressor-based Optimization** strategy, where a surrogate model (Random Forest, XGBoost, or Gaussian Process) learns the mapping between bitwidth configurations and model performance.

## Algorithm Workflow

### 1. Initialization
- **Evaluator Setup**: Configures the `MiCoEval` instance with the target objective and constraints.
- **Initial Sampling**: Generates an initial set of random candidates (`n_inits`) using `grid_sample` (orthogonal sampling) or `random_sample_min_max` (boundary + random).
- **Initial Evaluation**: Evaluates these initial candidates to bootstrap the regressor training data (`sampled_X`, `sampled_y`).

### 2. Iterative Search
The algorithm runs for `n_iter` iterations. In each iteration:

1.  **ROI Adjustment**: The "Region of Interest" (ROI) for sampling expands linearly from 0.2 to 0.5 over the iterations. This controls how close to the constraint boundary we sample.
2.  **Candidate Sampling**:
    - Generates a large pool of candidate schemes (`NUM_SAMPLES`, default 1000).
    - **Method**: Uses `near-constr` sampling (Genetic Algorithm) to find candidates that satisfy the constraint and are close to the constraint boundary (where high-accuracy solutions typically lie).
    - **Selection**: Filters candidates to strictly ensure they satisfy `constr(x) <= constr_value`.
3.  **Surrogate Modeling (`optimize`)**:
    - Trains a regressor (e.g., Random Forest) on all history data (`sampled_X`, `sampled_y`).
    - Predicts scores for all generated candidates.
    - Selects the candidate with the highest predicted score that has *not* yet been evaluated.
4.  **Evaluation**:
    - Runs the actual evaluation (inference on validation set) for the selected candidate.
    - Updates the history data and the global best solution found so far.

## Key Components

### Regressors (`regressor`)
- **`rf` (Random Forest)**: Default. Robust and handles non-linear relationships well.
- **`xgb` (XGBoost)**: Gradient boosting trees.
- **`bayes` (Gaussian Process)**: Uses `botorch` with Log Expected Improvement (LogEI) acquisition.
- **`bayes_ensemble`**: Ensemble of Gaussian Processes.

### Near-Constraint Sampling (`near-constr`)
To efficiently find valid candidates under constraints, it uses a Genetic Algorithm (GA):
- **Initialization**: Starts with the top 25% best-performing schemes found so far.
- **Evolution**: Applies crossover and mutation to generate new schemes.
- **Selection**: Keeps schemes that are within a specific distance (`roi`) from the constraint boundary.
- **Goal**: Concentrate search in the feasible region near the constraint limit, where optimal trade-offs usually exist.

## Parameters
- `n_inits`: Number of initial random samples.
- `qtypes`: List of available bitwidths (e.g., `[4, 5, 6, 7, 8]`).
- `regressor`: Type of surrogate model (`rf`, `xgb`, `bayes`).
- `sample_method`: Sampling strategy (`near-constr`, `random`).
- `dim_trans`: Optional dimension transformation to reduce search space for very deep networks.

## Current Limitations & Improvements
- **Tight Constraints**: The rejection sampling mechanism can fail if the constraint is too tight, leading to `ValueError`.
- **Small Networks**: The default Random Forest parameters may overfit on low-dimensional search spaces (shallow networks).
- **Stability**: Performance variance can be high due to the stochastic nature of the genetic algorithm and regressor initialization.
