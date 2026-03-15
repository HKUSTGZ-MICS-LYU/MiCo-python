import os
import re
import json
import argparse
import itertools
import numpy as np

from scipy.stats import spearmanr
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor


def parse_scheme_key(key: str):
    vals = re.findall(r"-?\d+", key)
    if not vals:
        return None
    return [int(v) for v in vals]


def parse_optional_int(token: str):
    lower = token.lower()
    if lower in ("none", "auto"):
        return None
    return int(token)


def parse_optional_number(token: str):
    lower = token.lower()
    if lower in ("none", "auto"):
        return None
    if "." in token or "e" in lower:
        return float(token)
    return int(token)


def parse_optional_str(token: str):
    lower = token.lower()
    if lower in ("none", "auto"):
        return None
    return token


def parse_list(raw: str, cast):
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        values.append(cast(token))
    return values


def to_label_value(v):
    return "auto" if v is None else str(v)


def load_eval_json(json_path: str, objective: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        if objective not in value:
            continue
        scheme = parse_scheme_key(key)
        if scheme is None:
            continue
        rows.append((tuple(scheme), float(value[objective])))

    if not rows:
        raise ValueError(f"No objective '{objective}' found in {json_path}.")

    rows.sort(key=lambda x: x[0])
    dims = len(rows[0][0])
    rows = [r for r in rows if len(r[0]) == dims]

    X = np.array([list(r[0]) for r in rows], dtype=float)
    y = np.array([r[1] for r in rows], dtype=float)
    return X, y, dims


def build_rf_kwargs(cfg: dict, dims: int, random_state: int, mimic_searcher_defaults: bool):
    kwargs = {}
    if mimic_searcher_defaults and dims > 10:
        kwargs.update({
            "n_estimators": 250,
            "max_depth": 15,
            "max_features": "sqrt",
        })

    if cfg["n_estimators"] is not None:
        kwargs["n_estimators"] = cfg["n_estimators"]
    if cfg["max_depth"] is not None:
        kwargs["max_depth"] = cfg["max_depth"]
    if cfg["max_features"] is not None:
        kwargs["max_features"] = cfg["max_features"]
    if cfg["min_samples_split"] is not None:
        kwargs["min_samples_split"] = cfg["min_samples_split"]
    if cfg["min_samples_leaf"] is not None:
        kwargs["min_samples_leaf"] = cfg["min_samples_leaf"]
    if random_state is not None:
        kwargs["random_state"] = random_state
    return kwargs


def score_with_cv(X, y, cfg, dims, folds, repeats, seed, mimic_searcher_defaults, progress=None):
    scores = []
    for rep in range(repeats):
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed + rep)
        for train_idx, test_idx in kf.split(X):
            model = RandomForestRegressor(
                **build_rf_kwargs(
                    cfg=cfg,
                    dims=dims,
                    random_state=seed + rep,
                    mimic_searcher_defaults=mimic_searcher_defaults,
                )
            )
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            corr, _ = spearmanr(y[test_idx], pred)
            if np.isnan(corr):
                corr = 0.0
            scores.append(float(corr))
            if progress is not None:
                progress.update(1)
    return scores


def build_experiments(args, dims):
    base = {
        "n_estimators": parse_optional_int(args.base_n_estimators),
        "max_depth": parse_optional_int(args.base_max_depth),
        "max_features": parse_optional_str(args.base_max_features),
        "min_samples_split": parse_optional_number(args.base_min_samples_split),
        "min_samples_leaf": parse_optional_number(args.base_min_samples_leaf),
    }

    n_estimators_vals = parse_list(args.n_estimators_values, parse_optional_int)
    max_depth_vals = parse_list(args.max_depth_values, parse_optional_int)
    max_features_vals = parse_list(args.max_features_values, parse_optional_str)
    min_split_vals = parse_list(args.min_samples_split_values, parse_optional_number)
    min_leaf_vals = parse_list(args.min_samples_leaf_values, parse_optional_number)

    experiments = {"base": base.copy()}

    if args.sweep_mode == "onefactor":
        for v in n_estimators_vals:
            if v == base["n_estimators"]:
                continue
            cfg = base.copy()
            cfg["n_estimators"] = v
            experiments[f"n_estimators={to_label_value(v)}"] = cfg

        for v in max_depth_vals:
            if v == base["max_depth"]:
                continue
            cfg = base.copy()
            cfg["max_depth"] = v
            experiments[f"max_depth={to_label_value(v)}"] = cfg

        for v in max_features_vals:
            if v == base["max_features"]:
                continue
            cfg = base.copy()
            cfg["max_features"] = v
            experiments[f"max_features={to_label_value(v)}"] = cfg

        for v in min_split_vals:
            if v == base["min_samples_split"]:
                continue
            cfg = base.copy()
            cfg["min_samples_split"] = v
            experiments[f"min_samples_split={to_label_value(v)}"] = cfg

        for v in min_leaf_vals:
            if v == base["min_samples_leaf"]:
                continue
            cfg = base.copy()
            cfg["min_samples_leaf"] = v
            experiments[f"min_samples_leaf={to_label_value(v)}"] = cfg

    elif args.sweep_mode == "grid":
        for ne, md, mf, mss, msl in itertools.product(
            n_estimators_vals,
            max_depth_vals,
            max_features_vals,
            min_split_vals,
            min_leaf_vals,
        ):
            cfg = {
                "n_estimators": ne,
                "max_depth": md,
                "max_features": mf,
                "min_samples_split": mss,
                "min_samples_leaf": msl,
            }
            label = (
                f"ne={to_label_value(ne)}|md={to_label_value(md)}|"
                f"mf={to_label_value(mf)}|mss={to_label_value(mss)}|msl={to_label_value(msl)}"
            )
            experiments[label] = cfg
    else:
        raise ValueError(f"Unsupported sweep_mode {args.sweep_mode}")

    return experiments


def main():
    parser = argparse.ArgumentParser(description="RF CV quality evaluation from MiCoEval JSON cache")
    parser.add_argument("json_path", type=str, help="Path to MiCoEval cache json file")
    parser.add_argument("--objective", type=str, default="ptq_acc", help="Objective key inside the JSON cache")
    parser.add_argument("--cv-folds", type=int, default=5, help="KFold splits")
    parser.add_argument("--cv-repeats", type=int, default=3, help="Number of repeated KFold runs")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--min-samples", type=int, default=20, help="Minimum required data points")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k configs to print")
    parser.add_argument("--sweep-mode", type=str, default="onefactor", choices=["onefactor", "grid"])
    parser.add_argument("--mimic-micosearcher-defaults", action="store_true",
                        help="Apply MiCoSearcher high-dim defaults (250/15/sqrt) before overrides")

    parser.add_argument("--base-n-estimators", type=str, default="auto")
    parser.add_argument("--base-max-depth", type=str, default="auto")
    parser.add_argument("--base-max-features", type=str, default="auto")
    parser.add_argument("--base-min-samples-split", type=str, default="auto")
    parser.add_argument("--base-min-samples-leaf", type=str, default="auto")

    parser.add_argument("--n-estimators-values", type=str, default="100,250,500")
    parser.add_argument("--max-depth-values", type=str, default="None,10,15,20")
    parser.add_argument("--max-features-values", type=str, default="None,sqrt,log2")
    parser.add_argument("--min-samples-split-values", type=str, default="2,4,8")
    parser.add_argument("--min-samples-leaf-values", type=str, default="1,2,4")

    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--out-txt", type=str, default=None)
    args = parser.parse_args()

    X, y, dims = load_eval_json(args.json_path, args.objective)
    if len(X) < args.min_samples:
        raise ValueError(f"Need >= {args.min_samples} samples, got {len(X)} from {args.json_path}")

    experiments = build_experiments(args, dims)
    print(f"Loaded {len(X)} evaluated schemes from {args.json_path} (dim={dims})")
    print(f"Objective: {args.objective}, experiments: {len(experiments)}")

    results = []
    total_steps = len(experiments) * args.cv_repeats * args.cv_folds
    with tqdm(total=total_steps, desc="RF CV", unit="fit") as pbar:
        for label, cfg in experiments.items():
            pbar.set_postfix_str(label)
            scores = score_with_cv(
                X=X,
                y=y,
                cfg=cfg,
                dims=dims,
                folds=args.cv_folds,
                repeats=args.cv_repeats,
                seed=args.seed,
                mimic_searcher_defaults=args.mimic_micosearcher_defaults,
                progress=pbar,
            )
            results.append({
                "label": label,
                "config": cfg,
                "mean_spearman": float(np.mean(scores)),
                "std_spearman": float(np.std(scores)),
                "fold_scores": scores,
            })

    results.sort(key=lambda x: x["mean_spearman"], reverse=True)
    print("\nTop configs by Spearman:")
    for i, r in enumerate(results[:args.top_k], 1):
        print(f"{i:2d}. {r['label']}: {r['mean_spearman']:.4f} ± {r['std_spearman']:.4f}")

    base_name = os.path.splitext(os.path.basename(args.json_path))[0]
    out_json = args.out_json or f"output/json/{base_name}_rf_cv_{args.objective}.json"
    out_txt = args.out_txt or f"output/txt/{base_name}_rf_cv_{args.objective}.txt"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)

    payload = {
        "json_path": args.json_path,
        "objective": args.objective,
        "num_samples": int(len(X)),
        "dims": int(dims),
        "cv_folds": args.cv_folds,
        "cv_repeats": args.cv_repeats,
        "seed": args.seed,
        "sweep_mode": args.sweep_mode,
        "mimic_micosearcher_defaults": args.mimic_micosearcher_defaults,
        "results": results,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    with open(out_txt, "w") as f:
        f.write("=== RF CV Quality (Spearman) ===\n")
        f.write(f"source={args.json_path}\n")
        f.write(f"objective={args.objective}\n")
        f.write(f"samples={len(X)}, dims={dims}, folds={args.cv_folds}, repeats={args.cv_repeats}\n\n")
        for i, r in enumerate(results, 1):
            f.write(
                f"{i:2d}. {r['label']}: mean={r['mean_spearman']:.6f}, "
                f"std={r['std_spearman']:.6f}, config={r['config']}\n"
            )

    print(f"\nSaved json: {out_json}")
    print(f"Saved txt : {out_txt}")


if __name__ == "__main__":
    main()
