import argparse
import json
import os

import matplotlib.pyplot as plt


def _load_history(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "runs" in data:
        return data["runs"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported dashboard data format in {path}")


def _run_label(run: dict):
    method = run.get("method", "unknown")
    seed = run.get("seed")
    if seed is None:
        return method
    return f"{method} (seed={seed})"


def _plot_acc_vs_constr(runs: list, objective: str, constraint: str, output_path: str):
    plt.figure(figsize=(8, 5))
    has_data = False
    for run in runs:
        points = run.get("history", [])
        xs = [p.get("constraint") for p in points if p.get("constraint") is not None and p.get("accuracy") is not None]
        ys = [p.get("accuracy") for p in points if p.get("constraint") is not None and p.get("accuracy") is not None]
        if not xs:
            continue
        has_data = True
        plt.plot(xs, ys, marker="o", linewidth=1.2, markersize=3, label=_run_label(run))

    if not has_data:
        raise ValueError("No valid history points with both constraint and accuracy.")

    plt.xlabel(constraint)
    plt.ylabel(objective)
    plt.title(f"{objective} vs {constraint}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


def _print_top_configs(runs: list, topk: int):
    rows = []
    for run in runs:
        for p in run.get("history", []):
            acc = p.get("accuracy")
            constr = p.get("constraint")
            scheme = p.get("scheme")
            if acc is None or constr is None:
                continue
            rows.append({
                "method": run.get("method", "unknown"),
                "seed": run.get("seed"),
                "accuracy": acc,
                "constraint": constr,
                "scheme": scheme
            })
    rows.sort(key=lambda x: x["accuracy"], reverse=True)
    print(f"Top {min(topk, len(rows))} configurations by accuracy:")
    for i, row in enumerate(rows[:topk], 1):
        print(
            f"{i}. method={row['method']}, seed={row['seed']}, "
            f"acc={row['accuracy']:.6f}, constr={row['constraint']:.6f}, "
            f"scheme={row['scheme']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple dashboard for MPQ search histories.")
    parser.add_argument("input_json", type=str, help="Dashboard JSON file (from mpq_search/deploy scripts).")
    parser.add_argument("--output", type=str, default="output/figs/mpq_dashboard_acc_vs_constraint.png")
    parser.add_argument("--objective", type=str, default="accuracy")
    parser.add_argument("--constraint", type=str, default="constraint")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    runs = _load_history(args.input_json)
    _plot_acc_vs_constr(runs, args.objective, args.constraint, args.output)
    _print_top_configs(runs, args.topk)
