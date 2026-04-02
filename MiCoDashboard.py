import json
import os

from matplotlib import pyplot as plt


class MiCoDashboard:
    @staticmethod
    def default_live_plot_path(model_name: str, method: str, seed: int, output_dir: str = "output/figs"):
        return os.path.join(output_dir, f"{model_name}_{method}_seed{seed}_live.png")

    @staticmethod
    def make_live_plot_hook(evaluator, constraint_name: str, objective_label: str,
                            constraint_label: str, output_path: str, every: int = 1):
        if every <= 0:
            every = 1

        def _hook(searcher, _, __):
            n = len(searcher.best_trace)
            if n == 0 or (n % every) != 0:
                return
            history = MiCoDashboard.build_run_history(searcher, evaluator, constraint_name)
            run = MiCoDashboard.build_run_entry(
                method=getattr(searcher, "__class__", type(searcher)).__name__,
                seed=None,
                objective=objective_label,
                constraint_name=constraint_label,
                constraint_limit=getattr(searcher, "constr_value", 0.0) or 0.0,
                history=history
            )
            MiCoDashboard.plot_acc_vs_constr(
                [run], objective_label, constraint_label, output_path, close_figure=True
            )

        return _hook

    @staticmethod
    def build_run_history(searcher, evaluator, constraint_name: str):
        history = []
        eval_map = evaluator.eval_dict()
        if constraint_name not in eval_map:
            valid_names = ", ".join(sorted(eval_map.keys()))
            raise ValueError(
                f"Unsupported constraint name '{constraint_name}'. Must be one of: {valid_names}"
            )
        if not hasattr(searcher, "best_trace"):
            raise AttributeError(
                f"{searcher.__class__.__name__} is missing required attribute 'best_trace'."
            )
        if not hasattr(searcher, "best_scheme_trace"):
            raise AttributeError(
                f"{searcher.__class__.__name__} is missing required attribute 'best_scheme_trace'."
            )
        if len(searcher.best_trace) != len(searcher.best_scheme_trace):
            raise ValueError(
                f"{searcher.__class__.__name__}: mismatched trace lengths: "
                f"len(best_trace)={len(searcher.best_trace)}, "
                f"len(best_scheme_trace)={len(searcher.best_scheme_trace)}."
            )
        constr_eval = eval_map[constraint_name]
        for idx, best_acc in enumerate(searcher.best_trace):
            scheme = searcher.best_scheme_trace[idx]
            constr_val = constr_eval(scheme) if scheme is not None else None
            history.append({
                "iter": idx + 1,
                "accuracy": float(best_acc) if best_acc is not None else None,
                "constraint": float(constr_val) if constr_val is not None else None,
                "scheme": scheme
            })
        return history

    @staticmethod
    def build_run_entry(method: str, seed, objective: str, constraint_name: str,
                        constraint_limit: float, history: list):
        return {
            "method": method,
            "seed": seed,
            "objective": objective,
            "constraint_name": constraint_name,
            "constraint_limit": float(constraint_limit),
            "history": history
        }

    @staticmethod
    def save_runs(path: str, runs: list):
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"runs": runs}, f, indent=2)

    @staticmethod
    def load_runs(path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "runs" in data:
            return data["runs"]
        if isinstance(data, list):
            return data
        raise ValueError(f"Unsupported dashboard data format in {path}")

    @staticmethod
    def run_label(run: dict):
        method = run.get("method", "unknown")
        seed = run.get("seed")
        if seed is None:
            return method
        return f"{method} (seed={seed})"

    @staticmethod
    def plot_acc_vs_constr(runs: list, objective: str, constraint: str, output_path: str,
                           close_figure: bool = True):
        fig = plt.figure(figsize=(8, 5))
        has_data = False
        cmap = plt.cm.tab10
        color_idx = 0
        for run in runs:
            points = run.get("history", [])
            valid_points = [p for p in points if p.get("constraint") is not None and p.get("accuracy") is not None]
            xs = [p.get("constraint") for p in valid_points]
            ys = [p.get("accuracy") for p in valid_points]
            if not xs:
                continue
            has_data = True
            color = cmap(color_idx % 10)
            color_idx += 1
            plt.plot(
                xs, ys, marker="o", linewidth=1.2, markersize=3,
                label=MiCoDashboard.run_label(run), color=color, alpha=0.8
            )

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
        fig.savefig(output_path)
        if close_figure:
            plt.close(fig)
        print(f"Saved plot to {output_path}")

    @staticmethod
    def print_top_configs(runs: list, topk: int):
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
