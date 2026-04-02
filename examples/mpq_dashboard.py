import argparse
from MiCoDashboard import MiCoDashboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple dashboard for MPQ search histories.")
    parser.add_argument("input_json", type=str, help="Dashboard JSON file (from mpq_search/deploy scripts).")
    parser.add_argument("--output", type=str, default="output/figs/mpq_dashboard_acc_vs_constraint.png")
    parser.add_argument("--objective", type=str, default=None,
                        help="Y-axis label (default: inferred from run metadata).")
    parser.add_argument("--constraint", type=str, default=None,
                        help="X-axis label (default: inferred from run metadata).")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    runs = MiCoDashboard.load_runs(args.input_json)
    if not runs:
        parser.error("No runs loaded from input JSON; cannot build dashboard.")

    default_objective = runs[0].get("objective", "accuracy")
    default_constraint = runs[0].get("constraint_name", "constraint")
    objective_label = args.objective or default_objective
    constraint_label = args.constraint or default_constraint

    MiCoDashboard.plot_acc_vs_constr(runs, objective_label, constraint_label, args.output)
    MiCoDashboard.print_top_configs(runs, args.topk)
