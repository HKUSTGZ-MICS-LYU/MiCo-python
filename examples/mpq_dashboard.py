import argparse
from MiCoDashboard import MiCoDashboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple dashboard for MPQ search histories.")
    parser.add_argument("input_json", type=str, help="Dashboard JSON file (from mpq_search/deploy scripts).")
    parser.add_argument("--output", type=str, default="output/figs/mpq_dashboard_acc_vs_constraint.png")
    parser.add_argument("--objective", type=str, default="accuracy")
    parser.add_argument("--constraint", type=str, default="constraint")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    runs = MiCoDashboard.load_runs(args.input_json)
    MiCoDashboard.plot_acc_vs_constr(runs, args.objective, args.constraint, args.output)
    MiCoDashboard.print_top_configs(runs, args.topk)
