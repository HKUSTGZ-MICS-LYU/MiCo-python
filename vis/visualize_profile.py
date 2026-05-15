#!/usr/bin/env python3
"""
Neural network operator profile visualizer

Input format example:
    [info] Benchmark Kernel 0: MiCo_bitconv2d_f32 occurrences=1 time=9805698 estimated=9805698

The script extracts each kernel's name, call count (occurrences), and total time.
It can optionally merge entries with the same operator name (--merge) and displays
the percentage of total time for each operator in the bar chart.
"""

import re
import sys
import argparse
import matplotlib.pyplot as plt

# Regex to match profile lines
PATTERN = re.compile(
    r'\[info\] Benchmark Kernel \d+:\s+(\S+)\s+occurrences=(\d+)\s+time=(\d+)'
)


def parse_lines(lines):
    """
    Parse profile data from text lines.
    Returns a list of tuples (kernel_name, occurrences, time_us).
    """
    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = PATTERN.search(line)
        if m:
            name = m.group(1)
            occurrences = int(m.group(2))
            time_us = int(m.group(3))
            data.append((name, occurrences, time_us))
    return data


def merge_by_name(data):
    """
    Merge entries with the same kernel name: sum occurrences and time.
    Returns a list of tuples (name, total_occurrences, total_time_us).
    """
    merged = {}
    for name, occ, t in data:
        if name in merged:
            merged[name][0] += occ
            merged[name][1] += t
        else:
            merged[name] = [occ, t]
    return [(name, occ, t) for name, (occ, t) in merged.items()]


def auto_scale_time(time_us):
    """Convert microseconds to a suitable unit (us, ms, s)."""
    if time_us >= 1_000_000:
        return time_us / 1_000_000.0, "s"
    elif time_us >= 1_000:
        return time_us / 1_000.0, "ms"
    else:
        return float(time_us), "us"


def format_label(name, occurrences, time_val, unit, percentage):
    """Build label for bar: name (xN): time unit (percentage%)"""
    return f"{name} (x{occurrences})\n{time_val:.2f} {unit} ({percentage:.1f}%)"


def plot_profile(data, output_file, merge_flag):
    """
    Plot horizontal bar chart showing kernel time consumption.
    If merge_flag is True, same-named kernels are merged.
    """
    if not data:
        print("Error: no valid profile data found.", file=sys.stderr)
        sys.exit(1)

    if merge_flag:
        data = merge_by_name(data)
        title_suffix = " (merged by name)"
    else:
        title_suffix = ""

    # Sort by total time descending
    data_sorted = sorted(data, key=lambda x: x[2], reverse=True)

    total_time = sum(t for _, _, t in data_sorted)

    names = []
    y_pos = []
    times_scaled = []
    labels = []

    # Determine global time unit from the largest entry
    max_time_us = data_sorted[0][2] if data_sorted else 0
    global_scale, global_unit = auto_scale_time(max_time_us)

    for i, (name, occ, t_us) in enumerate(data_sorted):
        # Scale time according to the global unit
        if global_unit == "s":
            scaled = t_us / 1_000_000.0
        elif global_unit == "ms":
            scaled = t_us / 1_000.0
        else:
            scaled = float(t_us)

        percentage = (t_us / total_time) * 100.0 if total_time > 0 else 0.0
        names.append(name)
        y_pos.append(i)
        times_scaled.append(scaled)
        labels.append(format_label(name, occ, scaled, global_unit, percentage))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(4, len(data) * 0.5)))
    bars = ax.barh(y_pos, times_scaled, color='steelblue', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # largest bar on top
    ax.set_xlabel(f"Time ({global_unit})")
    ax.set_title(f"Operator Profile Time Comparison{title_suffix}")
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Add labels at the end of each bar
    for bar, label in zip(bars, labels):
        width = bar.get_width()
        offset = 0.02 * max(times_scaled) if max(times_scaled) > 0 else 0.1
        ax.text(width + offset,
                bar.get_y() + bar.get_height() / 2,
                label, va='center', fontsize=9)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Chart saved to: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize neural network operator profile data (horizontal bar chart)."
    )
    parser.add_argument("file", nargs="?", help="Input file path (read from stdin if not provided)")
    parser.add_argument("-o", "--output", help="Save chart to file (e.g., plot.png); otherwise display interactively")
    parser.add_argument("--merge", action="store_true",
                        help="Merge entries with the same operator name (sum occurrences and time)")
    args = parser.parse_args()

    # Read data
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        print("Please provide profile data from stdin (Ctrl+D to end):", file=sys.stderr)
        lines = sys.stdin.readlines()

    data = parse_lines(lines)
    if not data:
        print("No matching profile lines found. Check input format.", file=sys.stderr)
        sys.exit(1)

    if args.merge:
        print(f"Parsed {len(data)} entries, merging to unique operator names...")
    else:
        print(f"Parsed {len(data)} entries (use --merge to combine same-named operators)")

    plot_profile(data, args.output, args.merge)


if __name__ == "__main__":
    main()