import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

folder = Path(sys.argv[1])
groups = defaultdict(list)
for path in sorted(folder.glob("*.csv")):
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = tuple(row[k] for k in ("dim", "width", "distribution", "backend", "precision", "threads", "keep_budget"))
            groups[key].append(row)

if not groups:
    raise SystemExit("No benchmark CSV files found")

def median(rows, key):
    return statistics.median(float(r[key]) for r in rows)

def table(filtered):
    text = ["| Layer | Input | Backend | Budget | Actually kept | Dense ms | Sparse total ms | Speedup (seed range) | Relative MSE | Cosine |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, rows in sorted(filtered):
        d, n, dist, backend, precision, threads, keep = key
        label = f"CPU {threads}t" if backend == "cpu" else f"GPU {precision}"
        speeds = [float(r["speedup"]) for r in rows]
        text.append(f"| {d} x {n} | {dist} | {label} | {float(keep):.0%} | "
                    f"{median(rows, 'kept_mean'):.1%} | {median(rows, 'dense_p50_ms'):.3f} | "
                    f"{median(rows, 'total_p50_ms'):.3f} | {statistics.median(speeds):.2f}x "
                    f"({min(speeds):.2f}–{max(speeds):.2f}) | {median(rows, 'relative_mse'):.3f} | "
                    f"{median(rows, 'cosine'):.3f} |")
    return "\n".join(text)

all_rows = list(groups.items())
report = "# Measured results\n\n"
report += ("Per cell: median across seeds of each run's per-query median latency; quality is the median of each "
           "run's mean over input vectors. Speedup range is min–max across seeds, not a confidence interval. "
           "GPU total includes CPU selection and host/device transfers. See machine.json, sources.json, and raw CSVs.\n\n")
report += "## All configurations\n\n" + table(all_rows) + "\n"
(folder / "summary.md").write_text(report, encoding="utf-8")
focus = [(k, v) for k, v in all_rows if k[2] == "gaussian" and abs(float(k[6])-.05) < 1e-9]
print(table(focus))
print("\nRuns:", len(list(folder.glob("*.csv"))), "Configurations:", sum(map(len, groups.values())))
