import csv
import math
import numpy as np
import os
import hashlib

# 1. Read actual 110-row benchmark CSV
benchmark_csv = "logs/phase10_master_benchmark.csv"
summary_csv = "logs/phase10_master_summary.csv"

with open(benchmark_csv, "r", encoding="utf-8") as f:
    raw_rows = list(csv.DictReader(f))

print(f"Total rows in benchmark CSV: {len(raw_rows)}")

configs = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "A1", "A2"]

# Check unique config-seed combinations
pairs = [(r["controller_id"], int(r["seed"])) for r in raw_rows]
unique_pairs = set(pairs)
print(f"Total config-seed pairs: {len(pairs)}, Unique pairs: {len(unique_pairs)}")

# 2 & 3: Print raw delays and calculate statistics
print("\n" + "="*80)
print("RAW DELAYS AND INDEPENDENT RECOMPUTATION")
print("="*80)

t_crit_9 = 2.2621571627409915  # scipy.stats.t.ppf(0.975, df=9)

recomputed = {}

for c in configs:
    c_rows = [r for r in raw_rows if r["controller_id"] == c]
    # sort by seed
    c_rows.sort(key=lambda r: int(r["seed"]))
    seeds = [int(r["seed"]) for r in c_rows]
    delays = [float(r["mean_delay"]) for r in c_rows]
    
    mean_val = float(np.mean(delays))
    sample_sd = float(np.std(delays, ddof=1))
    pop_sd = float(np.std(delays, ddof=0))
    median_val = float(np.median(delays))
    q25 = float(np.percentile(delays, 25))
    q75 = float(np.percentile(delays, 75))
    iqr_val = q75 - q25
    
    n = len(delays)
    se = sample_sd / math.sqrt(n)
    ci95_low = mean_val - t_crit_9 * se
    ci95_high = mean_val + t_crit_9 * se
    
    recomputed[c] = {
        "seeds": seeds,
        "delays": delays,
        "mean": mean_val,
        "sample_sd": sample_sd,
        "pop_sd": pop_sd,
        "median": median_val,
        "iqr": iqr_val,
        "ci95": (ci95_low, ci95_high)
    }
    
    delays_str = ", ".join(f"{d:.4f}" for d in delays)
    print(f"\nConfiguration {c}:")
    print(f"  Seeds: {seeds}")
    print(f"  Raw delays: [{delays_str}]")
    print(f"  Mean: {mean_val:.4f}")
    print(f"  Sample SD (ddof=1): {sample_sd:.4f}")
    print(f"  Pop SD (ddof=0):    {pop_sd:.4f}")
    print(f"  Median: {median_val:.4f}")
    print(f"  IQR (Q75 - Q25): {iqr_val:.4f} (Q25={q25:.4f}, Q75={q75:.4f})")
    print(f"  95% Student-t CI: [{ci95_low:.4f}, {ci95_high:.4f}]")

print("\n" + "="*80)
print("COMPARISON WITH phase10_master_summary.csv")
print("="*80)

with open(summary_csv, "r", encoding="utf-8") as f:
    summary_rows = list(csv.DictReader(f))

for s in summary_rows:
    cid = s["id"]
    s_mean = float(s["delay_mean"])
    s_std = float(s["delay_std"])
    comp = recomputed[cid]
    print(f"{cid:<3}: Summary CSV -> mean={s_mean:.4f}, std={s_std:.4f} | Recomputed -> mean={comp['mean']:.4f}, sample_sd={comp['sample_sd']:.4f}, pop_sd={comp['pop_sd']:.4f}")

