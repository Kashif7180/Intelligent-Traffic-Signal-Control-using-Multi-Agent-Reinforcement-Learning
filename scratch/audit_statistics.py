import csv
import numpy as np
import scipy.stats as stats

with open("logs/phase10_master_benchmark.csv", "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Total rows in benchmark CSV: {len(rows)}")

# Verify rows per controller
controllers = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "A1", "A2"
]

print(f"\nRow Counts & Uniqueness Check:")
for c in controllers:
    c_rows = [r for r in rows if r["controller_id"] == c]
    seeds = [int(r["seed"]) for r in c_rows]
    unique_seeds = len(set(seeds))
    print(f"  {c:<3}: total_rows={len(c_rows)}, unique_seeds={unique_seeds}, seeds={sorted(seeds)}")

# Statistical verification
print(f"\n{'ID':<3} | {'Delay Mean':<10} | {'Delay SD':<10} | {'Delay Median':<12} | {'Delay IQR':<10} | {'95% CI':<22} | {'Queue Mean':<10} | {'Thru Mean':<10}")
print("-" * 115)

stats_table = []
for c in controllers:
    c_rows = [r for r in rows if r["controller_id"] == c]
    delays = np.array([float(r["mean_delay"]) for r in c_rows])
    queues = np.array([float(r["mean_queue"]) for r in c_rows])
    thru = np.array([float(r["throughput"]) for r in c_rows])
    
    mean_d = np.mean(delays)
    std_d = np.std(delays, ddof=0)
    med_d = np.median(delays)
    iqr_d = np.percentile(delays, 75) - np.percentile(delays, 25)
    
    # 95% Student-t CI
    n = len(delays)
    se_d = np.std(delays, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_low = mean_d - t_crit * se_d
    ci_high = mean_d + t_crit * se_d
    
    mean_q = np.mean(queues)
    mean_t = np.mean(thru)
    
    ci_str = f"[{ci_low:.2f}, {ci_high:.2f}]"
    print(f"{c:<3} | {mean_d:<10.2f} | {std_d:<10.2f} | {med_d:<12.2f} | {iqr_d:<10.2f} | {ci_str:<22} | {mean_q:<10.2f} | {mean_t:<10.1f}")

