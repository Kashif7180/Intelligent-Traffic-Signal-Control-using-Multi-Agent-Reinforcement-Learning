import csv
import numpy as np
import math

with open("logs/phase10_master_benchmark.csv", "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Total rows in benchmark CSV: {len(rows)}")

controllers = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "A1", "A2"]

print(f"\nRow Counts & Uniqueness Check:")
for c in controllers:
    c_rows = [r for r in rows if r["controller_id"] == c]
    seeds = [int(r["seed"]) for r in c_rows]
    unique_seeds = len(set(seeds))
    print(f"  {c:<3}: total_rows={len(c_rows)}, unique_seeds={unique_seeds}, seeds={sorted(seeds)}")

# Statistical verification
# For n=10, df=9, two-tailed t critical for 95% is 2.262
t_crit = 2.262

print(f"\n{'ID':<3} | {'Delay Mean':<10} | {'Delay SD':<10} | {'Delay Median':<12} | {'Delay IQR':<10} | {'95% CI':<20} | {'Queue Mean':<10} | {'Thru Mean':<10}")
print("-" * 115)

for c in controllers:
    c_rows = [r for r in rows if r["controller_id"] == c]
    delays = np.array([float(r["mean_delay"]) for r in c_rows])
    queues = np.array([float(r["mean_queue"]) for r in c_rows])
    thru = np.array([float(r["throughput"]) for r in c_rows])
    
    mean_d = float(np.mean(delays))
    std_d = float(np.std(delays, ddof=1))
    med_d = float(np.median(delays))
    iqr_d = float(np.percentile(delays, 75) - np.percentile(delays, 25))
    
    n = len(delays)
    se_d = std_d / math.sqrt(n)
    ci_low = mean_d - t_crit * se_d
    ci_high = mean_d + t_crit * se_d
    
    mean_q = float(np.mean(queues))
    mean_t = float(np.mean(thru))
    
    ci_str = f"[{ci_low:.2f}, {ci_high:.2f}]"
    print(f"{c:<3} | {mean_d:<10.2f} | {std_d:<10.2f} | {med_d:<12.2f} | {iqr_d:<10.2f} | {ci_str:<20} | {mean_q:<10.2f} | {mean_t:<10.1f}")

