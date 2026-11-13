import csv

with open('logs/phase10_master_benchmark.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total rows: {len(rows)}")
print("Seed 1001 Comparison Across Controllers:")
print(f"{'ID':<3} | {'Controller':<30} | {'Delay':<7} | {'Queue':<6} | {'Thru':<5} | {'Keep %':<7} | {'Switch %':<8} | {'Extend %':<8} | {'Reduce %':<8} | {'Entropy':<7}")
print("-" * 105)
for r in rows:
    if r['seed'] == '1001':
        print(f"{r['controller_id']:<3} | {r['controller_name']:<30} | {r['mean_delay']:<7} | {r['mean_queue']:<6} | {r['throughput']:<5} | {r['action_keep_pct']:<7} | {r['action_switch_pct']:<8} | {r['action_extend_pct']:<8} | {r['action_reduce_pct']:<8} | {r['policy_entropy']:<7}")

