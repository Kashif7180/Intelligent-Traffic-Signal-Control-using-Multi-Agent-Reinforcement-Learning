import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load master benchmark CSVs
summary_df = pd.read_csv("logs/phase10_master_summary.csv")
raw_df = pd.read_csv("logs/phase10_master_benchmark.csv")

# Set up style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 15,
    "figure.titleweight": "bold"
})

fig, axes = plt.subplots(2, 3, figsize=(20, 11), dpi=300)
fig.patch.set_facecolor("#ffffff")

controllers = summary_df["id"].tolist()
names = [f"{row['id']}\n({row['name'][:10]}..)" if len(row['name']) > 12 else f"{row['id']}\n{row['name']}" for _, row in summary_df.iterrows()]
colors = [
    "#7f7f7f",  # C1: Gray
    "#1f77b4",  # C2: Blue
    "#2ca02c",  # C3: Green
    "#d62728",  # C4: Red
    "#9467bd",  # C5: Purple
    "#8c564b",  # C6: Brown
    "#e377c2",  # C7: Pink
    "#ff7f0e",  # C8: Orange
    "#2ca02c",  # C9: Deep Green (Shielded)
    "#17becf",  # A1: Cyan
    "#bcbd22",  # A2: Olive
]

# Panel 1: Average Vehicle Delay
ax1 = axes[0, 0]
delays = summary_df["delay_mean"].values
delay_err = summary_df["delay_std"].values
bars1 = ax1.bar(controllers, delays, yerr=delay_err, capsize=4, color=colors, edgecolor="#333333", alpha=0.85)
ax1.set_title("1. Average Vehicle Delay (s)")
ax1.set_ylabel("Mean Delay (seconds/veh)")
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.axhline(delays[0], color="red", linestyle=":", label="Fixed-Time Baseline (C1)")
ax1.set_ylim(0, max(delays) * 1.15)
for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, h + 3, f"{h:.1f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax1.legend(loc="upper right", frameon=True)

# Panel 2: Mean Queue Length
ax2 = axes[0, 1]
queues = summary_df["queue_mean"].values
queue_err = summary_df["queue_std"].values
bars2 = ax2.bar(controllers, queues, yerr=queue_err, capsize=4, color=colors, edgecolor="#333333", alpha=0.85)
ax2.set_title("2. Mean Intersection Queue Length (veh)")
ax2.set_ylabel("Queued Vehicles per Intersection")
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.set_ylim(0, max(queues) * 1.2)
for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

# Panel 3: Network Throughput
ax3 = axes[0, 2]
thru = summary_df["throughput_mean"].values
thru_err = summary_df["throughput_std"].values
bars3 = ax3.bar(controllers, thru, yerr=thru_err, capsize=4, color=colors, edgecolor="#333333", alpha=0.85)
ax3.set_title("3. Network Throughput (Completed Vehicles)")
ax3.set_ylabel("Total Completed Vehicles / 100 Steps")
ax3.grid(axis="y", linestyle="--", alpha=0.4)
ax3.set_ylim(0, max(thru) * 1.25)
for bar in bars3:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{int(h)}", ha="center", va="bottom", fontsize=8, fontweight="bold")

# Panel 4: Cumulative Episode Return
ax4 = axes[1, 0]
returns = summary_df["return_mean"].values
return_err = summary_df["return_std"].values
bars4 = ax4.bar(controllers, returns, yerr=return_err, capsize=4, color=colors, edgecolor="#333333", alpha=0.85)
ax4.set_title("4. Cumulative Multi-Agent Return")
ax4.set_ylabel("Cumulative Episode Return")
ax4.grid(axis="y", linestyle="--", alpha=0.4)
for bar in bars4:
    h = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2, h - 180, f"{int(h)}", ha="center", va="top", fontsize=8, fontweight="bold")

# Panel 5: Communication Efficiency & Bandwidth Ratio
ax5 = axes[1, 1]
comm_ratios = summary_df["comm_ratio_mean"].values * 100
bars5 = ax5.bar(controllers, comm_ratios, color=colors, edgecolor="#333333", alpha=0.85)
ax5.set_title("5. Communication Transmission Rate (%)")
ax5.set_ylabel("Actual Transmissions / Total Pairs (%)")
ax5.grid(axis="y", linestyle="--", alpha=0.4)
ax5.set_ylim(0, 115)
for bar in bars5:
    h = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{h:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

# Panel 6: Safety Shield Interventions & Conflict Violations
ax6 = axes[1, 2]
interventions = summary_df["interventions_mean"].values
conflicts = summary_df["conflicts_sum"].values
x_pos = np.arange(len(controllers))
width = 0.4
b1 = ax6.bar(x_pos - width/2, interventions, width, label="Safety Interventions", color="#ff7f0e", edgecolor="#333333", alpha=0.85)
b2 = ax6.bar(x_pos + width/2, conflicts, width, label="Signal Conflicts (Green Clashes)", color="#d62728", edgecolor="#333333", alpha=0.85)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(controllers)
ax6.set_title("6. Runtime Safety & Physical Invariant Violations")
ax6.set_ylabel("Count per Episode")
ax6.grid(axis="y", linestyle="--", alpha=0.4)
ax6.set_ylim(0, max(interventions) * 1.25)
ax6.legend(loc="upper right", frameon=True)
for bar in b1:
    h = bar.get_height()
    if h > 0:
        ax6.text(bar.get_x() + bar.get_width() / 2, h + 4, f"{int(h)}", ha="center", va="bottom", fontsize=8, fontweight="bold")
for bar in b2:
    h = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width() / 2, h + 4, f"{int(h)}", ha="center", va="bottom", fontsize=8, fontweight="bold", color="darkred")

# Polish layout
plt.suptitle("Intelligent Traffic Signal Control using Multi-Agent Reinforcement Learning\nComprehensive 11-Controller Master Benchmark Evaluation (10-Seed Average)", y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

os.makedirs("report_images", exist_ok=True)
os.makedirs("plots", exist_ok=True)
plt.savefig("report_images/07_six_panel_benchmark_comparison.png", dpi=300)
plt.savefig("plots/07_six_panel_benchmark_comparison.png", dpi=300)
plt.close()
print("Saved 6-panel benchmark comparison to report_images/07_six_panel_benchmark_comparison.png and plots/")

