import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import tkinter as tk
from PIL import ImageGrab
from visualize_phase10_gui import Phase10UnifiedGUIDashboard, THEME

os.makedirs("report_images", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("[Capture] Initializing Tkinter Dashboard...")
root = tk.Tk()
# Use headless/no sumo-gui for Tkinter capture so SUMO doesn't compete for screen focus
app = Phase10UnifiedGUIDashboard(
    root=root,
    config_path="config.yaml",
    initial_controller_id="C9",
    initial_seed=1001,
    max_steps=100,
    use_sumo_gui=False,
    auto_run=False,
    smoke_test=False
)

# Step 10 times to populate rich metrics, queues, attention weights, communication gates, and safety events
print("[Capture] Stepping simulation 15 times to populate live traffic state...")
for i in range(15):
    app.on_step()
    root.update()

root.update_idletasks()
root.update()
time.sleep(1.0)

# Get window bounding box
x = root.winfo_rootx()
y = root.winfo_rooty()
w = root.winfo_width()
h = root.winfo_height()
print(f"[Capture] Window geometry: x={x}, y={y}, w={w}, h={h}")

# 1. Capture Full Unified Dashboard (showing Controller C9, Live Intersection State, Topo Canvas, Cards)
bbox = (x, y, x + w, y + h)
img_main = ImageGrab.grab(bbox=bbox)
img_main.save("report_images/02_unified_tkinter_dashboard.png")
img_main.save("plots/02_unified_tkinter_dashboard.png")
print("[Capture] Saved: report_images/02_unified_tkinter_dashboard.png")

# 2. Switch to XAI Tab and select Node A
print("[Capture] Switching to XAI Tab...")
app.notebook.select(3)  # Tab index 3 is XAI Tab
app.on_xai_agent_selected("A")
root.update_idletasks()
root.update()
time.sleep(0.8)

img_xai = ImageGrab.grab(bbox=bbox)
img_xai.save("report_images/03_xai_tab_dashboard.png")
img_xai.save("plots/03_xai_tab_dashboard.png")
print("[Capture] Saved: report_images/03_xai_tab_dashboard.png")

# 3. Switch to Architecture & Deep Telemetry Tab (Comm-gate and causal-influence telemetry)
print("[Capture] Switching to Architecture Tab...")
app.notebook.select(0)  # Tab index 0 is Architecture Tab
root.update_idletasks()
root.update()
time.sleep(0.8)

img_arch = ImageGrab.grab(bbox=bbox)
img_arch.save("report_images/05_comm_causal_telemetry_dashboard.png")
img_arch.save("plots/05_comm_causal_telemetry_dashboard.png")
print("[Capture] Saved: report_images/05_comm_causal_telemetry_dashboard.png")

# 4. Switch to Safety Shield & Event Log Tab
print("[Capture] Switching to Event Log Tab...")
app.notebook.select(2)  # Tab index 2 is Event Log Tab
root.update_idletasks()
root.update()
time.sleep(0.8)

img_safety = ImageGrab.grab(bbox=bbox)
img_safety.save("report_images/06_safety_intervention_log_dashboard.png")
img_safety.save("plots/06_safety_intervention_log_dashboard.png")
print("[Capture] Saved: report_images/06_safety_intervention_log_dashboard.png")

# Teardown
app.env.close()
root.destroy()
print("[Capture] Completed all Tkinter dashboard captures successfully!")
