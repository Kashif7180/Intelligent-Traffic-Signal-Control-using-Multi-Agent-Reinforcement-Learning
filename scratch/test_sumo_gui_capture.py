import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import traci
from environment.traffic_env import get_sumo_binary

sumo_binary = get_sumo_binary(gui=True)
print(f"Using binary: {sumo_binary}")

# Start SUMO-GUI with 2x2 grid
sumo_cmd = [
    sumo_binary,
    "-c", "network/simulation.sumocfg",
    "--start",
    "--quit-on-end",
    "--window-size", "1280,800",
    "--delay", "50"
]

traci.start(sumo_cmd)
try:
    # Run 40 simulation steps so vehicles enter and populate the 2x2 network
    for step in range(40):
        traci.simulationStep()
    
    # Configure view
    views = traci.gui.getIDList()
    print("Available views:", views)
    if views:
        view_id = views[0]
        # 2x2 grid coordinates roughly center around (300, 300) or similar
        # Let's inspect boundary
        traci.gui.setZoom(view_id, 350.0)
        traci.gui.setOffset(view_id, 350.0, 350.0)
        time.sleep(0.5)
        os.makedirs("plots", exist_ok=True)
        out_path = os.path.abspath("plots/sumo_gui_2x2_network.png")
        traci.gui.screenshot(view_id, out_path)
        time.sleep(0.5)
        print(f"Screenshot requested to: {out_path}")
finally:
    traci.close()

if os.path.exists("plots/sumo_gui_2x2_network.png"):
    print("Screenshot exists! Size:", os.path.getsize("plots/sumo_gui_2x2_network.png"))
else:
    print("Screenshot file was not created by TraCI directly.")
