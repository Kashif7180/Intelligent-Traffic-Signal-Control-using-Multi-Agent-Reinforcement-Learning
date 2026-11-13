"""
Verify that all 11 controller configurations initialize and step properly
in the Phase10UnifiedGUIDashboard engine.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import tkinter as tk
from visualize_phase10_gui import Phase10UnifiedGUIDashboard, CONTROLLER_DISPLAY_NAMES

def test_all_11_controllers():
    configs = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "A1", "A2"]
    root = tk.Tk()
    root.withdraw()

    print(f"Testing all {len(configs)} controllers in GUI dashboard engine...")

    for cid in configs:
        print(f"\n--> Testing {cid}: {CONTROLLER_DISPLAY_NAMES[cid]}")
        app = Phase10UnifiedGUIDashboard(
            root=root,
            config_path="config.yaml",
            initial_controller_id=cid,
            initial_seed=1001,
            max_steps=2,
            use_sumo_gui=False,
            smoke_test=True
        )
        # Step twice
        for s in range(2):
            app.execute_decision_step()
        
        info_a = app.last_info["A"]["metrics"]
        print(f"    Success: step={app.current_step}, queue={info_a['queue_length']:.1f}, wait={info_a['waiting_time']:.1f}s, interventions={app.cumulative_interventions}")
        app.env.close()

    root.destroy()
    print("\nAll 11 controller configurations verified successfully in GUI engine!")

if __name__ == "__main__":
    test_all_11_controllers()
