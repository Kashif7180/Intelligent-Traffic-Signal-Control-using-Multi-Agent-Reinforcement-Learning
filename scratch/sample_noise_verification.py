"""
Forensic observation and policy probability sampling across seeds and agents.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import yaml
import numpy as np
import torch
from environment.traffic_env import MultiAgentTrafficEnv
from benchmark_phase12_noise import load_phase12_controllers, apply_feature_aware_noise, CONTINUOUS_FEATURE_INDICES, SIGNAL_STATE_INDICES

def sample_audit():
    controllers = load_phase12_controllers("config.yaml")
    c3 = controllers[1]["obj"] # MAPPO
    c8 = controllers[3]["obj"] # Causal Improved

    seeds = [3001, 3003, 3005]
    env = MultiAgentTrafficEnv(config_path="config.yaml", gui=False)

    print("=" * 80)
    print("FORENSIC SAMPLE AUDIT ACROSS MULTIPLE SEEDS AND AGENTS")
    print("=" * 80)

    try:
        for seed in seeds:
            route_file = f"network/route_cache_phase12/routes_medium_seed_{seed}.rou.xml"
            obs, info = env.reset(seed=seed, options={"safety_enabled": False, "route_file": route_file})

            # Run 5 initial steps to reach realistic traffic conditions
            for _ in range(5):
                actions = {a: 0 for a in env.agents}
                obs, _, _, _, info = env.step(actions)

            # Generate noise for CLEAN, MODERATE, HIGH
            rng_clean = np.random.Generator(np.random.PCG64(seed * 1000 + 0))
            rng_mod   = np.random.Generator(np.random.PCG64(seed * 1000 + 1))
            rng_high  = np.random.Generator(np.random.PCG64(seed * 1000 + 2))

            o_clean = apply_feature_aware_noise(obs, 0.00, rng_clean)
            o_mod   = apply_feature_aware_noise(obs, 0.05, rng_mod)
            o_high  = apply_feature_aware_noise(obs, 0.15, rng_high)

            print(f"\n--- SEED {seed} (Step 5 Traffic State) ---")
            eye_4 = torch.eye(4)

            # Measure observation perturbations
            for a in ["A", "B", "C", "D"]:
                diff_mod = np.max(np.abs(o_mod[a][CONTINUOUS_FEATURE_INDICES] - o_clean[a][CONTINUOUS_FEATURE_INDICES]))
                mean_diff_mod = np.mean(np.abs(o_mod[a][CONTINUOUS_FEATURE_INDICES] - o_clean[a][CONTINUOUS_FEATURE_INDICES]))
                diff_high = np.max(np.abs(o_high[a][CONTINUOUS_FEATURE_INDICES] - o_clean[a][CONTINUOUS_FEATURE_INDICES]))
                mean_diff_high = np.mean(np.abs(o_high[a][CONTINUOUS_FEATURE_INDICES] - o_clean[a][CONTINUOUS_FEATURE_INDICES]))

                # Internal signal state diff (must be strictly 0)
                sig_diff_mod = np.max(np.abs(o_mod[a][SIGNAL_STATE_INDICES] - o_clean[a][SIGNAL_STATE_INDICES]))
                sig_diff_high = np.max(np.abs(o_high[a][SIGNAL_STATE_INDICES] - o_clean[a][SIGNAL_STATE_INDICES]))

                print(f"Agent {a} Obs Diff: MODERATE max={diff_mod:.4f} (mean={mean_diff_mod:.4f}), HIGH max={diff_high:.4f} (mean={mean_diff_high:.4f}) | SigState Diff: mod={sig_diff_mod:.1f}, high={sig_diff_high:.1f}")

            # Check C3 MAPPO policy probabilities under CLEAN, MODERATE, HIGH
            for cond_name, o_dict in [("CLEAN", o_clean), ("MODERATE", o_mod), ("HIGH", o_high)]:
                inputs = [torch.cat([torch.tensor(o_dict[a], dtype=torch.float32), eye_4[i]], dim=-1) for i, a in enumerate(["A", "B", "C", "D"])]
                tensor_in = torch.stack(inputs, dim=0)
                with torch.no_grad():
                    dist = c3.shared_actor(tensor_in)
                    probs = dist.probs.cpu().numpy()
                    argmax_acts = np.argmax(probs, axis=-1)

                print(f"  {cond_name:<8} C3 Policy:")
                for i, a in enumerate(["A", "B", "C", "D"]):
                    p = probs[i]
                    p_switch = p[1]
                    p_alt = np.max([p[0], p[2], p[3]])
                    margin = p_switch - p_alt
                    print(f"    Node {a}: P(SWITCH)={p_switch:.4f}, AltMax={p_alt:.4f}, Margin={margin:+.4f}, argmax={argmax_acts[i]}")

    finally:
        env.close()

if __name__ == "__main__":
    sample_audit()
