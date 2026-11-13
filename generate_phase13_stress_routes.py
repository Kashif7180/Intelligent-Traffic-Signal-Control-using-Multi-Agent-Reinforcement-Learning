"""
Phase 13 Stress Route Generator: Creates dedicated stress-test route files.
Scenarios:
1. Empty: 0 vehicles.
2. Hyper-Saturated (Extreme): p=0.50 (~2000 vehicles in 500s).
3. Asymmetric Bottleneck: Node A p=0.65, others p=0.04.
4. Sudden Shock Surge: p=0.08 for t < 150s, p=0.55 for t >= 150s.
5. Gridlock & Dissipation: p=0.50 for t < 250s, p=0.00 for t >= 250s.
"""

import os
import random
import yaml
import hashlib
import xml.etree.ElementTree as ET
from environment.demand_generator import get_all_routes

def generate_all_stress_routes(config_path="config.yaml", target_dir="network/route_cache_phase13_stress", seeds=[4001, 4002, 4003]):
    os.makedirs(target_dir, exist_ok=True)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    routes_by_entry = get_all_routes()
    car_cfg = config["demand"]["vehicle_types"]["car"]
    sim_duration = 500  # 500 seconds simulation duration

    def create_base_root():
        root = ET.Element("routes")
        ET.SubElement(
            root, "vType",
            id="car",
            accel=str(car_cfg["accel"]),
            decel=str(car_cfg["decel"]),
            sigma=str(car_cfg["sigma"]),
            length=str(car_cfg["length"]),
            minGap=str(car_cfg["minGap"]),
            maxSpeed=str(car_cfg["maxSpeed"])
        )
        route_id_counter = 0
        route_map = {}
        for entry, path_list in routes_by_entry.items():
            for path in path_list:
                path_str = " ".join(path)
                if path_str not in route_map:
                    r_id = f"r_{route_id_counter}"
                    route_id_counter += 1
                    route_map[path_str] = r_id
                    ET.SubElement(root, "route", id=r_id, edges=path_str)
        return root, route_map

    manifest = {}

    # 1. Empty route file
    empty_path = os.path.join(target_dir, "routes_empty.rou.xml")
    root, _ = create_base_root()
    ET.ElementTree(root).write(empty_path, encoding="utf-8", xml_declaration=True)
    manifest["empty"] = (empty_path, 0)

    # Generate multi-seed stress files
    for s in seeds:
        # 2. Extreme Hyper-Saturation (p = 0.50)
        rng = random.Random(s)
        root, route_map = create_base_root()
        veh_id = 0
        for t in range(sim_duration):
            for entry, path_list in routes_by_entry.items():
                if rng.random() < 0.50:
                    chosen = rng.choice(path_list)
                    p_str = " ".join(chosen)
                    ET.SubElement(root, "vehicle", id=f"veh_ext_{veh_id}", type="car", route=route_map[p_str], depart=f"{t:.1f}")
                    veh_id += 1
        ext_path = os.path.join(target_dir, f"routes_extreme_seed_{s}.rou.xml")
        ET.ElementTree(root).write(ext_path, encoding="utf-8", xml_declaration=True)
        manifest[f"extreme_{s}"] = (ext_path, veh_id)

        # 3. Asymmetric Bottleneck (Node A p=0.65, others p=0.04)
        rng = random.Random(s)
        root, route_map = create_base_root()
        veh_id = 0
        for t in range(sim_duration):
            for entry, path_list in routes_by_entry.items():
                p_arr = 0.65 if entry in ["north_to_A", "west_to_A"] else 0.04
                if rng.random() < p_arr:
                    chosen = rng.choice(path_list)
                    p_str = " ".join(chosen)
                    ET.SubElement(root, "vehicle", id=f"veh_asym_{veh_id}", type="car", route=route_map[p_str], depart=f"{t:.1f}")
                    veh_id += 1
        asym_path = os.path.join(target_dir, f"routes_asymmetric_seed_{s}.rou.xml")
        ET.ElementTree(root).write(asym_path, encoding="utf-8", xml_declaration=True)
        manifest[f"asymmetric_{s}"] = (asym_path, veh_id)

        # 4. Sudden Shock Surge (p=0.08 for t < 150s, p=0.55 for t >= 150s)
        rng = random.Random(s)
        root, route_map = create_base_root()
        veh_id = 0
        for t in range(sim_duration):
            p_arr = 0.08 if t < 150 else 0.55
            for entry, path_list in routes_by_entry.items():
                if rng.random() < p_arr:
                    chosen = rng.choice(path_list)
                    p_str = " ".join(chosen)
                    ET.SubElement(root, "vehicle", id=f"veh_surge_{veh_id}", type="car", route=route_map[p_str], depart=f"{t:.1f}")
                    veh_id += 1
        surge_path = os.path.join(target_dir, f"routes_surge_seed_{s}.rou.xml")
        ET.ElementTree(root).write(surge_path, encoding="utf-8", xml_declaration=True)
        manifest[f"surge_{s}"] = (surge_path, veh_id)

        # 5. Gridlock & Dissipation (p=0.50 for t < 250s, p=0.00 for t >= 250s)
        rng = random.Random(s)
        root, route_map = create_base_root()
        veh_id = 0
        for t in range(sim_duration):
            p_arr = 0.50 if t < 250 else 0.00
            if p_arr > 0:
                for entry, path_list in routes_by_entry.items():
                    if rng.random() < p_arr:
                        chosen = rng.choice(path_list)
                        p_str = " ".join(chosen)
                        ET.SubElement(root, "vehicle", id=f"veh_diss_{veh_id}", type="car", route=route_map[p_str], depart=f"{t:.1f}")
                        veh_id += 1
        diss_path = os.path.join(target_dir, f"routes_dissipation_seed_{s}.rou.xml")
        ET.ElementTree(root).write(diss_path, encoding="utf-8", xml_declaration=True)
        manifest[f"dissipation_{s}"] = (diss_path, veh_id)

    print(f"Generated {len(manifest)} Phase 13 stress route files in {target_dir}")
    for k, (p, cnt) in sorted(manifest.items()):
        print(f"  {k}: {os.path.basename(p)} ({cnt} vehicles)")
    return manifest

if __name__ == "__main__":
    generate_all_stress_routes()

