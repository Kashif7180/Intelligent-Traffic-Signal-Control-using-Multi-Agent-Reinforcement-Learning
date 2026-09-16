"""
Realistic demand and route generator for the 2x2 grid network.
Configurable via config.yaml, supporting 'low', 'normal', and 'peak' demand levels.
"""

import os
import random
import yaml
import xml.etree.ElementTree as ET
from xml.dom import minidom
import networkx as nx


def build_edge_graph():
    """Build directed edge connectivity graph for the 2x2 grid."""
    G = nx.DiGraph()

    # Connections at Intersection A
    for inc in ['north_to_A', 'west_to_A', 'B_to_A', 'C_to_A']:
        for out in ['A_to_north', 'A_to_west', 'A_to_B', 'A_to_C']:
            if inc == 'north_to_A' and out == 'A_to_north': continue
            if inc == 'west_to_A' and out == 'A_to_west': continue
            if inc == 'B_to_A' and out == 'A_to_B': continue
            if inc == 'C_to_A' and out == 'A_to_C': continue
            G.add_edge(inc, out)

    # Connections at Intersection B
    for inc in ['north_to_B', 'east_to_B', 'A_to_B', 'D_to_B']:
        for out in ['B_to_north', 'B_to_east', 'B_to_A', 'B_to_D']:
            if inc == 'north_to_B' and out == 'B_to_north': continue
            if inc == 'east_to_B' and out == 'B_to_east': continue
            if inc == 'A_to_B' and out == 'B_to_A': continue
            if inc == 'D_to_B' and out == 'B_to_D': continue
            G.add_edge(inc, out)

    # Connections at Intersection C
    for inc in ['south_to_C', 'west_to_C', 'A_to_C', 'D_to_C']:
        for out in ['C_to_south', 'C_to_west', 'C_to_A', 'C_to_D']:
            if inc == 'south_to_C' and out == 'C_to_south': continue
            if inc == 'west_to_C' and out == 'C_to_west': continue
            if inc == 'A_to_C' and out == 'C_to_A': continue
            if inc == 'D_to_C' and out == 'C_to_D': continue
            G.add_edge(inc, out)

    # Connections at Intersection D
    for inc in ['south_to_D', 'east_to_D', 'C_to_D', 'B_to_D']:
        for out in ['D_to_south', 'D_to_east', 'D_to_C', 'D_to_B']:
            if inc == 'south_to_D' and out == 'D_to_south': continue
            if inc == 'east_to_D' and out == 'D_to_east': continue
            if inc == 'C_to_D' and out == 'D_to_C': continue
            if inc == 'B_to_D' and out == 'D_to_B': continue
            G.add_edge(inc, out)

    return G


def get_all_routes():
    """Extract realistic acyclic shortest and natural paths between all entry and exit edges."""
    G = build_edge_graph()
    entries = ['north_to_A', 'west_to_A', 'north_to_B', 'east_to_B', 'west_to_C', 'south_to_C', 'south_to_D', 'east_to_D']
    exits = ['A_to_north', 'A_to_west', 'B_to_north', 'B_to_east', 'C_to_west', 'C_to_south', 'D_to_south', 'D_to_east']

    routes_by_entry = {e: [] for e in entries}
    for entry in entries:
        for exit_edge in exits:
            # Avoid U-turn direct paths (e.g. entering from north_A and immediately turning back north_A)
            entry_origin = entry.split('_to_')[0]
            exit_dest = exit_edge.split('_to_')[1]
            if entry_origin == exit_dest:
                continue

            try:
                # Find all simple paths up to cutoff length 4 edges
                paths = list(nx.all_simple_paths(G, source=entry, target=exit_edge, cutoff=4))
                if paths:
                    # Prefer shortest paths
                    min_len = min(len(p) for p in paths)
                    for p in paths:
                        if len(p) <= min_len + 1:
                            routes_by_entry[entry].append(p)
            except nx.NetworkXNoPath:
                continue

    return routes_by_entry


def generate_routes(config_path: str = "config.yaml", demand_level: str = None, seed: int = None) -> str:
    """
    Generate SUMO route XML file based on configuration.
    
    Args:
        config_path: Path to config.yaml
        demand_level: Override demand level ('low', 'normal', 'peak')
        seed: Override random seed
        
    Returns:
        Path to generated route file
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.abspath(config_path))
    rou_file = os.path.join(base_dir, config["network"]["rou_file"])
    os.makedirs(os.path.dirname(rou_file), exist_ok=True)

    if demand_level is None:
        demand_level = config["demand"]["active_demand"]
    if demand_level not in config["demand"]["profiles"]:
        raise ValueError(f"Unknown demand level '{demand_level}'. Available: {list(config['demand']['profiles'].keys())}")

    profile = config["demand"]["profiles"][demand_level]
    prob_insertion = float(profile["prob_insertion_per_step"])
    sim_duration = int(config["demand"]["simulation_duration"])

    if seed is None:
        seed = int(config["simulation"]["seed"])
    rng = random.Random(seed)

    routes_by_entry = get_all_routes()
    car_cfg = config["demand"]["vehicle_types"]["car"]

    root = ET.Element("routes")
    ET.SubElement(
        root,
        "vType",
        id="car",
        accel=str(car_cfg["accel"]),
        decel=str(car_cfg["decel"]),
        sigma=str(car_cfg["sigma"]),
        length=str(car_cfg["length"]),
        minGap=str(car_cfg["minGap"]),
        maxSpeed=str(car_cfg["maxSpeed"])
    )

    # Register named routes
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

    # Generate vehicle trips across simulation duration
    veh_count = 0
    for t in range(sim_duration):
        for entry, path_list in routes_by_entry.items():
            if not path_list:
                continue
            if rng.random() < prob_insertion:
                chosen_path = rng.choice(path_list)
                path_str = " ".join(chosen_path)
                r_id = route_map[path_str]
                ET.SubElement(
                    root,
                    "vehicle",
                    id=f"veh_{veh_count}",
                    type="car",
                    route=r_id,
                    depart=str(t),
                    departLane="best",
                    departSpeed="max"
                )
                veh_count += 1

    # Format XML with indentation
    xml_str = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="    ")
    # Clean up empty lines created by toprettyxml
    cleaned_xml = "\n".join([line for line in xml_str.splitlines() if line.strip()])

    with open(rou_file, "w", encoding="utf-8") as f:
        f.write(cleaned_xml)

    print(f"Generated route file for demand '{demand_level}' with {veh_count} vehicles over {sim_duration}s -> {rou_file}")
    return rou_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate realistic traffic demand routes")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--demand", default=None, choices=["low", "normal", "peak"], help="Demand profile")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    generate_routes(args.config, args.demand, args.seed)
