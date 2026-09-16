"""
Script to build a 2x2 grid network with 4 signalized intersections (A-B / C-D layout)
using SUMO's netconvert utility, configured via config.yaml.
"""

import os
import sys
import subprocess
import yaml
import sumolib


def get_sumo_binary(binary_name: str) -> str:
    """Resolve SUMO binary path from environment, sumolib, or eclipse-sumo package."""
    try:
        return sumolib.checkBinary(binary_name)
    except Exception:
        import sumo
        sumo_dir = os.path.dirname(sumo.__file__)
        exe_path = os.path.join(sumo_dir, "bin", f"{binary_name}.exe")
        if os.path.exists(exe_path):
            return exe_path
        raise FileNotFoundError(f"Could not find SUMO binary: {binary_name}")


def build_network(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    net_cfg = config["network"]
    intersections = net_cfg["intersections"]
    lane_len = float(net_cfg["lane_length"])
    bound_len = float(net_cfg["boundary_length"])
    speed_lim = float(net_cfg["speed_limit"])
    num_lanes = int(net_cfg["num_lanes"])

    base_dir = os.path.dirname(os.path.abspath(config_path))
    network_dir = os.path.join(base_dir, "network")
    os.makedirs(network_dir, exist_ok=True)

    nod_file = os.path.join(network_dir, "2x2_grid.nod.xml")
    edg_file = os.path.join(network_dir, "2x2_grid.edg.xml")
    net_file = os.path.join(base_dir, net_cfg["net_file"])
    sumocfg_file = os.path.join(base_dir, net_cfg["sumocfg_file"])

    # Define Node XML
    # Coordinates:
    # A (0, lane_len)    B (lane_len, lane_len)
    # C (0, 0)           D (lane_len, 0)
    nodes_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    <!-- Signalized Intersections (2x2 Grid) -->
    <node id="A" x="0.0" y="{lane_len}" type="traffic_light"/>
    <node id="B" x="{lane_len}" y="{lane_len}" type="traffic_light"/>
    <node id="C" x="0.0" y="0.0" type="traffic_light"/>
    <node id="D" x="{lane_len}" y="0.0" type="traffic_light"/>

    <!-- Boundary Portals -->
    <node id="north_A" x="0.0" y="{lane_len + bound_len}" type="priority"/>
    <node id="north_B" x="{lane_len}" y="{lane_len + bound_len}" type="priority"/>
    <node id="south_C" x="0.0" y="{-bound_len}" type="priority"/>
    <node id="south_D" x="{lane_len}" y="{-bound_len}" type="priority"/>
    <node id="west_A" x="{-bound_len}" y="{lane_len}" type="priority"/>
    <node id="west_C" x="{-bound_len}" y="0.0" type="priority"/>
    <node id="east_B" x="{lane_len + bound_len}" y="{lane_len}" type="priority"/>
    <node id="east_D" x="{lane_len + bound_len}" y="0.0" type="priority"/>
</nodes>
"""

    # Define Edges XML
    edges_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    <!-- Internal bidirectional edges -->
    <edge id="A_to_B" from="A" to="B" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="B_to_A" from="B" to="A" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="A_to_C" from="A" to="C" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="C_to_A" from="C" to="A" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="C_to_D" from="C" to="D" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="D_to_C" from="D" to="C" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="B_to_D" from="B" to="D" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="D_to_B" from="D" to="B" numLanes="{num_lanes}" speed="{speed_lim}"/>

    <!-- Boundary edges for Intersection A -->
    <edge id="north_to_A" from="north_A" to="A" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="A_to_north" from="A" to="north_A" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="west_to_A" from="west_A" to="A" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="A_to_west" from="A" to="west_A" numLanes="{num_lanes}" speed="{speed_lim}"/>

    <!-- Boundary edges for Intersection B -->
    <edge id="north_to_B" from="north_B" to="B" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="B_to_north" from="B" to="north_B" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="east_to_B" from="east_B" to="B" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="B_to_east" from="B" to="east_B" numLanes="{num_lanes}" speed="{speed_lim}"/>

    <!-- Boundary edges for Intersection C -->
    <edge id="west_to_C" from="west_C" to="C" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="C_to_west" from="C" to="west_C" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="south_to_C" from="south_C" to="C" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="C_to_south" from="C" to="south_C" numLanes="{num_lanes}" speed="{speed_lim}"/>

    <!-- Boundary edges for Intersection D -->
    <edge id="south_to_D" from="south_D" to="D" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="D_to_south" from="D" to="south_D" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="east_to_D" from="east_D" to="D" numLanes="{num_lanes}" speed="{speed_lim}"/>
    <edge id="D_to_east" from="D" to="east_D" numLanes="{num_lanes}" speed="{speed_lim}"/>
</edges>
"""

    with open(nod_file, "w", encoding="utf-8") as f:
        f.write(nodes_xml)

    with open(edg_file, "w", encoding="utf-8") as f:
        f.write(edges_xml)

    # Compile network using netconvert
    netconvert_bin = get_sumo_binary("netconvert")
    cmd = [
        netconvert_bin,
        f"--node-files={nod_file}",
        f"--edge-files={edg_file}",
        f"--output-file={net_file}",
        "--no-warnings",
        "--junctions.join", "false"
    ]
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"netconvert error:\n{result.stderr}")
        sys.exit(result.returncode)

    print(f"Successfully generated network at: {net_file}")

    # Generate simulation.sumocfg
    sumocfg_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="2x2_grid.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <step-length value="1"/>
    </time>
    <processing>
        <waiting-time-memory value="{config['simulation']['waiting_time_memory']}"/>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>
"""
    with open(sumocfg_file, "w", encoding="utf-8") as f:
        f.write(sumocfg_xml)
    print(f"Successfully generated sumocfg at: {sumocfg_file}")


if __name__ == "__main__":
    build_network()
