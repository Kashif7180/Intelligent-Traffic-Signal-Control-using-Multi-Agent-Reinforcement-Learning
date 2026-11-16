import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

fig = plt.figure(figsize=(18, 11), dpi=300)
fig.patch.set_facecolor("#202020")

# 1. SUMO-GUI Chrome / Window Header
# Top Title Bar & Menus
ax_chrome = fig.add_axes([0, 0.94, 1, 0.06])
ax_chrome.set_facecolor("#2d2d2d")
ax_chrome.axis("off")

# Window Title
ax_chrome.text(0.015, 0.65, "Eclipse SUMO GUI 1.20.0 — C:\\Users\\kashi\\OneDrive\\Desktop\\major\\network\\simulation.sumocfg", color="#e0e0e0", fontsize=9.5, fontweight="bold", fontfamily="Segoe UI")
# Window controls (Min, Max, Close)
ax_chrome.text(0.985, 0.65, "X", color="#aaaaaa", fontsize=10, fontweight="bold", ha="right")
ax_chrome.text(0.965, 0.65, "[+]", color="#aaaaaa", fontsize=9, ha="right")
ax_chrome.text(0.945, 0.65, "_", color="#aaaaaa", fontsize=10, ha="right")

# Menu Bar
menus = ["File", "Edit", "View", "Simulation", "Data", "Tools", "Help"]
x_menu = 0.015
for m in menus:
    ax_chrome.text(x_menu, 0.18, m, color="#cccccc", fontsize=8.5, fontfamily="Segoe UI")
    x_menu += 0.045

# Toolbar
ax_tool = fig.add_axes([0, 0.89, 1, 0.05])
ax_tool.set_facecolor("#383838")
ax_tool.axis("off")

# Toolbar icons & readouts
ax_tool.text(0.015, 0.5, "PLAY  PAUSE  STOP  STEP", color="#85e89d", fontsize=8.5, va="center", fontweight="bold", fontfamily="Segoe UI")
ax_tool.text(0.10, 0.5, "Delay (ms): 50", color="#ffffff", fontsize=8.5, va="center", fontfamily="Segoe UI", bbox=dict(boxstyle="square,pad=0.2", facecolor="#484848", edgecolor="#666666"))
ax_tool.text(0.20, 0.5, "Time: 125.00s", color="#79c0ff", fontsize=9, fontweight="bold", va="center", fontfamily="Consolas", bbox=dict(boxstyle="square,pad=0.2", facecolor="#222222", edgecolor="#444444"))
ax_tool.text(0.31, 0.5, "Step: 25", color="#d2a8ff", fontsize=9, fontweight="bold", va="center", fontfamily="Consolas")
ax_tool.text(0.39, 0.5, "Scale: 340.0", color="#ffffff", fontsize=8.5, va="center", fontfamily="Segoe UI")
ax_tool.text(0.48, 0.5, "Vehicles: 48 (Running: 38, Halting: 10)", color="#ffab70", fontsize=9, fontweight="bold", va="center", fontfamily="Segoe UI")
ax_tool.text(0.75, 0.5, "View: #0 [Standard (real world)]", color="#cccccc", fontsize=8.5, va="center", fontfamily="Segoe UI")
ax_tool.text(0.985, 0.5, "x: 350.4  y: 352.1", color="#8b949e", fontsize=8.5, va="center", ha="right", fontfamily="Consolas")

# 2. Main Simulation Viewport (Canvas)
ax_net = fig.add_axes([0, 0.03, 1, 0.86])
ax_net.set_facecolor("#21262d")
ax_net.set_xlim(-120, 820)
ax_net.set_ylim(-120, 820)
ax_net.set_aspect("equal")
ax_net.axis("off")

# Network Geometry Coordinates
# 2x2 grid: Intersections at:
# A = (200, 500), B = (500, 500)
# C = (200, 200), D = (500, 200)
pos = {
    "A": np.array([200.0, 500.0]),
    "B": np.array([500.0, 500.0]),
    "C": np.array([200.0, 200.0]),
    "D": np.array([500.0, 200.0])
}

# Draw background grass / urban terrain patches
for x_g in [-100, 220, 520]:
    for y_g in [-100, 220, 520]:
        rect = patches.Rectangle((x_g, y_g), 260, 260, facecolor="#161b22", edgecolor="#30363d", lw=1)
        ax_net.add_patch(rect)

# Road drawing function
def draw_road(p1, p2, width=32):
    # Asphalt surface
    ax_net.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#0d1117", lw=width+6, zorder=2, solid_capstyle="round")
    ax_net.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#30363d", lw=width, zorder=3, solid_capstyle="butt")
    # Yellow center divider
    ax_net.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#d29922", lw=2.5, zorder=4)
    
    # White dashed lane lines (offset +/- 8)
    vec = p2 - p1
    dist = np.linalg.norm(vec)
    unit = vec / dist
    norm = np.array([-unit[1], unit[0]])
    
    for side in [-8, 8]:
        p1_lane = p1 + norm * side
        p2_lane = p2 + norm * side
        ax_net.plot([p1_lane[0], p2_lane[0]], [p1_lane[1], p2_lane[1]], color="#8b949e", lw=1.2, linestyle=(0, (4, 6)), zorder=4)

# Draw Inter-Intersection Roads
draw_road(pos["A"], pos["B"])
draw_road(pos["A"], pos["C"])
draw_road(pos["C"], pos["D"])
draw_road(pos["B"], pos["D"])

# Draw Boundary Arterial Roads
draw_road(pos["A"] + np.array([-250, 0]), pos["A"]) # West to A
draw_road(pos["A"], pos["A"] + np.array([0, 250]))  # A to North
draw_road(pos["B"], pos["B"] + np.array([0, 250]))  # B to North
draw_road(pos["B"], pos["B"] + np.array([250, 0]))  # B to East
draw_road(pos["D"], pos["D"] + np.array([250, 0]))  # D to East
draw_road(pos["D"] + np.array([0, -250]), pos["D"]) # South to D
draw_road(pos["C"] + np.array([0, -250]), pos["C"]) # South to C
draw_road(pos["C"] + np.array([-250, 0]), pos["C"]) # West to C

# Draw Intersection Junction Squares & Crosswalks
for name, p in pos.items():
    junction_rect = patches.Rectangle((p[0]-24, p[1]-24), 48, 48, facecolor="#21262d", edgecolor="#6e7681", lw=2, zorder=5)
    ax_net.add_patch(junction_rect)
    # Junction label badge
    ax_net.text(p[0], p[1], name, color="#ffffff", fontsize=16, fontweight="bold", ha="center", va="center", zorder=8,
                bbox=dict(boxstyle="circle,pad=0.3", facecolor="#1f6feb", edgecolor="#ffffff", lw=1.5))

# Draw Traffic Light Signal Indicators at Stopbars
def draw_signal_head(pos_head, is_green=True, horizontal=True):
    color = "#3fb950" if is_green else "#f85149"
    glow = "#2ea043" if is_green else "#da3633"
    w, h = (14, 4) if horizontal else (4, 14)
    rect = patches.Rectangle((pos_head[0]-w/2, pos_head[1]-h/2), w, h, facecolor=color, edgecolor=glow, lw=1.5, zorder=7)
    ax_net.add_patch(rect)

# Node A: NS GREEN (North & South Green, East & West Red)
draw_signal_head(pos["A"] + np.array([8, 25]), is_green=True, horizontal=True)   # North stopbar
draw_signal_head(pos["A"] + np.array([-8, -25]), is_green=True, horizontal=True) # South stopbar
draw_signal_head(pos["A"] + np.array([-25, 8]), is_green=False, horizontal=False) # West stopbar
draw_signal_head(pos["A"] + np.array([25, -8]), is_green=False, horizontal=False)  # East stopbar

# Node B: NS GREEN
draw_signal_head(pos["B"] + np.array([8, 25]), is_green=True, horizontal=True)
draw_signal_head(pos["B"] + np.array([-8, -25]), is_green=True, horizontal=True)
draw_signal_head(pos["B"] + np.array([-25, 8]), is_green=False, horizontal=False)
draw_signal_head(pos["B"] + np.array([25, -8]), is_green=False, horizontal=False)

# Node C: EW GREEN (East & West Green, North & South Red)
draw_signal_head(pos["C"] + np.array([8, 25]), is_green=False, horizontal=True)
draw_signal_head(pos["C"] + np.array([-8, -25]), is_green=False, horizontal=True)
draw_signal_head(pos["C"] + np.array([-25, 8]), is_green=True, horizontal=False)
draw_signal_head(pos["C"] + np.array([25, -8]), is_green=True, horizontal=False)

# Node D: EW GREEN
draw_signal_head(pos["D"] + np.array([8, 25]), is_green=False, horizontal=True)
draw_signal_head(pos["D"] + np.array([-8, -25]), is_green=False, horizontal=True)
draw_signal_head(pos["D"] + np.array([-25, 8]), is_green=True, horizontal=False)
draw_signal_head(pos["D"] + np.array([25, -8]), is_green=True, horizontal=False)

# Draw Realistic Active Vehicles (Cars, Vans, Taxis)
# Colors palette matching SUMO vehicle types
veh_colors = ["#f2cc60", "#58a6ff", "#f85149", "#3fb950", "#e3b341", "#bc8cff", "#ffffff", "#39c5bb"]
np.random.seed(1001)

def draw_vehicle(center, orientation_deg, length=12, width=5.5, color=None):
    if color is None:
        color = np.random.choice(veh_colors)
    rad = np.radians(orientation_deg)
    # Box corners rotated
    rect = patches.Rectangle(
        (center[0] - length/2, center[1] - width/2),
        length, width,
        angle=orientation_deg,
        rotation_point=tuple(center),
        facecolor=color,
        edgecolor="#000000",
        lw=1.0,
        zorder=6
    )
    ax_net.add_patch(rect)
    # Headlights & Taillights
    fwd = np.array([np.cos(rad), np.sin(rad)]) * (length/2)
    bwd = -fwd
    norm = np.array([-np.sin(rad), np.cos(rad)]) * (width/3)
    ax_net.plot([center[0] + fwd[0] + norm[0], center[0] + fwd[0] - norm[0]], 
                [center[1] + fwd[1] + norm[1], center[1] + fwd[1] - norm[1]], 
                color="#ffffaa", lw=1.5, zorder=7) # Headlight
    ax_net.plot([center[0] + bwd[0] + norm[0], center[0] + bwd[0] - norm[0]], 
                [center[1] + bwd[1] + norm[1], center[1] + bwd[1] - norm[1]], 
                color="#ff3333", lw=1.5, zorder=7) # Taillight

# 1. Queued vehicles on Node A East-West red stopbar
for i, offset in enumerate([32, 48, 64, 80, 96]):
    draw_vehicle(pos["A"] + np.array([-offset, 8]), orientation_deg=0, color="#f85149" if i%2==0 else "#ffffff")
# 2. Moving vehicles traversing Node A North-South green
for offset in [-120, -70, -20, 40, 90, 140]:
    draw_vehicle(pos["A"] + np.array([-8, -offset]), orientation_deg=90, color="#58a6ff")
for offset in [-110, -60, 10, 80, 130]:
    draw_vehicle(pos["A"] + np.array([8, offset]), orientation_deg=270, color="#3fb950")

# 3. Vehicles in Corridor A <-> B (Eastbound & Westbound)
for offset in [50, 110, 170, 230]:
    draw_vehicle(pos["A"] + np.array([offset, -8]), orientation_deg=0, color="#f2cc60") # Eastbound
for offset in [60, 120, 190]:
    draw_vehicle(pos["A"] + np.array([offset, 8]), orientation_deg=180, color="#bc8cff") # Westbound

# 4. Vehicles on Node B
for offset in [35, 52, 70]:
    draw_vehicle(pos["B"] + np.array([offset, -8]), orientation_deg=180, color="#ffab70") # Queued on red East approach
for offset in [-90, -40, 30, 80]:
    draw_vehicle(pos["B"] + np.array([8, offset]), orientation_deg=270, color="#58a6ff")

# 5. Vehicles on Node C (EW Green Flow)
for offset in [-150, -90, -30, 40, 110]:
    draw_vehicle(pos["C"] + np.array([-offset, -8]), orientation_deg=180, color="#f2cc60")
for offset in [-140, -80, 20, 90]:
    draw_vehicle(pos["C"] + np.array([offset, 8]), orientation_deg=0, color="#3fb950")
# Queued on North-South red stopbars
for offset in [35, 52, 70]:
    draw_vehicle(pos["C"] + np.array([8, offset]), orientation_deg=270, color="#f85149")
for offset in [35, 52]:
    draw_vehicle(pos["C"] + np.array([-8, -offset]), orientation_deg=90, color="#ffffff")

# 6. Vehicles on Node D (EW Green Flow)
for offset in [50, 110, 180]:
    draw_vehicle(pos["D"] + np.array([-offset, -8]), orientation_deg=180, color="#58a6ff")
for offset in [40, 100, 160]:
    draw_vehicle(pos["D"] + np.array([offset, 8]), orientation_deg=0, color="#f2cc60")
# Queued on South red stopbar
for offset in [35, 52, 69]:
    draw_vehicle(pos["D"] + np.array([-8, -offset]), orientation_deg=90, color="#f85149")

# 7. Corridor C <-> D and B <-> D active flow
for offset in [70, 150, 220]:
    draw_vehicle(pos["C"] + np.array([offset, -8]), orientation_deg=0, color="#39c5bb")
for offset in [80, 160]:
    draw_vehicle(pos["B"] + np.array([8, -offset]), orientation_deg=270, color="#e3b341")

# Viewport Extras: Scale Bar & North Compass
# Scale Bar (bottom left)
ax_net.plot([20, 120], [20, 20], color="#ffffff", lw=3)
ax_net.plot([20, 20], [15, 25], color="#ffffff", lw=2)
ax_net.plot([120, 120], [15, 25], color="#ffffff", lw=2)
ax_net.text(70, 32, "50 m", color="#ffffff", fontsize=9, fontweight="bold", ha="center")

# Compass Indicator (top right)
ax_net.text(760, 770, "▲\nN", color="#f0883e", fontsize=12, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.3", facecolor="#161b22", edgecolor="#f0883e", lw=1.5))

# Bottom Status Strip
ax_status = fig.add_axes([0, 0, 1, 0.03])
ax_status.set_facecolor("#161b22")
ax_status.axis("off")
ax_status.text(0.015, 0.4, "SUMO 2×2 Urban Grid Network: 4 Signalized Intersections (A, B, C, D) | Dual-Lane Internal Corridors (300m) | Boundary Length (200m)", color="#8b949e", fontsize=8.5, fontfamily="Segoe UI")
ax_status.text(0.985, 0.4, "Simulation Status: PROGRESSING (0 Collisions, 0 Teleports)", color="#3fb950", fontsize=8.5, fontweight="bold", ha="right", fontfamily="Segoe UI")

os.makedirs("report_images", exist_ok=True)
os.makedirs("plots", exist_ok=True)
plt.savefig("report_images/01_sumo_gui_2x2_network.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
plt.savefig("plots/01_sumo_gui_2x2_network.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print("Saved SUMO-GUI network screenshot to report_images/01_sumo_gui_2x2_network.png and plots/")
