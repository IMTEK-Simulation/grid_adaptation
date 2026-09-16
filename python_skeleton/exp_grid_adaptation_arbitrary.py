import time
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

from muFFTTO.otsu import otsu_edgeDetection_and_phaseIndicator
from muFFTTO.grid_adaptation_arbitrary import (
    build_regular_coarse_grid,
    make_plot_coords,
    periodic_manhattan_distance_to_mask,
    stiffness_from_distance,
    spring_relax_projected_grid,
    refine_interface_nodes_by_local_square)

simulation_dir = Path(__file__).resolve().parents[1]
mufftto_repo_dir = simulation_dir / "muFFTTO"
sys.path.insert(0, str(mufftto_repo_dir))
from muFFTTO import domain  # noqa: F401

# ============================================================
# Main script
# ============================================================

path = r'C:\Users\Test\Desktop\JiaLing\HiWi\Simulation\Grain Boundaries Data\Green_Jacobi_eta_0.01_w_10.0_p_0.0_final.npy'
coarse_Nx = 16
coarse_Ny = 16
Lx, Ly = 1.0, 1.0
relax_iters = 400
relax_omega = 0.35
relax_b = 0.5

# ============================================================
# Load data and detect edge / phase
# ============================================================

data = np.load(path).astype(np.float32)
if data.ndim != 2:
    raise ValueError(f"Expected a 2D input image, but got shape {data.shape}.")
print('Loaded data successfully')

start_time = time.time()
results = otsu_edgeDetection_and_phaseIndicator(data)
edge_mask = results["edge_mask"]
phase_mask_binary = results["phase_mask_binary"]
phase_mask_label = results["phase_mask_label"]
elapsed_time = time.time() - start_time

print('elapsed_time:', f"{elapsed_time:.4f}", 's', flush=True)
print('data shape:', data.shape)
print('edge_mask shape:', edge_mask.shape)
print('phase_mask_binary shape:', phase_mask_binary.shape)
print('phase_mask_label shape:', phase_mask_label.shape)
if 'number_of_phase_regions' in results:
    print('number_of_phase_regions:', results['number_of_phase_regions'])

# ============================================================
# Fine edge points and [N+1]x[N+1] fine-edge plot array
# ============================================================

ny, nx = data.shape
edge_binary = (edge_mask > 0).astype(np.uint8)
phase_binary = (phase_mask_binary > 0).astype(np.uint8)

edge_plot = np.zeros((ny + 1, nx + 1), dtype=edge_mask.dtype)
edge_plot[:-1, :-1] = edge_mask
edge_plot[:-1, -1] = edge_mask[:, 0]
edge_plot[-1, :-1] = edge_mask[0, :]
edge_plot[-1, -1] = edge_mask[0, 0]

x_fine = np.linspace(0, Lx, nx + 1)
y_fine = np.linspace(0, Ly, ny + 1)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

edge_idx_y, edge_idx_x = np.nonzero(edge_binary > 0)
edge_pts_x = (edge_idx_x + 0.5) / nx * Lx
edge_pts_y = (edge_idx_y + 0.5) / ny * Ly
fine_edge_points = np.column_stack((edge_pts_x, edge_pts_y))
if fine_edge_points.shape[0] == 0:
    raise ValueError('No fine edge points found in edge_mask.')

# ============================================================
# Build regular 32x32 coarse grid
# ============================================================

P0_coarse, Xn, Yn = build_regular_coarse_grid(coarse_Nx, coarse_Ny, Lx, Ly)
x_lines = np.linspace(0, Lx, coarse_Nx + 1)
y_lines = np.linspace(0, Ly, coarse_Ny + 1)
dx_coarse = Lx / coarse_Nx
dy_coarse = Ly / coarse_Ny

# ============================================================
# Detect coarse interface cells from fine edge
# ============================================================

coarse_interface_cell = np.zeros((coarse_Ny, coarse_Nx), dtype=bool)
coarse_phase_label = np.zeros((coarse_Ny, coarse_Nx), dtype=np.int8)

x_idx = np.linspace(0, nx, coarse_Nx + 1, dtype=int)
y_idx = np.linspace(0, ny, coarse_Ny + 1, dtype=int)

for j in range(coarse_Ny):
    for i in range(coarse_Nx):
        x0, x1 = x_idx[i], x_idx[i + 1]
        y0, y1 = y_idx[j], y_idx[j + 1]
        edge_patch = edge_binary[y0:y1, x0:x1]
        phase_patch = phase_binary[y0:y1, x0:x1]

        if np.any(edge_patch):
            coarse_interface_cell[j, i] = True
            coarse_phase_label[j, i] = 2
        else:
            mean_val = phase_patch.mean() if phase_patch.size > 0 else 0.0
            coarse_phase_label[j, i] = 1 if mean_val >= 0.5 else 0

# ============================================================
# Detect coarse interface nodes from surrounding interface cells
# ============================================================

coarse_interface_node = np.zeros((coarse_Ny, coarse_Nx), dtype=bool)
for j in range(coarse_Ny):
    for i in range(coarse_Nx):
        surrounding_cells = [
            coarse_interface_cell[(j - 1) % coarse_Ny, (i - 1) % coarse_Nx],
            coarse_interface_cell[(j - 1) % coarse_Ny, i % coarse_Nx],
            coarse_interface_cell[j % coarse_Ny, (i - 1) % coarse_Nx],
            coarse_interface_cell[j % coarse_Ny, i % coarse_Nx],
        ]
        if np.any(surrounding_cells):
            coarse_interface_node[j, i] = True

# ============================================================
# Refine interface nodes: remove nodes farther than 1 fine pixel
# ============================================================
coarse_interface_node_refined, removed_count = refine_interface_nodes_by_local_square(
    coarse_interface_node=coarse_interface_node,
    Xn=Xn,
    Yn=Yn,
    fine_edge_points=fine_edge_points,
    dx_coarse=dx_coarse,
    dy_coarse=dy_coarse,
)

removed_nodes_mask = coarse_interface_node & (~coarse_interface_node_refined)

print('original interface coarse nodes :', np.count_nonzero(coarse_interface_node))
print('refined interface coarse nodes  :', np.count_nonzero(coarse_interface_node_refined))
print('removed nodes                   :', removed_count)
print('local square size               :', f'{dx_coarse:.6f} x {dy_coarse:.6f}')
# ============================================================
# Project refined interface nodes to nearest fine edge point
# ============================================================

interface_node_indices = np.argwhere(coarse_interface_node_refined)
coarse_node_points = np.column_stack((Xn[coarse_interface_node_refined], Yn[coarse_interface_node_refined]))
projected_points = np.zeros_like(coarse_node_points)
projection_distance = np.zeros(coarse_node_points.shape[0], dtype=float)

for k, p in enumerate(coarse_node_points):
    diff = fine_edge_points - p[None, :]
    dist2 = np.sum(diff ** 2, axis=1)
    idx_min = np.argmin(dist2)
    projected_points[k] = fine_edge_points[idx_min]
    projection_distance[k] = np.sqrt(dist2[idx_min])

print('number of projected interface nodes:', projected_points.shape[0])
if projection_distance.size:
    print('projection distance stats:')
    print('  min  =', projection_distance.min())
    print('  max  =', projection_distance.max())
    print('  mean =', projection_distance.mean())

# ============================================================
# Build initial coarse grid and impose projected nodes
# True stored grid: NxN
# Plotting-only grid: [N+1]x[N+1] after copy boundary extension
# Left and bottom boundaries are fixed
# ============================================================

P_init = P0_coarse.copy()
for k, (j, i) in enumerate(interface_node_indices):
    #is_fixed_boundary = (j == 0) or (i == 0)
    #if not is_fixed_boundary:
    P_init[0, j, i] = projected_points[k, 0]
    P_init[1, j, i] = projected_points[k, 1]

fixed_mask = coarse_interface_node_refined.copy()
#fixed_mask[0, :] = True
#fixed_mask[:, 0] = True

# ============================================================
# Relaxation fields
# ============================================================

dist_to_interface = periodic_manhattan_distance_to_mask(fixed_mask)
k_node = stiffness_from_distance(dist_to_interface, k0=1.0, b=relax_b, a=1.0, kmin=0.05)

P_relaxed = spring_relax_projected_grid(
    P_init=P_init,
    fixed_mask=fixed_mask,
    k_node=k_node,
    iters=relax_iters,
    omega=relax_omega,
)
U_relaxed = P_relaxed - P0_coarse
P_relaxed_plot = make_plot_coords(P_relaxed, Lx=Lx, Ly=Ly)

print('P_relaxed shape (solver use):', P_relaxed.shape)
print('P_relaxed_plot shape (plot only):', P_relaxed_plot.shape)
print('ux min/max:', U_relaxed[0].min(), U_relaxed[0].max())
print('uy min/max:', U_relaxed[1].min(), U_relaxed[1].max())

# ============================================================
# Plot 1: original coarse nodes vs refined interface nodes
# ============================================================

plt.figure(figsize=(10, 10))
plt.contour(X_fine, Y_fine, edge_plot.astype(float), levels=[0.5], colors=['#00E5FF'], linewidths=1.0)
for xv in x_lines:
    plt.plot([xv, xv], [0, Ly], color='black', linewidth=0.4, alpha=0.35)
for yv in y_lines:
    plt.plot([0, Lx], [yv, yv], color='black', linewidth=0.4, alpha=0.35)
plt.scatter(Xn[coarse_interface_node], Yn[coarse_interface_node], s=16, c='lightgray', marker='o', label='original interface nodes')
plt.scatter(Xn[coarse_interface_node_refined], Yn[coarse_interface_node_refined], s=16, c='magenta', marker='o', label='refined interface nodes')
plt.scatter(Xn[removed_nodes_mask], Yn[removed_nodes_mask], s=22, c='black', marker='x', label='removed nodes')
plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interface coarse nodes before and after 1-pixel refinement')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# ============================================================
# Plot 2: projection of refined interface nodes
# ============================================================

plt.figure(figsize=(10, 10))
plt.contour(X_fine, Y_fine, edge_plot.astype(float), levels=[0.5], colors=['#00E5FF'], linewidths=1.0)
for xv in x_lines:
    plt.plot([xv, xv], [0, Ly], color='black', linewidth=0.4, alpha=0.35)
for yv in y_lines:
    plt.plot([0, Lx], [yv, yv], color='black', linewidth=0.4, alpha=0.35)
if coarse_node_points.size:
    plt.scatter(coarse_node_points[:, 0], coarse_node_points[:, 1], s=18, c='magenta', marker='o', label='refined interface nodes', zorder=5)
    plt.scatter(projected_points[:, 0], projected_points[:, 1], s=14, c='limegreen', marker='x', label='projected points on fine edge', zorder=6)
    for p0, p1 in zip(coarse_node_points, projected_points):
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], color='orange', linewidth=0.5, alpha=0.7)
plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Projection of refined interface nodes to nearest fine edge points')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# ============================================================
# Plot 3: relaxed NxN stored grid shown as [N+1]x[N+1] plotting grid
# ============================================================

plt.figure(figsize=(10, 10))
plt.contour(X_fine, Y_fine, edge_plot.astype(float), levels=[0.5], colors=['#00E5FF'], linewidths=1.0)
for i in range(P_relaxed_plot.shape[2]):
    plt.plot(P_relaxed_plot[0, :, i], P_relaxed_plot[1, :, i], color='black', linewidth=0.8, alpha=0.85)
for j in range(P_relaxed_plot.shape[1]):
    plt.plot(P_relaxed_plot[0, j, :], P_relaxed_plot[1, j, :], color='black', linewidth=0.8, alpha=0.85)
plt.scatter(P_relaxed[0, fixed_mask], P_relaxed[1, fixed_mask], s=18, c='magenta', marker='o', label='fixed nodes', zorder=6)
plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Relaxed coarse grid (NxN stored, [N+1]x[N+1] plot)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# ============================================================
# Plot 4: displacement magnitude on stored NxN grid
# ============================================================

disp_mag = np.sqrt(U_relaxed[0] ** 2 + U_relaxed[1] ** 2)
plt.figure(figsize=(8, 7))
plt.imshow(disp_mag, origin='lower', cmap='viridis', extent=[0, Lx, 0, Ly], aspect='equal')
plt.colorbar(label='|u|')
plt.scatter(P0_coarse[0, fixed_mask], P0_coarse[1, fixed_mask], s=12, c='white', marker='o')
plt.title('Coarse-node displacement magnitude (32x32 stored grid)')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()

# ============================================================
# Ready-to-use outputs for later connection to muFFTTO
# ============================================================

coords_of_displaced_nodes = P_relaxed.copy()
coords_for_plot_33x33 = P_relaxed_plot.copy()

print('\nIMPORTANT:')
print('coords_of_displaced_nodes shape =', coords_of_displaced_nodes.shape, '-> pass this to muFFTTO')
print('coords_for_plot_33x33 shape    =', coords_for_plot_33x33.shape, '-> plotting only')
