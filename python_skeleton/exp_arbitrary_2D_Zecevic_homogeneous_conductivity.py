import time
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

from muFFTTO.otsu import otsu_edgeDetection_and_phaseIndicator
from muFFTTO import domain
from muFFTTO.visualization_utils import plot_field_on_grid
from muFFTTO.grid_adaptation_methods_Zecevic_arbitrary import adapt_grid_to_arbitrary_shape

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = (
    BASE_DIR
    / "Grain Boundaries Data"
    / "Green_Jacobi_eta_0.01_w_10.0_p_0.0_final.npy"
)

# problem configuration
problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type =  'bilinear_rectangle'
domain_size = (1, 1)
number_of_pixels = (32,32)
# spring parameters
iters = 400
omega = 0.35
b = 0
k0 = 0.25
kmin = 0

# Load data
data = np.load(INPUT_FILE).astype(np.float32)

# otsu edge detection for fine-grid data
start_time = time.time()
fine_edge_results = otsu_edgeDetection_and_phaseIndicator(data)
fine_edge_mask = fine_edge_results["edge_mask"]
fine_phase_mask_binary = fine_edge_results["phase_mask_binary"]
fine_phase_mask_label = fine_edge_results["phase_mask_label"]
time_otsu_edge_detection = time.time() - start_time

# build coarse grid
my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)
discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
ref_grid_coords_ixyz = discretization.fft.coords
# Obtain coarse-grid phase field geom & def_grid_coords_inxyz
coarse_phase_field_geom, coarse_interface_cell, coarse_def_grid_coords_inxyz, coarse_def_spring_grid_coords_inxyz = adapt_grid_to_arbitrary_shape(fine_edge_results,ref_grid_coords_ixyz,domain_size,number_of_pixels,b=b)

# ============================================
# PLOT
# ============================================
phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = coarse_phase_field_geom

def_grid_coords_inxyz = discretization.get_displacement_sized_field(name='deformed_nodal_points_coordinates_inxyz')
grid_nodes_displacement_inxyz = discretization.get_displacement_sized_field(name='grid_nodes_displacement_inxyz')
grid_nodes_displacement_inxyz.s.fill(0)
grid_nodes_displacement_inxyz.s[:, 0, ...] = coarse_def_spring_grid_coords_inxyz - ref_grid_coords_ixyz
def_grid_coords_inxyz.s[:, 0, ...] = ref_grid_coords_ixyz[...] + grid_nodes_displacement_inxyz.s[:, 0, ...]

# Create coords for plot: top and right side are copies from left and bottom, the displacement should also be added on
# x_plot = discretization.get_nodal_points_coordinates_with_periodic_nodes()
# x_plot[..., :-1, :-1] += grid_nodes_displacement_inxyz.s[...]
# x_plot = np.squeeze(x_plot, axis=1)

u = grid_nodes_displacement_inxyz.s[:, 0, ...]
# u shape: (2, nx, ny)
u_periodic = np.zeros((2, u.shape[1] + 1, u.shape[2] + 1), dtype=u.dtype)
# Interior stored nodes
u_periodic[:, :-1, :-1] = u
# Periodic copies of the left and bottom boundaries
u_periodic[:, -1, :-1] = u[:, 0, :]
u_periodic[:, :-1, -1] = u[:, :, 0]
# Corner periodic copy
u_periodic[:, -1, -1] = u[:, 0, 0]
x_plot = discretization.get_nodal_points_coordinates_with_periodic_nodes()
x_plot = np.squeeze(x_plot, axis=1)
x_plot += u_periodic
plot_field_on_grid(coordinates_for_plot=x_plot, field_to_plot=phase_field.s[0, 0], name='Material')


print("end")