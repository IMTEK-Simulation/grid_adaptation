"""
Example of grid adaptation using Zecevic's method for a circular inclusion.

Implements the conformal grid approach for FFT-based micromechanical models as described in:

    Zecevic, M., Lebensohn, R. A., & Capolungo, L.
    "Achieving geometric accuracy in FFT-based micromechanical models using conformal grids."
    Los Alamos National Laboratory, Los Alamos, NM, USA.

"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from datetime import datetime
from pathlib import Path

from mpi4py import MPI
import numpy as np
import time
import matplotlib.pyplot as plt
from muGrid import Solvers
from muFFTTO import domain
from muFFTTO.visualization_utils import plot_field_on_grid
from muFFTTO.grid_adaptation_methods_Zecevic import adapt_grid_to_circle, build_periodic_plot_coordinates

# =============================================================
# USER PARAMETERS
# =============================================================
# problem configuration
problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'bilinear_rectangle'
domain_size = (1, 1)
number_of_pixels = (32, 32)

# circular inclusion (may cross the periodic boundary; radius < min(domain_size)/2)
center = (0.9, 0.9)
radius = 0.45

# spring parameters
iters = 600                           # fixed number of relaxation iterations
b = 0.0                               # stiffness decay exponent
k0 = 0.25                             # stiffness closest to the interface
kmin = 0                              # minimum stiffness
if_spring_relaxation = True           # False: use the projected grid directly
spring_method = "gauss_seidel"        # "gauss_seidel" (original) or "jacobi_displacement"
check_inversion_each_iter = False     # True: record inverted cells after every iteration
inversion_check = "quadrature"        # invalid-node filter: "quadrature" (solver det(F)), "corner" (stricter), "off" (original method)

# =============================================================
# OUTPUT
# =============================================================
OUTPUT_ROOT = Path(__file__).resolve().parent / "output_circle"

RUN_TAG = (
    f"N{number_of_pixels[0]}x{number_of_pixels[1]}"
    f"_L{domain_size[0]:g}x{domain_size[1]:g}"
    f"_c{center[0]:g}-{center[1]:g}_r{radius:g}"
    f"_it{iters}_b{b:g}_k{k0:g}_kmin{kmin:g}"
    f"_spr{int(if_spring_relaxation)}_{'gs' if spring_method == 'gauss_seidel' else 'jd'}"
    f"_ic{inversion_check}"
    f"_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
)

OUTPUT_DIR = OUTPUT_ROOT / RUN_TAG
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_field_and_save(coordinates_for_plot, field_to_plot, name, save_path):
    """Call plot_field_on_grid and save the figure right before plt.show()."""
    original_show = plt.show

    def _save_then_show(*args, **kwargs):
        plt.gcf().savefig(save_path, dpi=300, bbox_inches="tight")
        return original_show(*args, **kwargs)

    plt.show = _save_then_show
    try:
        plot_field_on_grid(coordinates_for_plot=coordinates_for_plot, field_to_plot=field_to_plot, name=name)
    finally:
        plt.show = original_show


my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)
discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()

mat_contrast = 1
mat_contrast_2 = 1e2
conductivity_C_1 = np.array([[1., 0], [0, 1.0]])
material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='conductivity_tensor')
material_data_field_C_0.s[...] = conductivity_C_1[:, :, np.newaxis, np.newaxis, np.newaxis]

# =============================================================
# GRID ADAPTATION (WORKFLOW A: CIRCULAR INCLUSION)
# =============================================================
ref_grid_coords_ixyz = np.array(discretization.get_nodal_points_coordinates().s[:, 0, ...])
adaptation = adapt_grid_to_circle(
    ref_grid_coords_ixyz=ref_grid_coords_ixyz,
    center=center,
    radius=radius,
    iters=iters,
    b=b,
    k0=k0,
    kmin=kmin,
    if_spring_relaxation=if_spring_relaxation,
    spring_method=spring_method,
    check_inversion_each_iter=check_inversion_each_iter,
    inversion_check=inversion_check,
)
coarse_phase_field_geom = adaptation["coarse_phase_field_geom"]
coarse_phase_field_binary = adaptation["coarse_phase_field_binary"]
coarse_interface_cell = adaptation["coarse_interface_cell"]
coarse_def_grid_coords_inxyz = adaptation["coarse_def_grid_coords_inxyz"]
coarse_def_spring_grid_coords_inxyz = adaptation["coarse_def_spring_grid_coords_inxyz"]

# ---------- GRID ADAPTATION DIAGNOSTICS (comment out to silence) ----------
print("===== GRID ADAPTATION DIAGNOSTICS =====")
for key, value in adaptation["diagnostics"].items():
    print(f"{key:35s}: {value}")
print("=======================================")
# --------------------------------------------------------------------------

np.save(OUTPUT_DIR / f"coarse_phase_field_geom_{RUN_TAG}.npy", coarse_phase_field_geom)
np.save(OUTPUT_DIR / f"coarse_phase_field_binary_{RUN_TAG}.npy", coarse_phase_field_binary)
np.save(OUTPUT_DIR / f"coarse_def_grid_coords_inxyz_{RUN_TAG}.npy", coarse_def_grid_coords_inxyz)
np.save(OUTPUT_DIR / f"coarse_def_spring_grid_coords_inxyz_{RUN_TAG}.npy", coarse_def_spring_grid_coords_inxyz)

phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = coarse_phase_field_geom
matrix_mask = coarse_phase_field_geom > 0
inc_mask = coarse_phase_field_geom == 0

material_data_field_C_0.s[..., matrix_mask] = mat_contrast_2 * material_data_field_C_0.s[..., matrix_mask]
material_data_field_C_0.s[..., inc_mask] = mat_contrast * material_data_field_C_0.s[..., inc_mask]

# coordinates and displacement used by the SOLVER (stored nodes only)
def_grid_coords_inxyz = discretization.get_displacement_sized_field(name='deformed_nodal_points_coordinates_inxyz')
grid_nodes_displacement_inxyz = discretization.get_displacement_sized_field(name='grid_nodes_displacement_inxyz')
grid_nodes_displacement_inxyz.s.fill(0)
grid_nodes_displacement_inxyz.s[:, 0, ...] = coarse_def_spring_grid_coords_inxyz - ref_grid_coords_ixyz
def_grid_coords_inxyz.s[:, 0, ...] = ref_grid_coords_ixyz[...] + grid_nodes_displacement_inxyz.s[:, 0, ...]

# coordinates used for PLOTTING only (with periodic end nodes, shape (2, Nx+1, Ny+1))
x_plot = build_periodic_plot_coordinates(
    def_coords=coarse_def_spring_grid_coords_inxyz,
    ref_coords=ref_grid_coords_ixyz,
    domain_size=domain_size,
)
plot_field_and_save(x_plot, phase_field.s[0, 0], 'Material',
                    OUTPUT_DIR / f"Material_{RUN_TAG}.png")

F_ijqxy = discretization.get_displacement_gradient_sized_field(name='Grid_Deformation_gradient_F_ijqxy')
discretization.fft.communicate_ghosts(grid_nodes_displacement_inxyz)
discretization.apply_gradient_operator_mugrid(grid_nodes_displacement_inxyz, F_ijqxy)
F_ijqxy.s[...] += np.eye(2)[:, :, None, None, None]

det_F = discretization.get_quad_field_scalar(name='determinant_F')
det_F.s[0, 0, ...] = np.linalg.det(F_ijqxy.s.transpose(2, 3, 4, 0, 1))

inv_F = discretization.get_displacement_gradient_sized_field(name='inverse_of_F')
inv_F.s[...] = np.linalg.pinv(F_ijqxy.s.transpose(2, 3, 4, 0, 1)).transpose(3, 4, 0, 1, 2)

plot_field_and_save(x_plot, det_F.s[0, 0, 0], 'det(F)',
                    OUTPUT_DIR / f"detF_{RUN_TAG}.png")


def K_fun(x, Ax):
    discretization.apply_system_matrix_mugrid_deformed_grid(material_data_field=material_data_field_C_0, input_field_inxyz=x, output_field_inxyz=Ax, det_of_deformation_gradient=det_F, inv_of_deformation_gradient=inv_F)
    discretization.fft.communicate_ghosts(Ax)


preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=conductivity_C_1)


def M_fun(x, Px):
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner, input_nodal_field_fnxyz=x, output_nodal_field_fnxyz=Px)


solution_field = discretization.get_unknown_size_field(name='solution')
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')

dim = discretization.domain_dimension
homogenized_A_ij = np.zeros(np.array(2 * [dim, ]))

for i in range(dim):
    # set macro gradient
    macro_gradient = np.zeros([dim])
    macro_gradient[i] = 1

    macro_gradient_field.sg.fill(0)
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient, macro_gradient_field_ijqxyz=macro_gradient_field)

    discretization.fft.communicate_ghosts(field=macro_gradient_field)

    rhs_field.sg.fill(0)
    discretization.get_rhs_mugrid_deformed_grid(material_data_field_ijklqxyz=material_data_field_C_0, macro_gradient_field_ijqxyz=macro_gradient_field, rhs_inxyz=rhs_field, det_of_deformation_gradient=det_F, inv_of_deformation_gradient=inv_F)

    def callback(iteration, fields):
        """
        Callback function to print the current solution, residual, and search direction after every iteration of solver.
        """
        norm_of_rr = fields['rr']
        if discretization.communicator.rank == 0:
            print(f"{iteration:5} norm of residual = {norm_of_rr:5}")
    Solvers.conjugate_gradients(comm=discretization.communicator, fc=discretization.field_collection, hessp=K_fun, b=rhs_field, x=solution_field, prec=M_fun, rtol=1e-6, maxiter=2000, callback=callback)

    if discretization.communicator.size == 1:
        plot_field_and_save(x_plot, solution_field.s[0, 0],
                            f'Solution field - macro gradient {macro_gradient} ',
                            OUTPUT_DIR / f"Solution_macro_gradient_{'xy'[i]}_{RUN_TAG}.png")
    discretization.fft.communicate_ghosts(field=solution_field)

    sum_sol = discretization.mpi_reduction.sum(solution_field.s, axis=tuple(range(-3, 0)))
    print('rank' f'{MPI.COMM_WORLD.rank:6} sum_sol =' f'{sum_sol}')

    homogenized_A_ij[i, :] = discretization.get_homogenized_stress_mugrid_deformed_grid(material_data_field_ijklqxyz=material_data_field_C_0, temperature_field_inxyz=solution_field, macro_gradient_field_ijqxyz=macro_gradient_field, det_of_deformation_gradient=det_F, inv_of_deformation_gradient=inv_F)

    print("homogenized conductivity tangent = \n" + np.array2string(homogenized_A_ij, formatter={'float_kind': lambda x: f"{x:0.8f}"}))

end_time = time.time()
elapsed_time = end_time - start_time

# print summary
if discretization.communicator.rank == 0:
    print("Elapsed time: ", elapsed_time, 'seconds')
    print("Elapsed time: ", elapsed_time / 60, 'minutes')
    # J_eff = mat_contrast_2*np.sqrt((mat_contrast_2+3*mat_contrast)/(3*mat_contrast_2+mat_contrast))
    # print(f'Analytical solution conductivity - A^eff_11 : {J_eff:0.8f}')
    print(f'Numerical solution conductivity - A^eff_11 : {homogenized_A_ij[0, 0]:0.8f}')

det_values = det_F.s[0, 0, ...]
print("det(F) min =", np.min(det_values))
print("det(F) max =", np.max(det_values))
print("number of det(F) <= 0 =", np.count_nonzero(det_values <= 0))
print("number of det(F) < 0 =", np.count_nonzero(det_values < 0))
print("number of det(F) == 0 =", np.count_nonzero(det_values == 0))
print("outputs saved in:", OUTPUT_DIR)
