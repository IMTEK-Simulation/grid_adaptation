"""
Example of grid adaptation using Zecević's method for a circular inclusion.

Implements the conformal grid approach for FFT-based micromechanical models as described in:

    Zecević, M., Lebensohn, R. A., & Capolungo, L.
    "Achieving geometric accuracy in FFT-based micromechanical models using conformal grids."
    Los Alamos National Laboratory, Los Alamos, NM, USA.

"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mpi4py import MPI
import numpy as np
import time
from muGrid import Solvers
from muFFTTO import domain
from muFFTTO.visualization_utils import plot_field_on_grid
from muFFTTO.grid_adaptation_methods_Zecevic import adapt_grid_to_circle

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type =  'bilinear_rectangle'
domain_size = (1, 1)
number_of_pixels = (32,32)
center = (0.5,0.5)
radius = 0.3

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

ref_grid_coords_ixyz = discretization.get_nodal_points_coordinates().s[:, 0, ...]
coords_of_displaced_nodes,phase_indicator_array = adapt_grid_to_circle(ref_grid_coords_ixyz,center,radius,b=0,kmin=0)

phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = phase_indicator_array
matrix_mask = phase_indicator_array > 0
inc_mask = phase_indicator_array == 0

material_data_field_C_0.s[..., matrix_mask] = mat_contrast_2 * material_data_field_C_0.s[..., matrix_mask]
material_data_field_C_0.s[..., inc_mask] = mat_contrast * material_data_field_C_0.s[..., inc_mask]

def_grid_coords_inxyz = discretization.get_displacement_sized_field(name='deformed_nodal_points_coordinates_inxyz')
grid_nodes_displacement_inxyz = discretization.get_displacement_sized_field(name='grid_nodes_displacement_inxyz')
grid_nodes_displacement_inxyz.s.fill(0)
grid_nodes_displacement_inxyz.s[:, 0, ...] = coords_of_displaced_nodes - ref_grid_coords_ixyz
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

F_ijqxy = discretization.get_displacement_gradient_sized_field(name='Grid_Deformation_gradient_F_ijqxy')
discretization.fft.communicate_ghosts(grid_nodes_displacement_inxyz)
discretization.apply_gradient_operator_mugrid(grid_nodes_displacement_inxyz, F_ijqxy)
F_ijqxy.s[...] += np.eye(2)[:, :, None, None, None]

det_F = discretization.get_quad_field_scalar(name='determinant_F')
det_F.s[0,0,...] = np.linalg.det(F_ijqxy.s.transpose(2, 3, 4, 0, 1))

inv_F = discretization.get_displacement_gradient_sized_field(name='inverse_of_F')
inv_F.s[...] = np.linalg.pinv(F_ijqxy.s.transpose(2, 3, 4, 0, 1)).transpose(3, 4, 0, 1, 2)

plot_field_on_grid(coordinates_for_plot=x_plot, field_to_plot=det_F.s[0,0,0], name='det(F)')

def K_fun(x, Ax):
    discretization.apply_system_matrix_mugrid_deformed_grid(material_data_field=material_data_field_C_0,input_field_inxyz=x, output_field_inxyz=Ax, det_of_deformation_gradient=det_F,inv_of_deformation_gradient=inv_F)
    discretization.fft.communicate_ghosts(Ax)

preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=conductivity_C_1)
def M_fun(x,Px):
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,input_nodal_field_fnxyz=x,output_nodal_field_fnxyz=Px)

solution_field = discretization.get_unknown_size_field(name='solution')
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')

dim = discretization.domain_dimension
homogenized_A_ij = np.zeros(np.array(2*[dim,]))

for i in range(dim):
    # set macro gradient
    macro_gradient = np.zeros([dim])
    macro_gradient[i] = 1

    macro_gradient_field.sg.fill(0)
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,macro_gradient_field_ijqxyz=macro_gradient_field)

    discretization.fft.communicate_ghosts(field=macro_gradient_field)

    rhs_field.sg.fill(0)
    discretization.get_rhs_mugrid_deformed_grid(material_data_field_ijklqxyz=material_data_field_C_0,macro_gradient_field_ijqxyz=macro_gradient_field,rhs_inxyz=rhs_field,det_of_deformation_gradient=det_F,inv_of_deformation_gradient=inv_F)

    def callback(iteration, fields):
        """
        Callback function to print the current solution, residual, and search direction after every iteration of solver.
        """
        norm_of_rr = fields['rr']
        if discretization.communicator.rank == 0:
            print(f"{iteration:5} norm of residual = {norm_of_rr:5}")
    Solvers.conjugate_gradients(comm=discretization.communicator,fc=discretization.field_collection,hessp=K_fun, b=rhs_field,x=solution_field,prec=M_fun,rtol=1e-6,maxiter=2000,callback=callback)

    if discretization.communicator.size == 1:
        plot_field_on_grid(coordinates_for_plot=x_plot,field_to_plot=solution_field.s[0,0],name=f'Solution field - macro gradient {macro_gradient} ')
    discretization.fft.communicate_ghosts(field=solution_field)

    sum_sol = discretization.mpi_reduction.sum(solution_field.s, axis=tuple(range(-3,0)))
    print('rank' f'{MPI.COMM_WORLD.rank:6} sum_sol =' f'{sum_sol}')

    homogenized_A_ij[i,:] = discretization.get_homogenized_stress_mugrid_deformed_grid(material_data_field_ijklqxyz=material_data_field_C_0,temperature_field_inxyz=solution_field,macro_gradient_field_ijqxyz=macro_gradient_field,det_of_deformation_gradient=det_F,inv_of_deformation_gradient=inv_F)

    print("homogenized conductivity tangent = \n" + np.array2string(homogenized_A_ij, formatter={'float_kind': lambda x: f"{x:0.8f}"}))

end_time = time.time()
elapsed_time = end_time - start_time

# print summary
if discretization.communicator.rank == 0:
    print("Elapsed time: ", elapsed_time, 'seconds')
    print("Elapsed time: ", elapsed_time/60, 'minutes')
    # J_eff = mat_contrast_2*np.sqrt((mat_contrast_2+3*mat_contrast)/(3*mat_contrast_2+mat_contrast))
    # print(f'Analytical solution conductivity - A^eff_11 : {J_eff:0.8f}')
    print(f'Numerical solution conductivity - A^eff_11 : {homogenized_A_ij[0,0]:0.8f}')

det_values = det_F.s[0, 0, ...]
print("det(F) min =", np.min(det_values))
print("det(F) max =", np.max(det_values))
print("number of det(F) <= 0 =", np.count_nonzero(det_values <= 0))
print("number of det(F) < 0 =", np.count_nonzero(det_values < 0))
print("number of det(F) == 0 =", np.count_nonzero(det_values == 0))