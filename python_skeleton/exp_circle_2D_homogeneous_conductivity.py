# !!! free the boundary!!!
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mpi4py import MPI
import numpy as np
import time
import matplotlib.pyplot as plt
from muGrid import Solvers

from muFFTTO import domain
from muFFTTO import microstructure_library
from muFFTTO.visualization_utils import plot_field_on_grid

from muFFTTO.grid_adaptation_methods import adapt_grid_to_circle, adapt_grid_to_circle_EXAMPLE_

# Copy of an example of how to usu muFFTTO to solve the homogenization problem for 2D heat conductivity problem
# using deformed grid with Jia's function for deformation

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
geometry_ID = 'square_inclusion'

domain_size = (1, 1)
number_of_pixels = (32, 32)

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()

# create material data field
mat_contrast = 1
mat_contrast_2 = 1e2
conductivity_C_1 = np.array([[1., 0], [0, 1.0]])

material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='conductivity_tensor')

# populate the field with C_1 material
material_data_field_C_0.s[...] = conductivity_C_1[:, :, np.newaxis, np.newaxis, np.newaxis]

# material distribution
# TODO[Jia]: Here you have to  place your function
result = adapt_grid_to_circle_EXAMPLE_(
    nb_grid_points=number_of_pixels,
    domain_size=domain_size, center=(0.5, 0.5), radius=0.2,
    reference_grid_points_coords=discretization.get_nodal_points_coordinates().s[:, 0, ...],
    iters=80, omega=0.8
)

coords_of_displaced_nodes = result["coords_of_displaced_nodes"]
phase_indicator_array = result["inside"]

phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = phase_indicator_array
matrix_mask = phase_indicator_array > 0
inc_mask = phase_indicator_array == 0

# apply material distribution
material_data_field_C_0.s[..., matrix_mask] = mat_contrast_2 * material_data_field_C_0.s[..., matrix_mask]
material_data_field_C_0.s[..., inc_mask] = mat_contrast * material_data_field_C_0.s[..., inc_mask]
# --------------------------------------------------------------------------------------------------------------------- #
# Reference coordinates
ref_grid_coords_ixyz = discretization.get_nodal_points_coordinates().s[:, 0, ...]

# Deformed coordinates
def_grid_coords_inxyz = discretization.get_displacement_sized_field(name='deformed_nodal_points_coordinates_inxyz')

# grid_nodes_displacement
grid_nodes_displacement_inxyz = discretization.get_displacement_sized_field(name='grid_nodes_displacement_inxyz')

grid_nodes_displacement_inxyz.s.fill(0)
grid_nodes_displacement_inxyz.s[:, 0, ...] = coords_of_displaced_nodes - ref_grid_coords_ixyz

# fill in the  deformation with analytical
def_grid_coords_inxyz.s[:, 0, ...] = ref_grid_coords_ixyz[...] + grid_nodes_displacement_inxyz.s[:, 0, ...]

# Deformed coords with periodic extension for plotting
x_plot = discretization.get_nodal_points_coordinates_with_periodic_nodes()
# add deformation
x_plot[..., :-1, :-1] += grid_nodes_displacement_inxyz.s[...]
x_plot = np.squeeze(x_plot, axis=1)  # removes axis for more nodal points
# Visualize grid and material
plot_field_on_grid(coordinates_for_plot=x_plot, field_to_plot=phase_field.s[0, 0], name='Material')

# Deformation gradient F = I + grad(u)
F_ijqxy = discretization.get_displacement_gradient_sized_field(name='Grid_Deformation_gradient_F_ijqxy')
discretization.fft.communicate_ghosts(grid_nodes_displacement_inxyz)
discretization.apply_gradient_operator_mugrid(grid_nodes_displacement_inxyz, F_ijqxy)
F_ijqxy.s[...] += np.eye(2)[:, :, None, None, None]
# determinant and inverse of the deformation gradient
det_F = np.linalg.det(F_ijqxy.s.transpose(2, 3, 4, 0, 1))
inv_F = np.linalg.pinv(F_ijqxy.s.transpose(2, 3, 4, 0, 1)).transpose(3, 4, 0, 1, 2)

# plot def_F in grid
plot_field_on_grid(coordinates_for_plot=x_plot, field_to_plot=det_F[0], name='det(F)')


def K_fun(x, Ax):
    """
    Matrix-free application of the Hessian matrix. For deformed grids
    """

    discretization.apply_system_matrix_mugrid_deformed_grid(material_data_field=material_data_field_C_0,
                                                            input_field_inxyz=x,
                                                            output_field_inxyz=Ax,
                                                            det_of_deformation_gradient=det_F,
                                                            inv_of_deformation_gradient=inv_F)
    discretization.fft.communicate_ghosts(Ax)


preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=conductivity_C_1)


def M_fun(x, Px):
    """
    Function to compute the product of the Preconditioner matrix with a vector.
    The Preconditioner is represented by the convolution operator.
    """
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                               input_nodal_field_fnxyz=x,
                                               output_nodal_field_fnxyz=Px)


solution_field = discretization.get_unknown_size_field(name='solution')
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')

dim = discretization.domain_dimension
homogenized_A_ij = np.zeros(np.array(2 * [dim, ]))

for i in range(dim):
    # set macroscopic gradient
    macro_gradient = np.zeros([dim])
    macro_gradient[i] = 1

    macro_gradient_field.sg.fill(0)
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                   macro_gradient_field_ijqxyz=macro_gradient_field)

    # Macro gradient in reference domain
    macro_gradient_field.s[...] = np.einsum('ij...,jk...->ik...', macro_gradient_field.s[...], inv_F)
    discretization.fft.communicate_ghosts(field=macro_gradient_field)

    # Solve equilibrium
    rhs_field.sg.fill(0)
    discretization.get_rhs_mugrid_deformed_grid(material_data_field_ijklqxyz=material_data_field_C_0,
                                                macro_gradient_field_ijqxyz=macro_gradient_field,
                                                rhs_inxyz=rhs_field,
                                                det_of_deformation_gradient=det_F,
                                                inv_of_deformation_gradient=inv_F)


    def callback(iteration, fields):
        """
        Callback function to print the current solution, residual, and search direction.
        """
        norm_of_rr = fields['rr']
        if discretization.communicator.rank == 0:
            print(f"{iteration:5} norm of residual = {norm_of_rr:.5}")


    Solvers.conjugate_gradients(
        comm=discretization.communicator,
        fc=discretization.field_collection,
        hessp=K_fun,  # linear operator
        b=rhs_field,  # right-hand side
        x=solution_field,
        prec=M_fun,
        rtol=1e-6,
        maxiter=2000,
        callback=callback)

    if discretization.communicator.size == 1:
        # Plot the first component of the solution field
        plot_field_on_grid(coordinates_for_plot=x_plot,
                           field_to_plot=solution_field.s[0, 0],
                           name=f'Solution field - macro gradient {macro_gradient} ')
    discretization.fft.communicate_ghosts(field=solution_field)

    sum_sol = discretization.mpi_reduction.sum(solution_field.s,
                                               axis=tuple(range(-3, 0)))
    print('rank' f'{MPI.COMM_WORLD.rank:6} sum_sol =' f'{sum_sol}')

    homogenized_A_ij[i, :] = discretization.get_homogenized_stress_mugrid_deformed_grid(
        material_data_field_ijklqxyz=material_data_field_C_0,
        temperature_field_inxyz=solution_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        det_of_deformation_gradient=det_F,
        inv_of_deformation_gradient=inv_F)

    # ----------------------------------------------------------------------
    print(
        "homogenized conductivity tangent =\n" +
        np.array2string(homogenized_A_ij, formatter={'float_kind': lambda x: f"{x:0.8f}"})
    )

end_time = time.time()
elapsed_time = end_time - start_time
if discretization.communicator.rank == 0:
    print("Elapsed time: ", elapsed_time, 'seconds')
    print("Elapsed time: ", elapsed_time / 60, 'minutes')
    J_eff = mat_contrast_2 * np.sqrt((mat_contrast_2 + 3 * mat_contrast) / (3 * mat_contrast_2 + mat_contrast))
    print(f'Analytical solution conductivity - A^eff_11  : {J_eff:0.8f}')
    print(f'Numerical solution  conductivity - A^eff_11  : {homogenized_A_ij[0, 0]:0.8f}')
