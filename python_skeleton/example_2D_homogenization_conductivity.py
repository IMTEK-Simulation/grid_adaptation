import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from mpi4py import MPI
import numpy as np
import time
from muGrid import Solvers

from muFFTTO import domain
from muFFTTO import microstructure_library

# Example of how to usu muFFTTO to solve the homogenization problem for 2D heat conductivity problem
# We use square inclusion geometry, for which we have analytical solution

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
geometry_ID = 'square_inclusion'

domain_size = [1, 1]
number_of_pixels = (128, 128)

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
phase_field_geom = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                       microstructure_name=geometry_ID,
                                                       coordinates=discretization.fft.coords)

phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = phase_field_geom
matrix_mask = phase_field_geom > 0
inc_mask = phase_field_geom == 0

# apply material distribution

material_data_field_C_0.s[..., matrix_mask] = mat_contrast_2 * material_data_field_C_0.s[..., matrix_mask]
material_data_field_C_0.s[..., inc_mask] = mat_contrast * material_data_field_C_0.s[..., inc_mask]

def K_fun(x, Ax):
    """
    Function to compute the product of the Hessian matrix with a vector.
    The Hessian is represented by the convolution operator.
    """

    discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
                                              input_field_inxyz=x,
                                              output_field_inxyz=Ax)
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
    discretization.fft.communicate_ghosts(field=macro_gradient_field)

    # Solve equilibrium
    rhs_field.sg.fill(0)
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                  macro_gradient_field_ijqxyz=macro_gradient_field,
                                  rhs_inxyz=rhs_field)

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
        import matplotlib.pyplot as plt
        plt.figure()
        plt.pcolormesh(discretization.fft.coords[0],
                       discretization.fft.coords[1],
                       solution_field.s[0,0])

        plt.title(f'Solution field - macro gradient {macro_gradient} ')
        plt.xlabel('x  / L')
        plt.ylabel('y  / L')
        plt.colorbar(label='Temperature / Potential')
        plt.show()
    discretization.fft.communicate_ghosts(field=solution_field)

    sum_sol = discretization.mpi_reduction.sum(solution_field.s,
                                               axis=tuple(range(-3, 0)))
    print('rank' f'{MPI.COMM_WORLD.rank:6} sum_sol =' f'{sum_sol}')

    homogenized_A_ij[i, :] = discretization.get_homogenized_stress_mugrid(
        material_data_field_ijklqxyz=material_data_field_C_0,
        displacement_field_inxyz=solution_field,
        macro_gradient_field_ijqxyz=macro_gradient_field)

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
    print(f'Numerical solution  conductivity - A^eff_11  : {homogenized_A_ij[0,0]:0.8f}')

