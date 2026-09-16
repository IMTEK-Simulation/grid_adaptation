"""muFFTTO conductivity homogenization on an image-adapted grid.

The image-processing workflow must return:

    result["coords_of_displaced_nodes"]
    result["full_plot_coords_of_displaced_nodes"]
    result["coarse_phase_label"]

The coarse phase labels are assigned one conductivity tensor each.

Run with one process first:

    python mufftto_grain_boundary_multiphase_case.py

For MPI, for example:

    mpiexec -n 4 python mufftto_grain_boundary_multiphase_case.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from muGrid import Solvers

# Adjust this import if your workflow is stored in another module.
from muFFTTO.grid_adaptation_arbitrary import (
    run_grid_adaptation_workflow,
)
from muFFTTO import domain
from muFFTTO.visualization_utils import plot_field_on_grid

# ============================================================================
# User settings
# ============================================================================

INPUT_FILE = Path(
    "/home/martin/Programming/new_programming_folders/grid_adaptation/python_skeleton/uv.lock"
)

PROBLEM_TYPE = "conductivity"
DISCRETIZATION_TYPE = "finite_element"
ELEMENT_TYPE = "linear_triangles"

DOMAIN_SIZE = (1.0, 1.0)
NUMBER_OF_PIXELS = (32,32)

RELAX_ITERS = 400
RELAX_OMEGA = 0.1
RELAX_B = 0.5

SOLVER_RTOL = 1e-6
SOLVER_MAXITER = 2000

# Edit this table according to the physical meaning of your phase labels.
# Example:
#   label 0 -> matrix conductivity 1.0
#   label 1 -> region conductivity 2.0
#   label 2 -> region conductivity 3.0
#   label 3 -> region conductivity 4.0
CONDUCTIVITY_BY_LABEL = {
    0: 1.0,
    1: 2.0,
    2: 3.0,
    3: 4.0,
}


# ============================================================================
# Material field
# ============================================================================


def create_material_fields(discretization, phase_indicator_array):
    """Create phase and conductivity fields from coarse phase labels."""
    phase_indicator_array = np.asarray(
        phase_indicator_array,
        dtype=np.int32,
    )

    expected_shape = tuple(NUMBER_OF_PIXELS)
    if phase_indicator_array.shape != expected_shape:
        raise ValueError(
            "coarse_phase_label has the wrong shape: "
            f"got {phase_indicator_array.shape}, "
            f"expected {expected_shape}."
        )

    actual_labels = np.unique(phase_indicator_array)
    if np.any(actual_labels < 0):
        raise ValueError(
            f"Phase labels must be non-negative, got {actual_labels}."
        )

    missing_conductivity_labels = [
        int(label)
        for label in actual_labels
        if int(label) not in CONDUCTIVITY_BY_LABEL
    ]
    if missing_conductivity_labels:
        raise ValueError(
            "No conductivity was assigned to phase labels: "
            f"{missing_conductivity_labels}. "
            "Update CONDUCTIVITY_BY_LABEL."
        )

    material_data_field = (
        discretization.get_material_data_size_field_mugrid(
            name="conductivity_tensor",
        )
    )
    material_data_field.s.fill(0.0)

    phase_field = discretization.get_scalar_field(
        name="phase_field",
    )
    phase_field.s[0, 0, ...] = phase_indicator_array

    conductivity_field = discretization.get_scalar_field(
        name="conductivity_value_field",
    )
    conductivity_field.s.fill(0.0)

    for label in actual_labels:
        label = int(label)
        conductivity_value = float(
            CONDUCTIVITY_BY_LABEL[label]
        )

        if conductivity_value <= 0.0:
            raise ValueError(
                f"Conductivity for label {label} must be positive."
            )

        material_mask = phase_indicator_array == label
        conductivity_tensor = conductivity_value * np.eye(2)

        material_data_field.s[..., material_mask] = (
            conductivity_tensor[..., np.newaxis, np.newaxis]
        )
        conductivity_field.s[0, 0, material_mask] = (
            conductivity_value
        )

        print(
            f"label {label}: "
            f"{int(material_mask.sum())} coarse cells, "
            f"conductivity = {conductivity_value:.6g} * I"
        )

    print("phase-field shape:", phase_field.s.shape)
    print(
        "material tensor field shape:",
        material_data_field.s.shape,
    )

    return (
        material_data_field,
        phase_field,
        conductivity_field,
    )

def compute_det_F_from_plot_coords(
    coordinates_for_plot: np.ndarray,
    Lx: float,
    Ly: float,
) -> np.ndarray:
    """Compute cell-wise det(F) from deformed plotting coordinates."""

    p00 = coordinates_for_plot[:, :-1, :-1]
    p10 = coordinates_for_plot[:, :-1, 1:]
    p01 = coordinates_for_plot[:, 1:, :-1]
    p11 = coordinates_for_plot[:, 1:, 1:]

    _, Ny_plus_one, Nx_plus_one = (
        coordinates_for_plot.shape
    )

    Nx = Nx_plus_one - 1
    Ny = Ny_plus_one - 1

    dx = Lx / Nx
    dy = Ly / Ny

    tangent_x = 0.5 * (
        (p10 - p00)
        + (p11 - p01)
    )

    tangent_y = 0.5 * (
        (p01 - p00)
        + (p11 - p10)
    )

    F11 = tangent_x[0] / dx
    F21 = tangent_x[1] / dx

    F12 = tangent_y[0] / dy
    F22 = tangent_y[1] / dy

    return F11 * F22 - F12 * F21

# ============================================================================
# Main simulation
# ============================================================================


def main() -> None:
    start_time = time.time()
    communicator = MPI.COMM_WORLD

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Input image does not exist: {INPUT_FILE}"
        )

    if communicator.rank == 0:
        print("Input image:", INPUT_FILE)
        print("MPI processes:", communicator.size)

    cell = domain.PeriodicUnitCell(
        domain_size=DOMAIN_SIZE,
        problem_type=PROBLEM_TYPE,
    )

    discretization = domain.Discretization(
        cell=cell,
        nb_of_pixels_global=NUMBER_OF_PIXELS,
        discretization_type=DISCRETIZATION_TYPE,
        element_type=ELEMENT_TYPE,
    )

    # ------------------------------------------------------------------------
    # Image-based grid adaptation and coarse phase labels
    # ------------------------------------------------------------------------
    result = run_grid_adaptation_workflow(
        input_path=INPUT_FILE,
        coarse_Nx=NUMBER_OF_PIXELS[0],
        coarse_Ny=NUMBER_OF_PIXELS[1],
        Lx=DOMAIN_SIZE[0],
        Ly=DOMAIN_SIZE[1],
        relax_iters=RELAX_ITERS,
        relax_omega=RELAX_OMEGA,
        relax_b=RELAX_B,
        verbose=(communicator.rank == 0),
    )

    coords_of_displaced_nodes = result[
        "coords_of_displaced_nodes"
    ]
    phase_indicator_array = result[
        "coarse_phase_label"
    ].astype(np.int32)
    coordinates_for_plot = result[
        "full_plot_coords_of_displaced_nodes"
    ]

    det_F_from_plot = compute_det_F_from_plot_coords(
        coordinates_for_plot=coordinates_for_plot,
        Lx=DOMAIN_SIZE[0],
        Ly=DOMAIN_SIZE[1],
    )


    reference_coordinates = (
        discretization.get_nodal_points_coordinates().s[:, 0, ...]
    )

    if coords_of_displaced_nodes.shape != reference_coordinates.shape:
        raise ValueError(
            "The deformed coordinate shape does not match the "
            "muFFTTO reference coordinate shape: "
            f"{coords_of_displaced_nodes.shape} != "
            f"{reference_coordinates.shape}."
        )

    if communicator.rank == 0:
        print(
            "coarse phase labels:",
            np.unique(phase_indicator_array),
        )
        print(
            "deformed coordinate shape:",
            coords_of_displaced_nodes.shape,
        )
        print(
            "plot coordinate shape:",
            coordinates_for_plot.shape,
        )

    # ------------------------------------------------------------------------
    # Material assignment by coarse phase label
    # ------------------------------------------------------------------------
    (
        material_data_field,
        phase_field,
        conductivity_field,
    ) = create_material_fields(
        discretization,
        phase_indicator_array,
    )

    # Visualize coarse phase and conductivity field.
    if discretization.communicator.size == 1:
        plot_field_on_grid(
            coordinates_for_plot=coordinates_for_plot,
            field_to_plot=phase_field.s[0, 0],
            name="Coarse phase label",
        )
        plot_field_on_grid(
            coordinates_for_plot=coordinates_for_plot,
            field_to_plot=conductivity_field.s[0, 0],
            name="Coarse conductivity",
        )

    # ------------------------------------------------------------------------
    # Deformed coordinates and grid displacement
    # ------------------------------------------------------------------------
    deformed_coordinates_field = (
        discretization.get_displacement_sized_field(
            name="deformed_nodal_points_coordinates_inxyz"
        )
    )

    grid_displacement_field = (
        discretization.get_displacement_sized_field(
            name="grid_nodes_displacement_inxyz"
        )
    )

    grid_displacement_field.s.fill(0.0)
    grid_displacement_field.s[:, 0, ...] = (
        coords_of_displaced_nodes - reference_coordinates
    )

    deformed_coordinates_field.s[:, 0, ...] = (
        reference_coordinates
        + grid_displacement_field.s[:, 0, ...]
    )

    # ------------------------------------------------------------------------
    # Deformation gradient F = I + grad(u)
    # ------------------------------------------------------------------------
    deformation_gradient = (
        discretization.get_displacement_gradient_sized_field(
            name="Grid_Deformation_gradient_F_ijqxy"
        )
    )

    discretization.fft.communicate_ghosts(
        grid_displacement_field
    )
    discretization.apply_gradient_operator_mugrid(
        grid_displacement_field,
        deformation_gradient,
    )
    deformation_gradient.s[...] += np.eye(2)[
        :, :, None, None, None
    ]

    deformation_gradient_array = (
        deformation_gradient.s.transpose(2, 3, 4, 0, 1)
    )
    det_F = np.linalg.det(deformation_gradient_array)
    inv_F = np.linalg.pinv(
        deformation_gradient_array
    ).transpose(3, 4, 0, 1, 2)

    if discretization.communicator.size == 1:
        plot_field_on_grid(
            coordinates_for_plot=coordinates_for_plot,
            field_to_plot=det_F[0],
            name="det(F)",
        )
    if discretization.communicator.size == 1:
        plot_field_on_grid(
            coordinates_for_plot=coordinates_for_plot,
            field_to_plot=det_F_from_plot,
            name="det(F) from corrected periodic coordinates",
        )


    # ------------------------------------------------------------------------
    # Matrix-free Hessian and preconditioner
    # ------------------------------------------------------------------------
    def K_fun(x, Ax):
        """Apply the deformed-grid Hessian matrix."""
        discretization.apply_system_matrix_mugrid_deformed_grid(
            material_data_field=material_data_field,
            input_field_inxyz=x,
            output_field_inxyz=Ax,
            det_of_deformation_gradient=det_F,
            inv_of_deformation_gradient=inv_F,
        )
        discretization.fft.communicate_ghosts(Ax)

    reference_conductivity = np.eye(2)
    preconditioner = (
        discretization.get_preconditioner_Green_mugrid(
            reference_material_data_ijkl=reference_conductivity,
        )
    )

    def M_fun(x, Px):
        """Apply the Green-operator preconditioner."""
        discretization.fft.communicate_ghosts(x)
        discretization.apply_preconditioner_mugrid(
            preconditioner_Fourier_fnfnqks=preconditioner,
            input_nodal_field_fnxyz=x,
            output_nodal_field_fnxyz=Px,
        )

    solution_field = discretization.get_unknown_size_field(
        name="solution"
    )
    macro_gradient_field = (
        discretization.get_gradient_size_field(
            name="macro_gradient_field"
        )
    )
    rhs_field = discretization.get_unknown_size_field(
        name="rhs_field"
    )

    dimension = discretization.domain_dimension
    homogenized_A_ij = np.zeros((dimension, dimension))

    # ------------------------------------------------------------------------
    # Solve one cell problem for each macroscopic gradient direction
    # ------------------------------------------------------------------------
    for direction in range(dimension):
        macro_gradient = np.zeros(dimension)
        macro_gradient[direction] = 1.0

        macro_gradient_field.sg.fill(0.0)
        discretization.get_macro_gradient_field_mugrid(
            macro_gradient_ij=macro_gradient,
            macro_gradient_field_ijqxyz=macro_gradient_field,
        )

        macro_gradient_field.s[...] = np.einsum(
            "ij...,jk...->ik...",
            macro_gradient_field.s[...],
            inv_F,
        )
        discretization.fft.communicate_ghosts(
            field=macro_gradient_field
        )

        rhs_field.sg.fill(0.0)
        discretization.get_rhs_mugrid_deformed_grid(
            material_data_field_ijklqxyz=material_data_field,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            rhs_inxyz=rhs_field,
            det_of_deformation_gradient=det_F,
            inv_of_deformation_gradient=inv_F,
        )

        def callback(iteration, fields):
            residual_norm = fields["rr"]
            if discretization.communicator.rank == 0:
                print(
                    f"{iteration:5} "
                    f"norm of residual = {residual_norm:.5e}"
                )

        solution_field.sg.fill(0.0)
        Solvers.conjugate_gradients(
            comm=discretization.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,
            b=rhs_field,
            x=solution_field,
            prec=M_fun,
            rtol=SOLVER_RTOL,
            maxiter=SOLVER_MAXITER,
            callback=callback,
        )

        if discretization.communicator.size == 1:
            plot_field_on_grid(
                coordinates_for_plot=coordinates_for_plot,
                field_to_plot=solution_field.s[0, 0],
                name=(
                    "Solution field - macro gradient "
                    f"{macro_gradient}"
                ),
            )

        discretization.fft.communicate_ghosts(
            field=solution_field
        )

        sum_sol = discretization.mpi_reduction.sum(
            solution_field.s,
            axis=tuple(range(-3, 0)),
        )

        print(
            f"rank {MPI.COMM_WORLD.rank:6} "
            f"sum_sol = {sum_sol}"
        )

        homogenized_A_ij[direction, :] = (
            discretization.get_homogenized_stress_mugrid_deformed_grid(
                material_data_field_ijklqxyz=material_data_field,
                temperature_field_inxyz=solution_field,
                macro_gradient_field_ijqxyz=macro_gradient_field,
                det_of_deformation_gradient=det_F,
                inv_of_deformation_gradient=inv_F,
            )
        )

        if discretization.communicator.rank == 0:
            print(
                "Homogenized conductivity tangent =\n"
                + np.array2string(
                    homogenized_A_ij,
                    formatter={
                        "float_kind": lambda value: (
                            f"{value:0.8f}"
                        )
                    },
                )
            )

    elapsed_time = time.time() - start_time

    if discretization.communicator.rank == 0:
        print("\nSimulation finished")
        print("Input image:", INPUT_FILE)
        print("Coarse labels:", np.unique(phase_indicator_array))
        print("Homogenized conductivity:")
        print(
            np.array2string(
                homogenized_A_ij,
                formatter={
                    "float_kind": lambda value: f"{value:0.8f}"
                },
            )
        )
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
        print(f"Elapsed time: {elapsed_time / 60:.4f} minutes")
        print("det(F) min:", det_F_from_plot.min())
        print("det(F) max:", det_F_from_plot.max())
        print("det(F) interior min:", det_F_from_plot[1:-1, 1:-1].min())
        print("det(F) interior max:", det_F_from_plot[1:-1, 1:-1].max())
   # health_report = run_homogenization_health_check(det_F, homogenized_A_ij)

if __name__ == "__main__":
    main()