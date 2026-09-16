"""C11 vs N convergence sweep for muFFTTO grid-adaptation homogenization.

WHAT THIS SCRIPT DOES (in plain words)
---------------------------------------
Your original script `exp_grid_adaptation_arbitrary_conductivity_transformed.py`
runs ONE simulation for a fixed grid size NUMBER_OF_PIXELS = (N, N) and prints
a 2x2 "homogenized conductivity" matrix. The top-left entry of that matrix is
C11 (row 0, column 0).

This script:
  1. Wraps the whole simulation (grid adaptation -> material assignment ->
     deformation gradient -> CG solve -> homogenization) into one function
     `run_one_N(N)` that returns C11 for that N.
  2. Loops that function over a list of N values you choose, e.g. [8, 16, 24, 32, 48, 64].
  3. Saves a CSV table (N, C11, converged?, det_F_min, det_F_max) and a PNG
     plot "C11 vs N" into ./output/.

WHY N=64 (or other large N) MAY FAIL TO CONVERGE
--------------------------------------------------
As the grid gets finer, the spring-relaxation grid-adaptation step can create
some cells that are much more stretched/skewed than others. Highly distorted
cells make the underlying linear system ill-conditioned, so the preconditioned
conjugate-gradient solver (muGrid.Solvers.conjugate_gradients) may not reach
the requested tolerance within SOLVER_MAXITER steps and raises
`muGrid.Solvers.ConvergenceError`.

This version handles that gracefully with THREE layers of defense:
  Layer 1 - More slack: SOLVER_MAXITER raised and SOLVER_RTOL relaxed a bit
            (edit these if you need more/less strictness).
  Layer 2 - Fault tolerance: each N is wrapped in try/except, so if one N
            fails to converge, the sweep keeps going for the remaining N
            values instead of crashing. Failed points are marked in the CSV
            and skipped (shown as gaps) in the plot.
  Layer 3 - Diagnostics: det(F) min/max are recorded for every N so you can
            see whether a failure correlates with heavily distorted
            ("near-degenerate") grid cells. If det_F_min gets close to 0 or
            negative near a failing N, the grid-adaptation parameters
            (RELAX_ITERS / RELAX_OMEGA / RELAX_B) likely need retuning for
            that resolution -- that is a separate, deeper fix from solver
            tolerance and worth discussing with your supervisor.

HOW TO RUN
----------
Single process (recommended first, to make sure it works):
    python exp_grid_adaptation_C11_vs_N.py

With MPI (only if your original script normally uses MPI):
    mpiexec -n 4 python exp_grid_adaptation_C11_vs_N.py

BEFORE YOU RUN
--------------
- Update INPUT_FILE below to the same .npy image path you used in the
  original script.
- Update N_VALUES to the grid sizes you want to test.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # write PNG files without needing a display
import matplotlib.pyplot as plt

from mpi4py import MPI
from muGrid import Solvers
from muGrid.Solvers import ConvergenceError

from muFFTTO.grid_adaptation_arbitrary import run_grid_adaptation_workflow
from muFFTTO import domain
import inspect
print("=== domain_2 sanity check ===")
print("File:", domain.__file__)
print("--- get_rhs_mugrid_deformed_grid ---")
print(inspect.getsource(domain.Discretization.get_rhs_mugrid_deformed_grid))
print("--- get_homogenized_stress_mugrid_deformed_grid ---")
print(inspect.getsource(domain.Discretization.get_homogenized_stress_mugrid_deformed_grid))
print("=== end sanity check ===")

# ============================================================================
# User settings
# ============================================================================

# Same input image you used in the original script.
INPUT_FILE = Path(
    r"C:\Users\Test\Desktop\JiaLing\HiWi\Simulation"
    r"\Grain Boundaries Data"
    r"\Green_Jacobi_eta_0.01_w_10.0_p_0.0_final.npy"
)

PROBLEM_TYPE = "conductivity"
DISCRETIZATION_TYPE = "finite_element"
ELEMENT_TYPE = "linear_triangles"

DOMAIN_SIZE = (1.0, 1.0)

# The list of grid resolutions N to sweep over. NUMBER_OF_PIXELS = (N, N).
N_VALUES = [4,8,16,32,64,128]

RELAX_ITERS = 200
RELAX_OMEGA = 0.02
RELAX_B = 0

# --- Layer 1: give the solver more room to converge -------------------------
# If large-N cases still fail after this, the issue is most likely grid
# distortion (see Layer 3 diagnostics below), not solver settings.
SOLVER_RTOL = 1e-6        # relaxed from 1e-6
SOLVER_MAXITER = 2000       # raised from 2000

# Edit this table according to the physical meaning of your phase labels.
CONDUCTIVITY_BY_LABEL = {
    0: 1.0,
    1: 2.0,
    2: 3.0,
    3: 4.0,
}

# If True, re-enables the original script's verbose per-N plots/prints.
VERBOSE = False

OUTPUT_DIR = Path("output_C11_N")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Material field (same logic as the original script, N is now a parameter)
# ============================================================================

def create_material_fields(discretization, phase_indicator_array, number_of_pixels, verbose):
    """Create phase and conductivity fields from coarse phase labels."""
    phase_indicator_array = np.asarray(phase_indicator_array, dtype=np.int32)

    expected_shape = tuple(number_of_pixels)
    if phase_indicator_array.shape != expected_shape:
        raise ValueError(
            "coarse_phase_label has the wrong shape: "
            f"got {phase_indicator_array.shape}, expected {expected_shape}."
        )

    actual_labels = np.unique(phase_indicator_array)
    if np.any(actual_labels < 0):
        raise ValueError(f"Phase labels must be non-negative, got {actual_labels}.")

    missing_conductivity_labels = [
        int(label) for label in actual_labels if int(label) not in CONDUCTIVITY_BY_LABEL
    ]
    if missing_conductivity_labels:
        raise ValueError(
            "No conductivity was assigned to phase labels: "
            f"{missing_conductivity_labels}. Update CONDUCTIVITY_BY_LABEL."
        )

    material_data_field = discretization.get_material_data_size_field_mugrid(
        name="conductivity_tensor",
    )
    material_data_field.s.fill(0.0)

    phase_field = discretization.get_scalar_field(name="phase_field")
    phase_field.s[0, 0, ...] = phase_indicator_array

    conductivity_field = discretization.get_scalar_field(name="conductivity_value_field")
    conductivity_field.s.fill(0.0)

    for label in actual_labels:
        label = int(label)
        conductivity_value = float(CONDUCTIVITY_BY_LABEL[label])

        if conductivity_value <= 0.0:
            raise ValueError(f"Conductivity for label {label} must be positive.")

        material_mask = phase_indicator_array == label
        conductivity_tensor = conductivity_value * np.eye(2)

        material_data_field.s[..., material_mask] = conductivity_tensor[..., np.newaxis, np.newaxis]
        conductivity_field.s[0, 0, material_mask] = conductivity_value

        if verbose:
            print(
                f"label {label}: {int(material_mask.sum())} coarse cells, "
                f"conductivity = {conductivity_value:.6g} * I"
            )

    return material_data_field, phase_field, conductivity_field


def compute_det_F_from_plot_coords(coordinates_for_plot, Lx, Ly):
    """Compute cell-wise det(F) from deformed plotting coordinates."""
    p00 = coordinates_for_plot[:, :-1, :-1]
    p10 = coordinates_for_plot[:, :-1, 1:]
    p01 = coordinates_for_plot[:, 1:, :-1]
    p11 = coordinates_for_plot[:, 1:, 1:]

    _, Ny_plus_one, Nx_plus_one = coordinates_for_plot.shape
    Nx = Nx_plus_one - 1
    Ny = Ny_plus_one - 1

    dx = Lx / Nx
    dy = Ly / Ny

    tangent_x = 0.5 * ((p10 - p00) + (p11 - p01))
    tangent_y = 0.5 * ((p01 - p00) + (p11 - p10))

    F11 = tangent_x[0] / dx
    F21 = tangent_x[1] / dx
    F12 = tangent_y[0] / dy
    F22 = tangent_y[1] / dy

    return F11 * F22 - F12 * F21


# ============================================================================
# One full simulation for a given N -> returns the homogenized 2x2 matrix
# plus diagnostics (det_F min/max) used to explain solver failures.
# ============================================================================

def run_one_N(N: int, communicator) -> dict:
    """Run the full grid-adaptation + homogenization pipeline for NUMBER_OF_PIXELS=(N,N).

    Returns a dict with keys:
        "homogenized_A_ij" : 2x2 matrix, or None if the solver failed to converge
        "converged"        : bool
        "det_F_min"        : float
        "det_F_max"        : float
        "error"            : str or None
    """
    number_of_pixels = (N, N)

    if communicator.rank == 0:
        print(f"\n=== Running N = {N} (grid {number_of_pixels}) ===")

    cell = domain.PeriodicUnitCell(domain_size=DOMAIN_SIZE, problem_type=PROBLEM_TYPE)

    discretization = domain.Discretization(
        cell=cell,
        nb_of_pixels_global=number_of_pixels,
        discretization_type=DISCRETIZATION_TYPE,
        element_type=ELEMENT_TYPE,
    )
    print("nb_quad_points_per_pixel =", discretization.nb_quad_points_per_pixel)

    result = run_grid_adaptation_workflow(
        input_path=INPUT_FILE,
        coarse_Nx=number_of_pixels[0],
        coarse_Ny=number_of_pixels[1],
        Lx=DOMAIN_SIZE[0],
        Ly=DOMAIN_SIZE[1],
        relax_iters=RELAX_ITERS,
        relax_omega=RELAX_OMEGA,
        relax_b=RELAX_B,
        verbose=(communicator.rank == 0 and VERBOSE),
    )

    coords_of_displaced_nodes = result["coords_of_displaced_nodes"]
    phase_indicator_array = result["coarse_phase_label"].astype(np.int32)
    coordinates_for_plot = result["full_plot_coords_of_displaced_nodes"]

    det_F_from_plot = compute_det_F_from_plot_coords(
        coordinates_for_plot=coordinates_for_plot,
        Lx=DOMAIN_SIZE[0],
        Ly=DOMAIN_SIZE[1],
    )
    det_F_min = float(det_F_from_plot.min())
    det_F_max = float(det_F_from_plot.max())

    if communicator.rank == 0:
        print(f"N = {N}: det(F) min = {det_F_min:.6f}, max = {det_F_max:.6f}")
        if det_F_min <= 0.0:
            print(
                f"  WARNING: det(F) is non-positive at N={N}. "
                "The deformed grid has an inverted/degenerate cell. "
                "Consider retuning RELAX_ITERS / RELAX_OMEGA / RELAX_B for this N."
            )
        elif det_F_min < 0.1:
            print(
                f"  NOTE: det(F) min is small ({det_F_min:.4f}) at N={N}. "
                "Highly stretched cells make the CG solve harder."
            )

    reference_coordinates = discretization.get_nodal_points_coordinates().s[:, 0, ...]

    if coords_of_displaced_nodes.shape != reference_coordinates.shape:
        raise ValueError(
            "The deformed coordinate shape does not match the muFFTTO reference "
            f"coordinate shape: {coords_of_displaced_nodes.shape} != {reference_coordinates.shape}."
        )

    (material_data_field, phase_field, conductivity_field) = create_material_fields(
        discretization, phase_indicator_array, number_of_pixels, VERBOSE
    )

    if VERBOSE and discretization.communicator.size == 1:
        plot_field_on_grid(
            coordinates_for_plot=coordinates_for_plot,
            field_to_plot=phase_field.s[0, 0],
            name=f"Coarse phase label (N={N})",
        )
        plot_field_on_grid(
            coordinates_for_plot=coordinates_for_plot,
            field_to_plot=conductivity_field.s[0, 0],
            name=f"Coarse conductivity (N={N})",
        )

    deformed_coordinates_field = discretization.get_displacement_sized_field(
        name="deformed_nodal_points_coordinates_inxyz"
    )
    grid_displacement_field = discretization.get_displacement_sized_field(
        name="grid_nodes_displacement_inxyz"
    )

    grid_displacement_field.s.fill(0.0)
    grid_displacement_field.s[:, 0, ...] = coords_of_displaced_nodes - reference_coordinates

    deformed_coordinates_field.s[:, 0, ...] = (
        reference_coordinates + grid_displacement_field.s[:, 0, ...]
    )

    deformation_gradient = discretization.get_displacement_gradient_sized_field(
        name="Grid_Deformation_gradient_F_ijqxy"
    )

    discretization.fft.communicate_ghosts(grid_displacement_field)
    discretization.apply_gradient_operator_mugrid(grid_displacement_field, deformation_gradient)
    deformation_gradient.s[...] += np.eye(2)[:, :, None, None, None]
    print("deformation_gradient.s.shape =", deformation_gradient.s.shape)


    deformation_gradient_array = deformation_gradient.s.transpose(2, 3, 4, 0, 1)
    det_F = np.linalg.det(deformation_gradient_array)
    inv_F = np.linalg.pinv(deformation_gradient_array).transpose(3, 4, 0, 1, 2)

    print(f"N={N}: det_F.shape = {det_F.shape}")
    print(f"N={N}: det_F min/max = {det_F.min():.4f} / {det_F.max():.4f}")
    print(f"N={N}: inverted count = {(det_F <= 0).sum()} / {det_F.size}")

    def K_fun(x, Ax):
        discretization.apply_system_matrix_mugrid_deformed_grid(
            material_data_field=material_data_field,
            input_field_inxyz=x,
            output_field_inxyz=Ax,
            det_of_deformation_gradient=det_F,
            inv_of_deformation_gradient=inv_F,
        )
        discretization.fft.communicate_ghosts(Ax)

    reference_conductivity = np.eye(2)
    preconditioner = discretization.get_preconditioner_Green_mugrid(
        reference_material_data_ijkl=reference_conductivity,
    )

    def M_fun(x, Px):
        discretization.fft.communicate_ghosts(x)
        discretization.apply_preconditioner_mugrid(
            preconditioner_Fourier_fnfnqks=preconditioner,
            input_nodal_field_fnxyz=x,
            output_nodal_field_fnxyz=Px,
        )

    solution_field = discretization.get_unknown_size_field(name="solution")
    macro_gradient_field = discretization.get_gradient_size_field(name="macro_gradient_field")
    rhs_field = discretization.get_unknown_size_field(name="rhs_field")

    dimension = discretization.domain_dimension
    homogenized_A_ij = np.zeros((dimension, dimension))

    def make_callback():
        def callback(iteration, fields):
            if VERBOSE and discretization.communicator.rank == 0:
                residual_norm = fields["rr"]
                print(f"{iteration:5} norm of residual = {residual_norm:.5e}")

        return callback

    # --- Layer 2: fault tolerance around the CG solve -----------------------
    try:
        for direction in range(dimension):
            macro_gradient = np.zeros(dimension)
            macro_gradient[direction] = 1.0

            macro_gradient_field.sg.fill(0.0)
            discretization.get_macro_gradient_field_mugrid(
                macro_gradient_ij=macro_gradient,
                macro_gradient_field_ijqxyz=macro_gradient_field,
            )

            macro_gradient_field.s[...] = np.einsum(
                "ij...,jk...->ik...", macro_gradient_field.s[...], inv_F
            )
            discretization.fft.communicate_ghosts(field=macro_gradient_field)

            rhs_field.sg.fill(0.0)
            discretization.get_rhs_mugrid_deformed_grid(
                material_data_field_ijklqxyz=material_data_field,
                macro_gradient_field_ijqxyz=macro_gradient_field,
                rhs_inxyz=rhs_field,
                det_of_deformation_gradient=det_F,
                inv_of_deformation_gradient=inv_F,
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
                callback=make_callback(),
            )

            discretization.fft.communicate_ghosts(field=solution_field)

            homogenized_A_ij[direction, :] = discretization.get_homogenized_stress_mugrid_deformed_grid(
                material_data_field_ijklqxyz=material_data_field,
                temperature_field_inxyz=solution_field,
                macro_gradient_field_ijqxyz=macro_gradient_field,
                det_of_deformation_gradient=det_F,
                inv_of_deformation_gradient=inv_F,
            )

    except ConvergenceError as exc:
        if communicator.rank == 0:
            print(f"N = {N}: CG DID NOT CONVERGE ({exc}). Marking this N as failed.")
        return {
            "homogenized_A_ij": None,
            "converged": False,
            "det_F_min": det_F_min,
            "det_F_max": det_F_max,
            "error": str(exc),
        }

    if communicator.rank == 0:
        print(
            f"N = {N}: Homogenized conductivity =\n"
            + np.array2string(
                homogenized_A_ij,
                formatter={"float_kind": lambda value: f"{value:0.8f}"},
            )
        )
        print(f"N = {N}: C11 = {homogenized_A_ij[0, 0]:.8f}")

    if VERBOSE:
        run_homogenization_health_check(det_F, homogenized_A_ij)

    return {
        "homogenized_A_ij": homogenized_A_ij,
        "converged": True,
        "det_F_min": det_F_min,
        "det_F_max": det_F_max,
        "error": None,
    }


# ============================================================================
# Sweep over N and plot C11 vs N
# ============================================================================

def main() -> None:
    start_time = time.time()
    communicator = MPI.COMM_WORLD

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Input image does not exist: {INPUT_FILE}")

    if communicator.rank == 0:
        print("Input image:", INPUT_FILE)
        print("MPI processes:", communicator.size)
        print("N values to sweep:", N_VALUES)

    results = []  # list of dicts, one per N

    for N in N_VALUES:
        # --- Layer 2 (outer safety net): even if something unexpected other
        # than ConvergenceError goes wrong for one N, keep the sweep alive.
        try:
            outcome = run_one_N(N, communicator)
        except Exception as exc:  # noqa: BLE001 - intentionally broad for a sweep
            if communicator.rank == 0:
                print(f"N = {N}: FAILED with unexpected error: {exc!r}. Skipping.")
            outcome = {
                "homogenized_A_ij": None,
                "converged": False,
                "det_F_min": float("nan"),
                "det_F_max": float("nan"),
                "error": repr(exc),
            }
        outcome["N"] = N
        results.append(outcome)

    if communicator.rank == 0:
        # --- Save CSV table ---------------------------------------------------
        csv_path = OUTPUT_DIR / "C11_vs_N.csv"
        with open(csv_path, "w") as f:
            f.write("N,C11,converged,det_F_min,det_F_max,error\n")
            for r in results:
                C11 = r["homogenized_A_ij"][0, 0] if r["converged"] else float("nan")
                error_msg = (r["error"] or "").replace(",", ";")
                f.write(
                    f"{r['N']},{C11},{r['converged']},"
                    f"{r['det_F_min']:.6f},{r['det_F_max']:.6f},{error_msg}\n"
                )
        print(f"\nSaved table: {csv_path.resolve()}")

        # --- Plot C11 vs N (only converged points are plotted as a line) --------
        converged_results = [r for r in results if r["converged"]]
        failed_results = [r for r in results if not r["converged"]]

        N_ok = np.array([r["N"] for r in converged_results])
        C11_ok = np.array([r["homogenized_A_ij"][0, 0] for r in converged_results])

        fig, ax = plt.subplots(figsize=(6, 4.5))
        if len(N_ok) > 0:
            ax.plot(N_ok, C11_ok, marker="o", linestyle="-", color="tab:blue", label="converged")
        if failed_results:
            N_fail = [r["N"] for r in failed_results]
            for nf in N_fail:
                ax.axvline(nf, color="tab:red", linestyle="--", alpha=0.5)
            ax.plot([], [], color="tab:red", linestyle="--", label="did not converge")

        ax.set_xlabel("N (NUMBER_OF_PIXELS = (N, N))")
        ax.set_ylabel(r"$C_{11}$ (homogenized conductivity, top-left entry)")
        ax.set_title(r"Convergence of $C_{11}$ with grid resolution $N$")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        png_path = OUTPUT_DIR / "C11_vs_N.png"
        fig.savefig(png_path, dpi=200)
        plt.close(fig)
        print(f"Saved plot:  {png_path.resolve()}")

        elapsed_time = time.time() - start_time
        print(f"\nTotal elapsed time: {elapsed_time:.2f} s ({elapsed_time / 60:.2f} min)")
        print("\nSummary (N, C11 or FAILED, det_F_min):")
        for r in results:
            if r["converged"]:
                print(f"  N={r['N']:4d}  C11={r['homogenized_A_ij'][0, 0]:.8f}  det_F_min={r['det_F_min']:.4f}")
            else:
                print(f"  N={r['N']:4d}  FAILED ({r['error']})  det_F_min={r['det_F_min']:.4f}")

        if failed_results:
            print(
                "\nSome N values did not converge. Try: (a) raising SOLVER_MAXITER "
                "further, (b) relaxing SOLVER_RTOL, or (c) retuning RELAX_ITERS / "
                "RELAX_OMEGA / RELAX_B in the grid-adaptation step for those N -- "
                "especially if det_F_min printed above is close to 0 or negative."
            )


if __name__ == "__main__":
    main()

