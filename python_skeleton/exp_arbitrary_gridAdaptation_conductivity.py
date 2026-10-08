from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from muGrid import Solvers

# Adjust this import if your workflow is stored in another module.
from muFFTTO.grid_adaptation_methods_Zecevic_arbitrary import (
    adapt_grid_to_arbitrary_shape,
)
from muFFTTO import domain
from muFFTTO.visualization_utils import plot_field_on_grid
from muFFTTO.otsu import otsu_edgeDetection_and_phaseIndicator

# from muFFTTO.check_homogenization_health import run_homogenization_health_check

# ============================================================================
# User settings
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "Grain Boundaries Data"
    / "Green_Jacobi_eta_0.01_w_10.0_p_0.0_final.npy"
)

PROBLEM_TYPE = "conductivity"
DISCRETIZATION_TYPE = "finite_element"
ELEMENT_TYPE = "bilinear_rectangle"

DOMAIN_SIZE = (1.0, 1.0)
NUMBER_OF_PIXELS = (32,32)

RELAX_ITERS = 400
RELAX_OMEGA = 0.1
RELAX_B = 0.5

SOLVER_RTOL = 1e-6
SOLVER_MAXITER = 2000

CONDUCTIVITY_BY_LABEL = {
    0: 1.0,
    1: 20.0,
    2: 30.0,
    3: 40.0,
}

def main():
    cell = domain.PeriodicUnitCell(domain_size=DOMAIN_SIZE,problem_type=PROBLEM_TYPE)
    discretization = domain.Discretization(cell=cell,nb_of_pixels_global=NUMBER_OF_PIXELS,discretization_type=DISCRETIZATION_TYPE,element_type=ELEMENT_TYPE)
    # Obtain deformed grids coords of arbitrary shape
    result = adapt_grid_to_arbitrary_shape(INPUT_FILE)

    print("Done")

if __name__ == "__main__":
    main()

