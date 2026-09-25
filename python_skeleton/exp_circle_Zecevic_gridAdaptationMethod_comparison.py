import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from muFFTTO.grid_adaptation_methods import (
    adapt_grid_to_circle,
    pack_adapted_grid_to_fields,
    print_field_summary,
)


DOMAIN_SIZE = 40.0
CIRCLE_CENTER = (20.0, 20.0)
CIRCLE_RADIUS = 10.0
NB_ITERS = 400
OMEGA = 0.8
K0 = 0.25
KMIN = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run periodic-style analytical grid adaptation for one specified N and b."
    )
    parser.add_argument(
        "--N",
        type=int,
        default=32,
        help="Number of cells / stored points per direction (default: 32)",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=1.0,
        help="Stiffness exponent b (default: 1.0)",
    )
    parser.add_argument(
        "--ghosts",
        type=int,
        default=1,
        help="Number of muGrid ghost cells per side (default: 1)",
    )
    parser.add_argument(
        "--no-store-positions",
        action="store_true",
        help="Do not store P0 and P as muGrid fields",
    )
    return parser.parse_args()


def close_periodic_grid_for_plot(P: np.ndarray, L: float) -> np.ndarray:
    """
    Create a temporary closed grid for plotting.

    Input shape is [xy, N, N]. Output shape is [xy, N+1, N+1].
    The last column copies the first column with x shifted by +L.
    The last row copies the first row with y shifted by +L.
    """
    N = P.shape[1]
    Pc = np.zeros((2, N + 1, N + 1), dtype=P.dtype)

    Pc[:, :N, :N] = P
    Pc[:, N, :N] = P[:, 0, :]
    Pc[0, N, :N] += L

    Pc[:, :N, N] = P[:, :, 0]
    Pc[1, :N, N] += L

    Pc[:, N, N] = P[:, 0, 0]
    Pc[0, N, N] += L
    Pc[1, N, N] += L

    return Pc


def plot_initial_and_adapted_mesh(
    P0: np.ndarray,
    P: np.ndarray,
    N: int,
    b: float,
    elapsed_time: float,
    L: float,
) -> None:
    P0c = close_periodic_grid_for_plot(P0, L=L)
    Pc = close_periodic_grid_for_plot(P, L=L)
    npx, npy = P0c.shape[1], P0c.shape[2]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax_init, ax_def = axes

    for i in range(npx):
        ax_init.plot(P0c[0, i, :], P0c[1, i, :], color="0.2", linewidth=0.7)
    for j in range(npy):
        ax_init.plot(P0c[0, :, j], P0c[1, :, j], color="0.2", linewidth=0.7)

    ax_init.set_aspect("equal", adjustable="box")
    ax_init.set_title("Initial periodic grid")
    ax_init.set_xlabel("x")
    ax_init.set_ylabel("y")

    for i in range(npx):
        ax_def.plot(Pc[0, i, :], Pc[1, i, :], color="0.2", linewidth=0.7)
    for j in range(npy):
        ax_def.plot(Pc[0, :, j], Pc[1, :, j], color="0.2", linewidth=0.7)

    ax_def.set_aspect("equal", adjustable="box")
    ax_def.set_title("Adapted periodic grid")
    ax_def.set_xlabel("x")
    ax_def.set_ylabel("y")

    fig.suptitle(
        f"Periodic grid adaptation to circle | N = {N}, b = {b:.2f}, time = {elapsed_time:.4f} s"
    )
    fig.tight_layout()
    plt.show()


def print_numpy_summary(field_bundle: dict) -> None:
    disp = field_bundle["numpy"]["displacement"]
    inside = field_bundle["numpy"]["inside"]
    interface = field_bundle["numpy"]["interface"]
    dist = field_bundle["numpy"]["dist"]
    k_node = field_bundle["numpy"]["k_node"]

    print("\nNumpy summary")
    print("-" * 72)
    print("displacement shape:", disp.shape)
    print("ux min/max:", disp[0].min(), disp[0].max())
    print("uy min/max:", disp[1].min(), disp[1].max())
    print("inside sum:", inside.sum())
    print("interface sum:", interface.sum())
    print("dist min/max:", dist.min(), dist.max())
    print("k_node min/max:", k_node.min(), k_node.max())


def main() -> None:
    args = parse_args()

    start_time = time.perf_counter()
    result = adapt_grid_to_circle(
        N=args.N,
        L=DOMAIN_SIZE,
        center=CIRCLE_CENTER,
        R=CIRCLE_RADIUS,
        iters=NB_ITERS,
        omega=OMEGA,
        b=args.b,
        k0=K0,
        kmin=KMIN,
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    plot_initial_and_adapted_mesh(
        result["P0"],
        result["P"],
        N=args.N,
        b=args.b,
        elapsed_time=elapsed_time,
        L=DOMAIN_SIZE,
    )

    field_bundle = pack_adapted_grid_to_fields(
        result,
        ghosts=args.ghosts,
        store_positions=not args.no_store_positions,
        verbose=True,
    )

    print_field_summary(field_bundle)
    print_numpy_summary(field_bundle)
    #print(field_bundle)


if __name__ == "__main__":
    main()

'''
//successful version

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt


from muFFTTO.feature import adapt_grid_to_circle



DOMAIN_SIZE = 40.0
CIRCLE_CENTER = (20.0, 20.0)
CIRCLE_RADIUS = 10.0
NB_ITERS = 400
OMEGA = 0.8
K0 = 0.25
KMIN = 0.005



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run periodic-style analytical grid adaptation for one specified N and b."
    )
    parser.add_argument(
        "--N",
        type=int,
        default=32,
        help="Number of cells / stored points per direction (default: 32)",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=1.0,
        help="Stiffness exponent b (default: 1.0)",
    )
    return parser.parse_args()




def close_periodic_grid_for_plot(P: np.ndarray, L: float) -> np.ndarray:
    """
    Create a temporary closed grid for plotting.


    Input shape is [xy, N, N]. Output shape is [xy, N+1, N+1].
    The last column copies the first column with x shifted by +L.
    The last row copies the first row with y shifted by +L.
    """
    N = P.shape[1]
    Pc = np.zeros((2, N + 1, N + 1), dtype=P.dtype)


    Pc[:, :N, :N] = P
    Pc[:, N, :N] = P[:, 0, :]
    Pc[0, N, :N] += L


    Pc[:, :N, N] = P[:, :, 0]
    Pc[1, :N, N] += L


    Pc[:, N, N] = P[:, 0, 0]
    Pc[0, N, N] += L
    Pc[1, N, N] += L


    return Pc




def plot_initial_and_adapted_mesh(
    P0: np.ndarray,
    P: np.ndarray,
    N: int,
    b: float,
    elapsed_time: float,
    L: float,
) -> None:
    P0c = close_periodic_grid_for_plot(P0, L=L)
    Pc = close_periodic_grid_for_plot(P, L=L)
    npx, npy = P0c.shape[1], P0c.shape[2]


    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax_init, ax_def = axes


    for i in range(npx):
        ax_init.plot(P0c[0, i, :], P0c[1, i, :], color="0.2", linewidth=0.7)
    for j in range(npy):
        ax_init.plot(P0c[0, :, j], P0c[1, :, j], color="0.2", linewidth=0.7)


    ax_init.set_aspect("equal", adjustable="box")
    ax_init.set_title("Initial periodic grid")
    ax_init.set_xlabel("x")
    ax_init.set_ylabel("y")


    for i in range(npx):
        ax_def.plot(Pc[0, i, :], Pc[1, i, :], color="0.2", linewidth=0.7)
    for j in range(npy):
        ax_def.plot(Pc[0, :, j], Pc[1, :, j], color="0.2", linewidth=0.7)


    ax_def.set_aspect("equal", adjustable="box")
    ax_def.set_title("Adapted periodic grid")
    ax_def.set_xlabel("x")
    ax_def.set_ylabel("y")


    fig.suptitle(
        f"Periodic grid adaptation to circle | N = {N}, b = {b:.2f}, time = {elapsed_time:.4f} s"
    )
    fig.tight_layout()
    plt.show()




def main() -> None:
    args = parse_args()


    start_time = time.perf_counter()
    result = adapt_grid_to_circle(
        N=args.N,
        L=DOMAIN_SIZE,
        center=CIRCLE_CENTER,
        R=CIRCLE_RADIUS,
        iters=NB_ITERS,
        omega=OMEGA,
        b=args.b,
        k0=K0,
        kmin=KMIN,
    )
    end_time = time.perf_counter()


    elapsed_time = end_time - start_time


    plot_initial_and_adapted_mesh(
        result["P0"],
        result["P"],
        N=args.N,
        b=args.b,
        elapsed_time=elapsed_time,
        L=DOMAIN_SIZE,
    )



if __name__ == "__main__":
    main()
'''