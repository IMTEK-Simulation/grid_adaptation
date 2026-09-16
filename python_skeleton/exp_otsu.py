"""
Build muFFTTO-compatible material fields from a two-dimensional
grain-boundary image.

The script loads a grayscale ``.npy`` image and applies an Otsu-based
image-processing pipeline to detect grain boundaries and segment phase
regions. The image is converted into the following spatial data:

- an edge mask identifying detected grain boundaries;
- a binary phase mask separating background and foreground;
- a connected-component label map assigning an integer label to each
  separated phase region.

A muFFTTO ``PeriodicUnitCell`` and ``Discretization`` are then created
with the same resolution as the input image. Consequently, each image
pixel corresponds to one cell in the simulation grid.

The raw image and segmentation results are stored as scalar muGrid
fields. A distinct isotropic conductivity tensor is generated for each
region label and assigned to the corresponding cells in the muFFTTO
material-data field.

The script also produces visualizations of the detected edges and
labelled phase regions. It is intended as a preprocessing experiment
that connects image-based microstructure data to simulation-ready
muFFTTO material fields.

Notes
-----
The current implementation treats every connected foreground region as
an independent material label. Grouping physically equivalent regions
into shared material classes can be added in a later processing step.
"""

import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import muGrid

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from muFFTTO.otsu import otsu_edgeDetection_and_phaseIndicator

from pathlib import Path
import sys
simulation_dir = Path(__file__).resolve().parents[1]
mufftto_repo_dir = simulation_dir / "muFFTTO"
sys.path.insert(0, str(mufftto_repo_dir))
from muFFTTO import domain

# ============================================================
# Import Data and Find Edges & Phases
# ============================================================
path = r'C:\Users\Test\Desktop\JiaLing\HiWi\Simulation\Grain Boundaries Data\Green_Jacobi_eta_0.01_w_10.0_p_0.0_final.npy'
data = np.load(path).astype(np.float32)
if data.ndim != 2:
    raise ValueError(
        f"Expected a 2D input image, but got shape {data.shape}."
    )
print('Loaded data successfully')

start_time = time.time()
# Apply otsu edge detection, get (1)edge_mask (2)phase_mask_binary (3)phase_mask_label
results = otsu_edgeDetection_and_phaseIndicator(data)
edge_mask = results["edge_mask"]
phase_mask_binary = results["phase_mask_binary"]
phase_mask_label = results["phase_mask_label"]

end_time = time.time()
elapsed_time = end_time - start_time

print('elapsed_time: ', f"{elapsed_time:.4f}", "s", flush=True)
print("data shape:", data.shape)
print("edge_mask shape:", edge_mask.shape)
print("phase_mask_binary shape:", phase_mask_binary.shape)
print("phase_mask_label shape:", phase_mask_label.shape)

print("edge_mask dtype:", edge_mask.dtype)
print("phase_mask_binary dtype:", phase_mask_binary.dtype)
print("phase_mask_label dtype:", phase_mask_label.dtype)

if "number_of_phase_regions" in results:
    print("number_of_phase_regions:", results["number_of_phase_regions"])

# ============================================================
# Discretization setup — sized to match the loaded image
# ============================================================
problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
geometry_ID = 'square_inclusion'

domain_size = (1, 1)
number_of_pixels = data.shape  # (H, W) -> match the real image resolution

my_cell = domain.PeriodicUnitCell(
    domain_size=domain_size,
    problem_type=problem_type,
)

discretization = domain.Discretization(
    cell=my_cell,
    nb_of_pixels_global=number_of_pixels,
    discretization_type=discretization_type,
    element_type=element_type,
)

# ============================================================
# Material data field: one conductivity tensor per region label
# ============================================================
if "number_of_phase_regions" not in results:
    raise ValueError(
        "Missing 'number_of_phase_regions' in results. "
        "Call otsu_edgeDetection_and_phaseIndicator("
        "..., regions_label=True)."
    )

nb_of_phase_regions = int(results["number_of_phase_regions"])

# Label 0 is background, Foreground region labels are 1, 2, ..., nb_of_phase_regions.
nb_of_materials = nb_of_phase_regions + 1

# phase_mask_label is the region-ID image:
# 0 = background, 1...N = separate connected regions.
phase_indicator_array = phase_mask_label.astype(np.int32)

expected_labels = np.arange(nb_of_materials, dtype=np.int32)
actual_labels = np.unique(phase_indicator_array)

if not np.array_equal(actual_labels, expected_labels):
    raise ValueError(
        "Unexpected labels in phase_mask_label. "
        f"Expected {expected_labels.tolist()}, "
        f"but found {actual_labels.tolist()}."
    )

if phase_indicator_array.shape != data.shape:
    raise ValueError(
        "phase_mask_label must have the same shape as the raw image. "
        f"Got {phase_indicator_array.shape} and {data.shape}."
    )

# ------------------------------------------------------------
# Create one 2x2 isotropic conductivity tensor for each label.
# conductivity_C[k] is the tensor assigned to region label k.
# k = 0 -> background conductivity = 0.0 * I
# k = 1 -> region 1 conductivity = 1.0 * I
# k = 2 -> region 2 conductivity = 2.0 * I
# ------------------------------------------------------------

conductivity_C = np.zeros(
    (nb_of_materials, 2, 2),
    dtype=np.float64,
)

for k in range(nb_of_materials):
    conductivity_value = float(k)

    conductivity_C[k] = np.array(
        [
            [conductivity_value, 0.0],
            [0.0, conductivity_value],
        ],
        dtype=np.float64,
    )

print("number of foreground regions:", nb_of_phase_regions)
print("number of materials including background:", nb_of_materials)
print("conductivity_C shape:", conductivity_C.shape)

for k in range(nb_of_materials):
    print(
        f"label {k}: "
        f"conductivity = {conductivity_C[k, 0, 0]:.1f} * I"
    )


# ------------------------------------------------------------
# Create muFFTTO material tensor field.
# ------------------------------------------------------------
material_data_field_C_global = (
    discretization.get_material_data_size_field_mugrid(
        name="conductivity_tensor",
    )
)
material_data_field_C_global.s.fill(0.0)

# ------------------------------------------------------------
# Store image-processing data in discretization scalar fields.
# ------------------------------------------------------------

# Original grayscale image
raw_data_field = discretization.get_scalar_field(
    name="raw_data_field",
)
raw_data_field.s[0, 0, ...] = data

# Detected contour/edge field: 0 or 1
edge_mask_field = discretization.get_scalar_field(
    name="edge_mask_field",
)
edge_mask_field.s[0, 0, ...] = edge_mask

# Otsu binary segmentation: 0 or 1
phase_mask_binary_field = discretization.get_scalar_field(
    name="phase_mask_binary_field",
)
phase_mask_binary_field.s[0, 0, ...] = phase_mask_binary

# Connected-component labels: 0, 1, 2, ..., N
phase_mask_label_field = discretization.get_scalar_field(
    name="phase_mask_label_field",
)
phase_mask_label_field.s[0, 0, ...] = phase_mask_label

# ------------------------------------------------------------
# Assign conductivity tensor C[k] to all pixels with label k.
# ------------------------------------------------------------

for k in range(nb_of_materials):
    material_mask = phase_indicator_array == k

    material_data_field_C_global.s[..., material_mask] = (
        conductivity_C[k][..., np.newaxis, np.newaxis]
    )

    print(
        f"label {k}: "
        f"{int(material_mask.sum())} pixels; "
        f"conductivity = {conductivity_C[k, 0, 0]:.1f}"
    )


# ------------------------------------------------------------
# Check storage shapes.
# ------------------------------------------------------------

print("raw data field shape:", raw_data_field.s.shape)
print("edge mask field shape:", edge_mask_field.s.shape)
print(
    "binary phase-mask field shape:",
    phase_mask_binary_field.s.shape,
)
print(
    "region-label phase field shape:",
    phase_mask_label_field.s.shape,
)
print(
    "material tensor field shape:",
    material_data_field_C_global.s.shape,
)
print(discretization.field_collection)
print(dir(discretization.field_collection))

# ============================================================
# Plot 1: Original image with edge-mask overlay (from discretization fields)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(raw_data_field.s[0, 0, ...], cmap="gray", origin="lower")

edge_overlay = np.ma.masked_where(edge_mask_field.s[0, 0, ...] == 0, edge_mask_field.s[0, 0, ...])
ax.imshow(
    edge_overlay,
    cmap=ListedColormap(["green"]),
    alpha=1,
    origin="lower",
)

ax.set_title("Original image with detected edge mask")
ax.set_xlabel("y index")
ax.set_ylabel("x index")

plt.tight_layout()
plt.savefig("original_with_edge_mask.png", dpi=200, bbox_inches="tight")
plt.show()

# ============================================================
# Plot 2: Labelled phase-mask regions (reuse Otsu's phase_mask_label)
# ============================================================
labels = phase_mask_label_field.s[0, 0, ...].astype(np.int32)  # already computed inside otsu_edgeDetection_and_phaseIndicator
num_labels = int(labels.max()) + 1  # 0 = background
number_of_regions = num_labels - 1
print("number_of_phase_regions:", number_of_regions)

label_display = labels.astype(float)
label_display[labels == 0] = np.nan

fig, ax = plt.subplots(figsize=(10, 10))

image = ax.imshow(
    label_display,
    cmap="nipy_spectral",
    interpolation="nearest",
    origin="lower",
)

# compute centroid + area per label to place text annotations
min_label_area = 10
for label_id in range(1, num_labels):
    ys, xs = np.where(labels == label_id)
    area = ys.size
    if area < min_label_area:
        continue

    center_x = xs.mean()
    center_y = ys.mean()

    ax.text(
        center_x,
        center_y,
        str(label_id),
        color="white",
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
        bbox={
            "facecolor": "black",
            "alpha": 0.55,
            "edgecolor": "none",
            "pad": 1.0,
        },
    )

ax.set_title(f"Phase-mask connected regions: {number_of_regions} regions")
ax.set_xlabel("y index")
ax.set_ylabel("x index")

colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
colorbar.set_label("Connected-component label")

plt.tight_layout()
plt.savefig("phase_mask_labels.png", dpi=200, bbox_inches="tight")
plt.show()

# notes:original coordinates --> ref_grid_coords_ixyz = discretization.get_nodal_points_coordinates().s[:, 0, ...]
