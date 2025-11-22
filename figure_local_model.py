# Visualize the local D2 zero-divisor model with PyVista and Matplotlib support.

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import marching_cubes
import pyvista as pv


def compute_local_model_grid(
    num_samples: int = 81,
    grid_limit: float = 2.0,
    e3_offset: float = 0.5,
    e4_offset: float = 0.5,
):
    """Compute D2(X,Y,Z) on a cubic grid around the cyclic slice."""
    x = np.linspace(-grid_limit, grid_limit, num_samples)
    y = np.linspace(-grid_limit, grid_limit, num_samples)
    z = np.linspace(-grid_limit, grid_limit, num_samples)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    v1 = np.zeros(X.shape + (8,))
    v2 = np.zeros_like(v1)

    v1[..., 0], v1[..., 1], v1[..., 2] = X, Y, Z
    v1[..., 3] = e3_offset
    v2[..., 0], v2[..., 1], v2[..., 2] = Y, Z, X
    v2[..., 4] = e4_offset

    n1 = np.sum(v1**2, axis=-1)
    n2 = np.sum(v2**2, axis=-1)
    dot = np.sum(v1 * v2, axis=-1)
    d2 = (n1 - n2) ** 2 + 4 * dot**2
    return x, y, z, d2, dot


def plot_z0_slice(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    d2_volume: np.ndarray,
    epsilon: float,
    output_path: str = "sedenion_local_singularity_2d.png",
):
    """Plot the Z=0 slice (heat map + contour comparison)."""
    z_index = int(np.argmin(np.abs(z)))
    slice_d2 = d2_volume[:, :, z_index]

    X, Y = np.meshgrid(x, y, indexing="ij")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("D2 slice at Z=0 (log scale)")
    plt.imshow(
        np.log1p(slice_d2).T,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="magma",
    )
    plt.colorbar(label="log(1 + D2)")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.subplot(1, 2, 2)
    plt.title("Contour vs theory (XY = 0)")
    plt.contour(
        X,
        Y,
        slice_d2,
        levels=[epsilon],
        colors="red",
        linewidths=2,
    )
    plt.axhline(0, color="blue", linestyle="--", linewidth=1.5, label="Y = 0")
    plt.axvline(0, color="blue", linestyle="--", linewidth=1.5, label="X = 0")
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def export_isosurface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    d2_volume: np.ndarray,
    dot_volume: np.ndarray,
    epsilon: float,
    screenshot_path: str | None = "sedenion_local_singularity_3d.png",
    show_interactive: bool = True,
):
    """Build and optionally show/export the marching-cubes isosurface."""
    spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    verts, faces, _, _ = marching_cubes(d2_volume, level=epsilon, spacing=spacing)
    verts += np.array([x[0], y[0], z[0]])

    interpolator = RegularGridInterpolator((x, y, z), dot_volume, bounds_error=False, fill_value=0)
    colors = interpolator(verts)

    faces_pv = np.column_stack((np.full(len(faces), 3), faces)).astype(np.int64).ravel()
    mesh = pv.PolyData(verts, faces_pv)
    mesh.point_data["interaction"] = colors

    plotter = pv.Plotter(off_screen=not show_interactive)
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        scalars="interaction",
        cmap="twilight",
        opacity=1.0,
        smooth_shading=True,
        show_scalar_bar=True,
    )
    plotter.add_axes()
    plotter.add_text("Local model: $XY + YZ + ZX = 0$", color="black", font_size=10)

    show_kwargs = {}
    if screenshot_path:
        show_kwargs["screenshot"] = screenshot_path

    if show_kwargs or show_interactive:
        plotter.show(**show_kwargs)
    else:
        plotter.close()


def main(
    *,
    num_samples: int = 81,
    grid_limit: float = 2.0,
    e3_offset: float = 0.5,
    e4_offset: float = 0.5,
    epsilon: float = 0.01,
    show_interactive: bool = True,
    screenshot: bool = True,
):
    """Generate the 2D slice and 3D isosurface with the requested options."""
    x, y, z, d2, dot = compute_local_model_grid(
        num_samples=num_samples,
        grid_limit=grid_limit,
        e3_offset=e3_offset,
        e4_offset=e4_offset,
    )
    plot_z0_slice(
        x,
        y,
        z,
        d2,
        epsilon,
        output_path="sedenion_local_singularity_2d.png",
    )
    export_isosurface(
        x,
        y,
        z,
        d2,
        dot,
        epsilon,
        screenshot_path="sedenion_local_singularity_3d.png" if screenshot else None,
        show_interactive=show_interactive,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate zero-divisor local-model visualisations with interactive 3D output."
    )
    parser.add_argument("--num-samples", type=int, default=81, help="Grid size per axis.")
    parser.add_argument("--grid-limit", type=float, default=2.0, help="Range limit for X,Y,Z.")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Isosurface tightness.")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip opening the PyVista interactive window.",
    )
    parser.add_argument(
        "--no-screenshot",
        action="store_true",
        help="Skip writing the 3D screenshot (only 2D slice will be saved).",
    )
    args = parser.parse_args()

    main(
        num_samples=args.num_samples,
        grid_limit=args.grid_limit,
        epsilon=args.epsilon,
        show_interactive=not args.no_interactive,
        screenshot=not args.no_screenshot,
    )
