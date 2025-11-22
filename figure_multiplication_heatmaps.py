# Generate the norm annihilation and associator norm heatmaps used in the manuscript.

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sedenion_numeric import sedenion_multiply


def run_norm_annihilation_experiment(
    plane1_basis: tuple[int, int],
    plane2_basis: tuple[int, int],
    filename: str,
    resolution: int = 200,
):
    """Plot ||ab|| for a ∈ span(e_i, e_j) and b ∈ span(e_k, e_ℓ)."""
    theta_vals = np.linspace(0, 2 * np.pi, resolution)
    phi_vals = np.linspace(0, 2 * np.pi, resolution)
    norm_grid = np.zeros((resolution, resolution))
    e = np.eye(16)

    for i_theta, theta in enumerate(tqdm(theta_vals, desc="norm θ")):
        a = np.cos(theta) * e[plane1_basis[0]] + np.sin(theta) * e[plane1_basis[1]]
        for j_phi, phi in enumerate(phi_vals):
            b = np.cos(phi) * e[plane2_basis[0]] + np.sin(phi) * e[plane2_basis[1]]
            ab = sedenion_multiply(a, b)
            norm_grid[i_theta, j_phi] = np.linalg.norm(ab)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        norm_grid.T,
        origin="lower",
        extent=[0, 2 * np.pi, 0, 2 * np.pi],
        cmap="viridis",
        interpolation="bicubic",
    )
    cbar = fig.colorbar(im)
    cbar.set_label(r"$\|ab\|$", fontsize=12)

    ax.set_xlabel(
        rf"Angle $\theta$ in plane $(e_{{{plane1_basis[0]}}}, e_{{{plane1_basis[1]}}})$",
        fontsize=12,
    )
    ax.set_ylabel(
        rf"Angle $\phi$ in plane $(e_{{{plane2_basis[0]}}}, e_{{{plane2_basis[1]}}})$",
        fontsize=12,
    )
    ax.set_title("Sedenion norm annihilation pattern", fontsize=14)
    ax.set_aspect("equal")

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_associativity_experiment(
    plane1_basis: tuple[int, int],
    plane2_basis: tuple[int, int],
    c_vector: np.ndarray,
    filename: str,
    resolution: int = 200,
):
    """Plot ||(ab)c - a(bc)|| for selected planes and fixed c."""
    theta_vals = np.linspace(0, 2 * np.pi, resolution)
    phi_vals = np.linspace(0, 2 * np.pi, resolution)
    grid = np.zeros((resolution, resolution))
    e = np.eye(16)

    for i_theta, theta in enumerate(tqdm(theta_vals, desc="assoc θ")):
        a = np.cos(theta) * e[plane1_basis[0]] + np.sin(theta) * e[plane1_basis[1]]
        for j_phi, phi in enumerate(phi_vals):
            b = np.cos(phi) * e[plane2_basis[0]] + np.sin(phi) * e[plane2_basis[1]]

            ab = sedenion_multiply(a, b)
            left_assoc = sedenion_multiply(ab, c_vector)

            bc = sedenion_multiply(b, c_vector)
            right_assoc = sedenion_multiply(a, bc)

            grid[i_theta, j_phi] = np.linalg.norm(left_assoc - right_assoc)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        grid.T,
        origin="lower",
        extent=[0, 2 * np.pi, 0, 2 * np.pi],
        cmap="inferno",
        interpolation="bicubic",
    )
    cbar = fig.colorbar(im)
    cbar.set_label(r"$\|(ab)c - a(bc)\|$", fontsize=12)

    ax.set_xlabel(
        rf"Angle $\theta$ in plane $(e_{{{plane1_basis[0]}}}, e_{{{plane1_basis[1]}}})$",
        fontsize=12,
    )
    ax.set_ylabel(
        rf"Angle $\phi$ in plane $(e_{{{plane2_basis[0]}}}, e_{{{plane2_basis[1]}}})$",
        fontsize=12,
    )
    ax.set_title(
        rf"Sedenion associator norm (fixed c = $e_{{{int(np.argmax(c_vector))}}}$)",
        fontsize=14,
    )
    ax.set_aspect("equal")

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_all_heatmaps():
    """Generate the heatmaps used in the manuscript."""
    run_norm_annihilation_experiment(
        plane1_basis=(0, 8),
        plane2_basis=(1, 10),
        filename="fig4_norm_e0e8_vs_e1e10.png",
    )
    run_norm_annihilation_experiment(
        plane1_basis=(1, 10),
        plane2_basis=(4, 15),
        filename="fig4_norm_e1e10_vs_e4e15.png",
    )

    e = np.eye(16)
    run_associativity_experiment(
        plane1_basis=(1, 10),
        plane2_basis=(4, 15),
        c_vector=e[2],
        filename="fig5_assoc_c_e2.png",
    )
    run_associativity_experiment(
        plane1_basis=(1, 10),
        plane2_basis=(4, 15),
        c_vector=e[9],
        filename="fig5_assoc_c_e9.png",
    )


if __name__ == "__main__":
    run_all_heatmaps()
