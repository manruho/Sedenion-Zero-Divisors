# Produce the torus and angle-angle holonomy plots referenced in the paper.

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from common import (
    base_loop_on_s7,
    parallel_transport_along_loop,
    project_to_fiber,
    random_unit_vector,
)


def holonomy_probe(
    num_steps: int = 400,
    t_start: float = -np.pi,
    t_end: float = np.pi,
    fiber_plane: tuple[int, int] = (0, 1),
    seed: int | None = 42,
):
    """Return (t_array, phi_array, v1_path, v2_path) for the holonomy loop."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    t_array = np.linspace(t_start, t_end, num_steps)
    v1_path = base_loop_on_s7(t_array)

    v2_initial = random_unit_vector(8, rng)
    v2_initial = project_to_fiber(v2_initial, v1_path[0])
    v2_path = parallel_transport_along_loop(v1_path, v2_initial)

    i, j = fiber_plane
    x = v2_path[:, i]
    y = v2_path[:, j]
    phi_array = np.arctan2(y, x)

    return t_array, phi_array, v1_path, v2_path


def plot_torus_curve(
    t_array: np.ndarray,
    phi_array: np.ndarray,
    savepath: str = "fig_torus_holonomy.png",
):
    """Plot the holonomy curve on S^1 × S^1."""
    theta = np.mod(t_array, 2 * np.pi)
    phi = np.mod(phi_array, 2 * np.pi)

    for i in range(len(theta) - 1):
        if abs(phi[i + 1] - phi[i]) > np.pi:
            theta[i + 1] = np.nan
            phi[i + 1] = np.nan

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(theta, phi, linewidth=1.5)

    ax.set_xlabel(r"Base angle $\theta$ on $S^1_{\mathrm{base}}$ [rad]")
    ax.set_ylabel(r"Fiber angle $\phi$ on $S^1_{\mathrm{fiber}}$ [rad]")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    labels = [r"0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_title(r"Holonomy curve on $S^1_{\mathrm{base}}\times S^1_{\mathrm{fiber}}$")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close(fig)


def plot_angle_vs_angle(
    t_array: np.ndarray,
    phi_array: np.ndarray,
    savepath: str = "fig_angle_vs_angle.png",
):
    """Plot unwrapped φ(t) against t and overlay the best linear fit."""
    t_unwrap = np.unwrap(t_array)
    phi_unwrap = np.unwrap(phi_array)

    A = np.vstack([t_unwrap, np.ones_like(t_unwrap)]).T
    slope, intercept = np.linalg.lstsq(A, phi_unwrap, rcond=None)[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_unwrap, phi_unwrap, ".", markersize=2, label="Observed angle")
    ax.plot(
        t_unwrap,
        slope * t_unwrap + intercept,
        "-",
        label=f"Linear fit (slope = {slope:.3f})",
    )
    ax.set_xlabel(r"Base angle $\theta$ [rad]")
    ax.set_ylabel(r"Fiber angle $\phi$ [rad]")
    ax.set_title("Base vs fiber angle (holonomy probe)")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close(fig)


def main():
    t_array, phi_array, _, _ = holonomy_probe()
    plot_torus_curve(t_array, phi_array, savepath="fig_torus_holonomy.png")
    plot_angle_vs_angle(t_array, phi_array, savepath="fig_angle_vs_angle.png")


if __name__ == "__main__":
    main()
