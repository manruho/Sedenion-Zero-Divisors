# Create projected zero-divisor density snapshots along the holonomy loop.

from __future__ import annotations

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from tqdm import tqdm

from common import sample_zero_divisor_pair


def project_to_3d_holonomy(a: np.ndarray, method: str):
    """Project (v1, v2) to (x, y, t) for the holonomy-style plots."""
    v1 = a[:8]
    v2 = a[8:]

    if method == "holonomy_probe":
        t = np.arctan2(v1[1], v1[0])
        labels = {
            "x": "$v_2$ component 0",
            "y": "$v_2$ component 1",
            "t": r"Angle $\theta$ on $S^7$ ($e_0$–$e_1$ plane)",
        }
    elif method == "holonomy_probe_e0e2":
        t = np.arctan2(v1[2], v1[0])
        labels = {
            "x": "$v_2$ component 0",
            "y": "$v_2$ component 1",
            "t": r"Angle $\phi$ on $S^7$ ($e_0$–$e_2$ plane)",
        }
    else:
        raise ValueError(f"Unknown projection method: {method}")

    x = v2[0]
    y = v2[1]
    time_range = (-np.pi, np.pi)
    return np.array([x, y, t]), labels, time_range


def generate_projected_zero_divisors_3d(
    n_points: int,
    method: str = "holonomy_probe",
    seed: int | None = 0,
):
    """Sample zero divisors, project them, and return (N,3) points."""
    rng = np.random.default_rng(seed)
    projected_points = np.zeros((n_points, 3))
    _, labels, time_range = project_to_3d_holonomy(np.zeros(16), method)

    for i in tqdm(range(n_points), desc=f"[density] {method}", unit="pts"):
        v1, v2 = sample_zero_divisor_pair(rng)
        a = np.concatenate([v1, v2])
        projected_points[i], _, _ = project_to_3d_holonomy(a, method)

    return projected_points, labels, time_range


def save_snapshot_at_time(
    points_3d: np.ndarray,
    axis_labels: dict[str, str],
    max_range: float,
    current_time: float,
    output_path: str,
    slice_thickness: float = 0.08,
    bins: int = 100,
):
    """Save a density snapshot for points with t in [t ± slice/2]."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    time_min = current_time - slice_thickness / 2
    time_max = current_time + slice_thickness / 2
    mask = (points_3d[:, 2] >= time_min) & (points_3d[:, 2] <= time_max)
    sliced_points = points_3d[mask]

    if len(sliced_points) > 0:
        _, _, _, im = ax.hist2d(
            sliced_points[:, 0],
            sliced_points[:, 1],
            bins=bins,
            range=[[-max_range, max_range], [-max_range, max_range]],
            cmap="viridis",
            norm=colors.LogNorm(),
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Point density (log scale)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        0.03,
        0.95,
        f"{axis_labels['t']} = {current_time:.2f}",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        verticalalignment="top",
        color="white",
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="black")
    plt.close(fig)


def generate_density_snapshots(
    num_points: int = 500_000,
    method: str = "holonomy_probe",
    slice_thickness: float = 0.08,
    bins: int = 120,
    snapshot_times: tuple[float, ...] = (-np.pi, 0.0, np.pi),
    output_prefix: str = "fig_zero_density",
    seed: int | None = 0,
):
    """Generate density snapshots at the requested times."""
    points_3d, labels, _ = generate_projected_zero_divisors_3d(
        num_points,
        method=method,
        seed=seed,
    )
    max_range = 1.1 * np.max(np.abs(points_3d[:, :2]))

    for suffix, current_time in zip(["start", "mid", "end"], snapshot_times):
        save_snapshot_at_time(
            points_3d,
            labels,
            max_range,
            current_time,
            f"{output_prefix}_{suffix}.png",
            slice_thickness=slice_thickness,
            bins=bins,
        )


if __name__ == "__main__":
    generate_density_snapshots()
