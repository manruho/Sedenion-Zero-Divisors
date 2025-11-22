# Generate holonomy rotation diagnostics and comparison plots for the paper.

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

from common import normalize, random_unit_vector, sample_zero_divisor_pair


def project_to_3d_simple(a: np.ndarray):
    """Map (v1, v2) to (x, y, t) using the e0–e1 loop."""
    v1 = a[:8]
    v2 = a[8:]
    t = np.arctan2(v1[1], v1[0])
    x = v2[0]
    y = v2[1]
    labels = {
        "x": "$v_2$ component 0",
        "y": "$v_2$ component 1",
        "t": r"Angle $\theta$ on $S^7$",
    }
    time_range = (-np.pi, np.pi)
    return np.array([x, y, t]), labels, time_range


def generate_data_zero_vs_random(
    n_points: int,
    scenario: str,
    seed: int | None = 0,
):
    """Generate projected samples for the zero-divisor vs random comparison."""
    rng = np.random.default_rng(seed)
    projected_points = np.zeros((n_points, 3))
    _, labels, time_range = project_to_3d_simple(np.zeros(16))

    for i in tqdm(range(n_points), desc=f"[holonomy] {scenario}", unit="pts"):
        if scenario == "zero_divisor":
            v1, v2 = sample_zero_divisor_pair(rng)
        elif scenario == "random_vector":
            v1 = random_unit_vector(8, rng)
            v2 = random_unit_vector(8, rng)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        a = np.concatenate([v1, v2])
        projected_points[i], _, _ = project_to_3d_simple(a)

    return projected_points, labels, time_range


def analyze_rotation_robust(
    points_3d: np.ndarray,
    time_range: tuple[float, float],
    slice_thickness: float = 0.08,
    num_steps: int = 100,
):
    """Run a PCA-based orientation estimator for each time slice."""
    times = np.linspace(time_range[0], time_range[1], num_steps)
    corrected_vectors = []
    previous_vector = None

    for t in times:
        time_min = t - slice_thickness / 2
        time_max = t + slice_thickness / 2
        mask = (points_3d[:, 2] >= time_min) & (points_3d[:, 2] <= time_max)
        sliced_points = points_3d[mask]

        if len(sliced_points) < 2:
            corrected_vectors.append([np.nan, np.nan])
            continue

        pca = PCA(n_components=1)
        pca.fit(sliced_points[:, :2])
        current_vector = pca.components_[0]

        if previous_vector is not None and np.dot(current_vector, previous_vector) < 0:
            current_vector *= -1.0

        corrected_vectors.append(current_vector)
        previous_vector = current_vector

    corrected_vectors = np.array(corrected_vectors)
    angles = np.arctan2(corrected_vectors[:, 1], corrected_vectors[:, 0])
    return times, angles


def plot_angle_graph(
    times: np.ndarray,
    angles: np.ndarray,
    labels: dict[str, str],
    output_filename: str,
    scenario_title: str,
    ylim: tuple[float, float] = (0, 9),
):
    """Plot unwrapped angles against the base loop parameter."""
    unwrapped = np.unwrap(angles)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(times, unwrapped, "o-", label="Observed angle (PCA-corrected)")

    valid = ~np.isnan(unwrapped)
    slope = np.nan
    if np.sum(valid) > 1:
        m, c = np.polyfit(times[valid], unwrapped[valid], 1)
        slope = m
        ax.plot(times, m * times + c, "r--", label=f"Linear fit (slope = {m:.2f})")

    ax.set_xlabel(f"Base loop parameter t [rad] ({labels['t']})")
    ax.set_ylabel("Fiber angle θ(t) [rad]")
    ax.set_title(scenario_title)
    ax.grid(True)
    if not np.isnan(slope):
        ax.legend()
    ax.set_ylim(*ylim)

    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_zero_vs_random(
    num_points: int = 500_000,
    slice_thickness: float = 0.08,
    num_steps: int = 100,
    seed: int | None = 0,
):
    """Run the zero-divisor vs random holonomy benchmark."""
    for scenario, prefix in [
        ("zero_divisor", "fig2_zero_divisor"),
        ("random_vector", "fig2_random_vector"),
    ]:
        pts, labels, time_range = generate_data_zero_vs_random(
            num_points,
            scenario,
            seed=seed,
        )
        times, angles = analyze_rotation_robust(
            pts,
            time_range,
            slice_thickness=slice_thickness,
            num_steps=num_steps,
        )
        title = (
            r"Zero Divisors ($v_2 \perp v_1$)"
            if scenario == "zero_divisor"
            else r"Random comparison ($v_2$ independent)"
        )
        plot_angle_graph(
            times,
            angles,
            labels,
            output_filename=f"{prefix}_rotation_graph.png",
            scenario_title=title,
        )


def generate_data_v2r8_v2r7(
    n_points: int,
    scenario: str,
    seed: int | None = 0,
):
    """Sample from V2(R^8) or V2(R^7) and project to (x, y, t)."""
    if scenario == "v2r8":
        dim = 8
    elif scenario == "v2r7":
        dim = 7
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    rng = np.random.default_rng(seed)
    projected_points = np.zeros((n_points, 3))
    for i in tqdm(range(n_points), desc=f"[holonomy] {scenario}", unit="pts"):
        v1 = random_unit_vector(dim, rng)
        v2_candidate = rng.standard_normal(dim)
        v2 = normalize(v2_candidate - np.dot(v1, v2_candidate) * v1)

        t = np.arctan2(v1[1], v1[0])
        x = v2[0]
        y = v2[1]
        projected_points[i] = [x, y, t]

    labels = {
        "x": "$v_2$ projection X",
        "y": "$v_2$ projection Y",
        "t": "Loop angle",
    }
    time_range = (-np.pi, np.pi)
    return projected_points, labels, time_range


def figure_v2_comparison(
    num_points: int = 500_000,
    slice_thickness: float = 0.08,
    num_steps: int = 100,
    seed: int | None = 0,
    output_filename: str = "fig3_v2r8_vs_v2r7_holonomy.png",
):
    """Plot the holonomy law for V2(R^8) and V2(R^7) on the same axes."""
    scenarios = [
        ("v2r8", r"$V_2(\mathbb{R}^8)$"),
        ("v2r7", r"$V_2(\mathbb{R}^7)$"),
    ]
    results = {}

    for scenario, _ in scenarios:
        pts, _, time_range = generate_data_v2r8_v2r7(
            num_points,
            scenario,
            seed=seed,
        )
        results[scenario] = analyze_rotation_robust(
            pts,
            time_range,
            slice_thickness=slice_thickness,
            num_steps=num_steps,
        )

    fig, ax = plt.subplots(figsize=(12, 7))
    color_map = {"v2r8": "tab:blue", "v2r7": "tab:green"}

    for scenario, label in scenarios:
        times, angles = results[scenario]
        unwrapped = np.unwrap(angles)
        valid = ~np.isnan(unwrapped)
        slope = np.nan
        if np.sum(valid) > 1:
            m, _ = np.polyfit(times[valid], unwrapped[valid], 1)
            slope = m
        ax.plot(
            times,
            unwrapped,
            "o-",
            color=color_map[scenario],
            label=f"{label}, slope ≈ {slope:.2f}",
            markersize=4,
        )

    ax.set_xlabel("Base loop angle t [rad]")
    ax.set_ylabel("Fiber angle θ(t) [rad]")
    ax.set_title("Holonomy comparison: $V_2(\\mathbb{R}^8)$ vs $V_2(\\mathbb{R}^7)$")
    ax.grid(True)
    ax.legend()

    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure_zero_vs_random()
    figure_v2_comparison()
