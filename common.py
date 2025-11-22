# Shared utilities for the zero-divisor research scripts.

from __future__ import annotations

import numpy as np


def _require_rng(seed_or_rng: int | None | np.random.Generator) -> np.random.Generator:
    """Return a numpy Generator from a seed, existing generator, or None."""
    if isinstance(seed_or_rng, np.random.Generator):
        return seed_or_rng
    return np.random.default_rng(seed_or_rng)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return v / ||v|| and raise if the norm is too small."""
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero (or near-zero) vector cannot be normalized.")
    return v / n


def random_unit_vector(
    dim: int = 8,
    rng: int | None | np.random.Generator = None,
) -> np.ndarray:
    """Sample a unit vector in R^dim using the provided RNG (or a new one)."""
    generator = _require_rng(rng)
    return normalize(generator.standard_normal(dim))


def sample_zero_divisor_pair(
    rng: int | None | np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (v1, v2) with ||v1||=||v2||=1 and v1 ⟂ v2."""
    generator = _require_rng(rng)
    v1 = random_unit_vector(8, generator)
    v2_candidate = generator.standard_normal(8)
    v2 = v2_candidate - np.dot(v1, v2_candidate) * v1
    return v1, normalize(v2)


def base_loop_on_s7(t_array: np.ndarray) -> np.ndarray:
    """Parameterise the standard e0–e1 circle on S^7."""
    v1_path = np.zeros((len(t_array), 8))
    v1_path[:, 0] = np.cos(t_array)
    v1_path[:, 1] = np.sin(t_array)
    return v1_path


def project_to_fiber(v: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Project v to the fiber orthogonal to base and renormalise."""
    return normalize(v - np.dot(v, base) * base)


def parallel_transport_along_loop(
    v1_path: np.ndarray,
    v2_initial: np.ndarray,
) -> np.ndarray:
    """Discrete Levi–Civita transport used in the holonomy probe."""
    num_steps, _ = v1_path.shape
    v2_path = np.zeros_like(v1_path)

    v2 = project_to_fiber(v2_initial, v1_path[0])
    v2_path[0] = v2

    for k in range(1, num_steps):
        v2 = project_to_fiber(v2, v1_path[k])
        v2_path[k] = v2

    return v2_path
