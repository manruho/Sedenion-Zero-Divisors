# Numeric Cayley–Dickson helpers used by the heatmap scripts.

from __future__ import annotations

import numpy as np


def cross_product_7d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Imaginary octonion cross product."""
    c = np.zeros(7)
    c[0] = a[1]*b[2] - a[2]*b[1] + a[3]*b[4] - a[4]*b[3] + a[6]*b[5] - a[5]*b[6]
    c[1] = a[2]*b[0] - a[0]*b[2] + a[3]*b[5] - a[5]*b[3] + a[4]*b[6] - a[6]*b[4]
    c[2] = a[0]*b[1] - a[1]*b[0] + a[4]*b[5] - a[5]*b[4] + a[6]*b[3] - a[3]*b[6]
    c[3] = a[4]*b[0] - a[0]*b[4] + a[5]*b[1] - a[1]*b[5] + a[6]*b[2] - a[2]*b[6]
    c[4] = a[0]*b[3] - a[3]*b[0] + a[2]*b[5] - a[5]*b[2] + a[6]*b[1] - a[1]*b[6]
    c[5] = a[6]*b[0] - a[0]*b[6] + a[1]*b[3] - a[3]*b[1] + a[2]*b[4] - a[4]*b[2]
    c[6] = a[0]*b[5] - a[5]*b[0] + a[1]*b[4] - a[4]*b[1] + a[2]*b[3] - a[3]*b[2]
    return c


def octonion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the Cayley–Dickson product of two octonions."""
    a0, a_vec = a[0], a[1:]
    b0, b_vec = b[0], b[1:]
    c0 = a0 * b0 - np.dot(a_vec, b_vec)
    c_vec = a0 * b_vec + b0 * a_vec + cross_product_7d(a_vec, b_vec)
    return np.concatenate(([c0], c_vec))


def sedenion_multiply(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Return the Cayley–Dickson product of two sedenions."""
    a, b = s1[:8], s1[8:]
    c, d = s2[:8], s2[8:]

    c_conj = c.copy()
    d_conj = d.copy()
    c_conj[1:] *= -1
    d_conj[1:] *= -1

    term1 = octonion_multiply(a, c) - octonion_multiply(d_conj, b)
    term2 = octonion_multiply(d, a) + octonion_multiply(b, c_conj)
    return np.concatenate([term1, term2])
