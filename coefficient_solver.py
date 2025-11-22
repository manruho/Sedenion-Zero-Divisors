# Reconstruct Appendix A's coefficient system and verify it with SymPy.

from __future__ import annotations

import sympy as sp


def cross_product_7d(a: sp.Matrix, b: sp.Matrix) -> sp.Matrix:
    """Imaginary octonion cross product using the Fano plane conventions."""
    c = sp.zeros(7, 1)
    c[0] = a[1]*b[2] - a[2]*b[1] + a[3]*b[4] - a[4]*b[3] + a[6]*b[5] - a[5]*b[6]
    c[1] = a[2]*b[0] - a[0]*b[2] + a[3]*b[5] - a[5]*b[3] + a[4]*b[6] - a[6]*b[4]
    c[2] = a[0]*b[1] - a[1]*b[0] + a[4]*b[5] - a[5]*b[4] + a[6]*b[3] - a[3]*b[6]
    c[3] = a[4]*b[0] - a[0]*b[4] + a[5]*b[1] - a[1]*b[5] + a[6]*b[2] - a[2]*b[6]
    c[4] = a[0]*b[3] - a[3]*b[0] + a[2]*b[5] - a[5]*b[2] + a[6]*b[1] - a[1]*b[6]
    c[5] = a[6]*b[0] - a[0]*b[6] + a[1]*b[3] - a[3]*b[1] + a[2]*b[4] - a[4]*b[2]
    c[6] = a[0]*b[5] - a[5]*b[0] + a[1]*b[4] - a[4]*b[1] + a[2]*b[3] - a[3]*b[2]
    return c


def octonion_conjugate(x: sp.Matrix) -> sp.Matrix:
    """Return the conjugate (x0, -x1, ..., -x7)."""
    conj = x.copy()
    conj[1:] *= -1
    return conj


def octonion_multiply(a: sp.Matrix, b: sp.Matrix) -> sp.Matrix:
    """Cayley–Dickson product of two octonions."""
    a0, b0 = a[0], b[0]
    a_vec, b_vec = a[1:], b[1:]
    c0 = a0 * b0 - a_vec.dot(b_vec)
    c_vec = a0 * b_vec + b0 * a_vec + cross_product_7d(a_vec, b_vec)
    return sp.Matrix.vstack(sp.Matrix([c0]), c_vec)


def sedenion_multiply(s1: sp.Matrix, s2: sp.Matrix) -> sp.Matrix:
    """Cayley–Dickson product of two sedenions."""
    a, b = sp.Matrix(s1[:8]), sp.Matrix(s1[8:])
    c, d = sp.Matrix(s2[:8]), sp.Matrix(s2[8:])

    term1 = octonion_multiply(a, c) - octonion_multiply(octonion_conjugate(d), b)
    term2 = octonion_multiply(d, a) + octonion_multiply(b, octonion_conjugate(c))
    return sp.Matrix.vstack(term1, term2)


def left_multiplication_matrix(v: sp.Matrix) -> sp.Matrix:
    """Return the 16×16 real matrix of L_v."""
    basis = [sp.Matrix.eye(16)[:, i] for i in range(16)]
    columns = [sedenion_multiply(v, e) for e in basis]
    return sp.Matrix.hstack(*columns)


def invariants(v1: sp.Matrix, v2: sp.Matrix) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return (a, b, c) as defined in the paper."""
    norm1 = v1.dot(v1)
    norm2 = v2.dot(v2)
    inner = v1.dot(v2)
    a = norm1 + norm2
    b = norm1 - norm2
    c = inner
    return a, b, c


def octonion_basis(index: int) -> sp.Matrix:
    """Return e_index in the standard octonion basis."""
    vec = sp.zeros(8, 1)
    vec[index] = 1
    return vec


def sedenion_vector(v1: sp.Matrix, v2: sp.Matrix) -> sp.Matrix:
    """Embed (v1, v2) into S = O ⊕ O e8."""
    return sp.Matrix.vstack(v1, v2)


def build_cases():
    """Construct the six representative pairs listed in Appendix A."""
    e0, e1 = octonion_basis(0), octonion_basis(1)
    zero = sp.zeros(8, 1)
    return [
        (e0, zero),
        (2 * e0, zero),
        (e0, e1),
        (e0, 2 * e1),
        (e0 + e1, e0 - e1),
        (e0 + e1, 2 * e0),
    ]


def build_linear_system():
    """Return (A, delta_vec, abcs) for the six equations."""
    monomials = [
        lambda a, b, c: a**4,
        lambda a, b, c: a**2 * b**2,
        lambda a, b, c: a**2 * c**2,
        lambda a, b, c: b**4,
        lambda a, b, c: b**2 * c**2,
        lambda a, b, c: c**4,
    ]

    rows = []
    deltas = []
    abcs = []

    for idx, (v1, v2) in enumerate(build_cases(), start=1):
        v = sedenion_vector(v1, v2)
        mat = left_multiplication_matrix(v)
        delta = sp.factor(mat.det())
        a, b, c = invariants(v1, v2)

        row = [m(a, b, c) for m in monomials]
        rows.append(row)
        deltas.append(delta)
        abcs.append((a, b, c, delta))

        print(f"Case {idx}: (a, b, c) = ({a}, {b}, {c}), Δ = {delta}")

    A = sp.Matrix(rows)
    delta_vec = sp.Matrix(deltas)
    return A, delta_vec, abcs


def main():
    A, delta_vec, _ = build_linear_system()
    detA = sp.factor(A.det())
    print("\nCoefficient matrix A:")
    sp.pprint(A)
    print(f"\nDeterminant det(A) = {detA}")

    solution = A.LUsolve(delta_vec)
    coeff_names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    print("\nSolved coefficients:")
    for name, value in zip(coeff_names, solution):
        print(f"  {name} = {sp.simplify(value)}")

    alpha, beta, gamma, delta, epsilon, zeta = solution
    a, b, c = sp.symbols("a b c")
    G = (
        alpha * a**4
        + beta * a**2 * b**2
        + gamma * a**2 * c**2
        + delta * b**4
        + epsilon * b**2 * c**2
        + zeta * c**4
    )
    simplified = sp.factor(G)
    print(f"\nResulting polynomial G(a,b,c) = {simplified}")


if __name__ == "__main__":
    main()
