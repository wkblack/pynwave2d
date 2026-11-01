"""
Boundary closures of the sixth-order compact stencils: D2_JT6 and JP6_JT4.

Run from the repository root:  python -m pytest tests/test_compact_closures.py
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "bssn")]

import bssnutils
from nwave.cd import NCompactDerivative
from nwave.types import CFDSolve, DerivType


def operator(x, order, name, method=CFDSolve.LUSOLVE, masked=False):
    dtype = getattr(DerivType, f"D{order}_{name}")
    if masked:
        inner = getattr(DerivType, f"D{order}_ME642")
        return NCompactDerivative.bh_deriv(x, dtype, inner, method, 0.0, 0.05)
    return NCompactDerivative.deriv(x, dtype, method)


def matrix(x, order, name):
    d = operator(x, order, name, method=CFDSolve.SCIPY)
    return np.column_stack([d.grad(e) for e in np.eye(len(x))])


def exact_power(x, degree, order):
    """d^order/dx^order of x**degree."""
    if degree < order:
        return np.zeros_like(x)
    c = degree * (degree - 1) if order == 2 else degree
    return c * x ** (degree - order)


class ClosureTests(unittest.TestCase):
    # (name, derivative order, exact through degree, edge order, atol)
    CASES = [("JT6", 2, 7, 6, 1e-7),
             ("JP6_JT4", 1, 4, 4, 1e-10), ("JP6_JT4", 2, 5, 4, 1e-7)]

    def test_exact_on_polynomials_at_every_point(self):
        x = np.linspace(-1.0, 1.0, 81)
        for name, order, top, _, atol in self.CASES:
            for method in (CFDSolve.LUSOLVE, CFDSolve.SCIPY):
                for masked in (False, True):
                    d = operator(x, order, name, method, masked)
                    for degree in range(top + 1):
                        with self.subTest(name=name, order=order, method=method,
                                          masked=masked, degree=degree):
                            np.testing.assert_allclose(
                                d.grad(x**degree), exact_power(x, degree, order),
                                rtol=0, atol=atol)

    def test_convergence_order_at_the_edge_and_in_the_interior(self):
        for name, order, _, edge_order, _ in self.CASES:
            edge, inner = [], []
            for n in (81, 161, 321):
                x = np.linspace(-1.0, 1.0, n)
                f = np.sin(11 * x + 0.2)
                exact = (11 * np.cos(11 * x + 0.2) if order == 1
                         else -121 * np.sin(11 * x + 0.2))
                e = np.abs(operator(x, order, name).grad(f) - exact)
                edge.append(max(e[:4].max(), e[-4:].max()))
                inner.append(e[np.abs(x) < 0.5].max())
            for errors, rate in ((edge, edge_order - 0.5), (inner, 5.5)):
                with self.subTest(name=name, order=order, rate=rate):
                    rates = np.log2(np.divide(errors[:-1], errors[1:]))
                    self.assertTrue(np.all(rates > rate), rates)

    def test_original_jp6_fails_where_jp6_jt4_is_exact(self):
        x = np.linspace(-1.0, 1.0, 81)
        for order in (1, 2):
            f, exact = x**order, exact_power(x, order, order)
            jp6 = np.abs(operator(x, order, "JP6").grad(f) - exact).max()
            fixed = np.abs(operator(x, order, "JP6_JT4").grad(f) - exact).max()
            self.assertGreater(jp6, 1e-3)
            self.assertLess(fixed, 1e-8)

    def test_reflection_parity(self):
        x = np.linspace(-1.0, 1.0, 129)
        f = np.exp(0.3 * x) + np.sin(4 * x + 0.2)
        for name, order, *_ in self.CASES:
            d = operator(x, order, name)
            np.testing.assert_allclose(d.grad(f[::-1]), (-1) ** order * d.grad(f)[::-1],
                                       rtol=2e-10, atol=2e-10)

    def test_radial_wave_with_sommerfeld_boundary_is_stable(self):
        # u_tt = u_rr + 2 u_r / r on r in [10, 20], with the solver's
        # Sommerfeld condition on the outer three points; the inner three
        # points are held fixed.  RK4 at CFL 0.1 must not amplify any mode.
        for d1, d2 in (("JT6", "JT6"), ("JP6_JT4", "JP6_JT4")):
            for n in (64, 128):
                h = 10 / (n - 7)
                r = 10 + (np.arange(n) - 3) * h
                D, DD = matrix(r, 1, d1), matrix(r, 2, d2)
                eye = np.eye(n)
                A = np.block([[np.zeros((n, n)), eye],
                              [DD + 2 / r[:, None] * D, np.zeros((n, n))]])
                S = -D - np.diag(1 / r)
                for field in (0, 1):
                    for i in range(n - 3, n):
                        A[field * n + i] = 0.0
                        A[field * n + i, field * n:(field + 1) * n] = S[i]
                keep = np.r_[np.arange(3, n), n + np.arange(3, n)]
                eig = np.linalg.eigvals(A[np.ix_(keep, keep)])
                with self.subTest(d1=d1, d2=d2, n=n):
                    self.assertLessEqual(eig.real.max(), 1e-10)
                    z = 0.1 * h * eig
                    self.assertLessEqual(
                        np.abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24).max(), 1 + 1e-10)

    def test_jp6_jt4_dirichlet_wave_is_stable(self):
        # u_tt = u_xx with u = 0 at both ends.  JT6's second-derivative
        # closure does not pass this test; JP6_JT4 does.
        for n in (32, 64):
            x = np.linspace(0.0, 1.0, n)
            DD = matrix(x, 2, "JP6_JT4")[1:-1, 1:-1]
            m = n - 2
            wave = np.block([[np.zeros((m, m)), np.eye(m)], [DD, np.zeros((m, m))]])
            self.assertLessEqual(np.linalg.eigvals(wave).real.max(), 1e-7)

    def test_parameter_names_select_the_operators(self):
        self.assertIs(bssnutils.get_d1_type("JP6"), DerivType.D1_JP6)
        self.assertIs(bssnutils.get_d1_type("JP6_JT4"), DerivType.D1_JP6_JT4)
        self.assertIs(bssnutils.get_d2_type("JP6_JT4"), DerivType.D2_JP6_JT4)
        self.assertIs(bssnutils.get_d2_type("JT6"), DerivType.D2_JT6)
        x = np.linspace(-2.0, 2.0, 97)
        for name in ("JT6", "JP6_JT4"):
            params = {"D1": [name, "ME642"], "D2": [name, "ME642"],
                      "BHMask": True, "BHMaskPos": 0.0, "BHMaskWidth": 0.05}
            D, DD = bssnutils.init_derivative_operators(x, params)
            np.testing.assert_allclose(D.grad(x), 1.0, rtol=0, atol=1e-10)
            np.testing.assert_allclose(DD.grad(x * x), 2.0, rtol=0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
