"""
JP6_JT4: the JP6 interior with JT4 boundary rows.

Run from the repository root:  python -m pytest tests/test_jp6_jt4.py
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


def operator(x, order, name="JP6_JT4", method=CFDSolve.LUSOLVE, masked=False):
    dtype = getattr(DerivType, f"D{order}_{name}")
    if masked:
        inner = getattr(DerivType, f"D{order}_ME642")
        return NCompactDerivative.bh_deriv(x, dtype, inner, method, 0.0, 0.05)
    return NCompactDerivative.deriv(x, dtype, method)


def exact_power(x, degree, order):
    """d^order/dx^order of x**degree."""
    c = degree * (degree - 1) if order == 2 else degree
    return c * x ** (degree - order) if degree >= order else np.zeros_like(x)


class JP6JT4Tests(unittest.TestCase):
    def test_exact_on_low_degree_polynomials_at_every_point(self):
        # JT4 boundary rows are exact through x^4 (D1) and x^5 (D2).
        x = np.linspace(-1.0, 1.0, 81)
        for order, top, atol in ((1, 4, 1e-10), (2, 5, 1e-7)):
            for method in (CFDSolve.LUSOLVE, CFDSolve.SCIPY):
                for masked in (False, True):
                    d = operator(x, order, method=method, masked=masked)
                    for degree in range(top + 1):
                        with self.subTest(order=order, method=method,
                                          masked=masked, degree=degree):
                            np.testing.assert_allclose(
                                d.grad(x**degree), exact_power(x, degree, order),
                                rtol=0, atol=atol)

    def test_original_jp6_fails_where_the_hybrid_is_exact(self):
        x = np.linspace(-1.0, 1.0, 81)
        for order in (1, 2):
            f, exact = x**order, exact_power(x, order, order)
            jp6 = np.abs(operator(x, order, "JP6").grad(f) - exact).max()
            hybrid = np.abs(operator(x, order).grad(f) - exact).max()
            self.assertGreater(jp6, 1e-3)
            self.assertLess(hybrid, 1e-8)

    def test_fourth_order_at_the_edge_sixth_in_the_interior(self):
        for order in (1, 2):
            edge, inner = [], []
            for n in (81, 161, 321):
                x = np.linspace(-1.0, 1.0, n)
                f = np.sin(11 * x + 0.2)
                exact = (11 * np.cos(11 * x + 0.2) if order == 1
                         else -121 * np.sin(11 * x + 0.2))
                e = np.abs(operator(x, order).grad(f) - exact)
                edge.append(max(e[:4].max(), e[-4:].max()))
                inner.append(e[np.abs(x) < 0.5].max())
            for errors, rate in ((edge, 3.5), (inner, 5.5)):
                with self.subTest(order=order, rate=rate):
                    self.assertTrue(np.all(np.log2(np.divide(errors[:-1], errors[1:])) > rate),
                                    errors)

    def test_reflection_parity(self):
        x = np.linspace(-1.0, 1.0, 129)
        f = np.exp(0.3 * x) + np.sin(4 * x + 0.2)
        for order in (1, 2):
            d = operator(x, order)
            np.testing.assert_allclose(d.grad(f[::-1]), (-1) ** order * d.grad(f)[::-1],
                                       rtol=2e-10, atol=2e-10)

    def test_dirichlet_wave_is_stable_under_rk4(self):
        # u_tt = u_xx with u = 0 at both ends; the closure must have no growing modes.
        for n in (32, 64):
            x = np.linspace(0.0, 1.0, n)
            d2 = operator(x, 2, method=CFDSolve.SCIPY)
            DD = np.column_stack([d2.grad(e) for e in np.eye(n)])[1:-1, 1:-1]
            m = n - 2
            wave = np.block([[np.zeros((m, m)), np.eye(m)], [DD, np.zeros((m, m))]])
            eig = np.linalg.eigvals(wave)
            self.assertLessEqual(eig.real.max(), 1e-7)
            z = 0.1 * (x[1] - x[0]) * eig
            self.assertLessEqual(np.abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24).max(),
                                 1 + 1e-10)

    def test_parameter_names_select_the_operators(self):
        self.assertIs(bssnutils.get_d1_type("JP6"), DerivType.D1_JP6)
        self.assertIs(bssnutils.get_d1_type("JP6_JT4"), DerivType.D1_JP6_JT4)
        self.assertIs(bssnutils.get_d2_type("JP6_JT4"), DerivType.D2_JP6_JT4)
        x = np.linspace(-2.0, 2.0, 97)
        params = {"D1": ["JP6_JT4", "ME642"], "D2": ["JP6_JT4", "ME642"],
                  "BHMask": True, "BHMaskPos": 0.0, "BHMaskWidth": 0.05}
        D, DD = bssnutils.init_derivative_operators(x, params)
        np.testing.assert_allclose(D.grad(x), 1.0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(DD.grad(x * x), 2.0, rtol=0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
