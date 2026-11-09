"""Analytic regressions for spherical BSSN outer boundaries and ghost masks.

Run: python -m unittest discover -s tests -p test_bssn_boundaries.py -v
No evolution or bssnutils import is needed. BSSN_SOURCE_DIR optionally selects
an archived source tree for demonstrating the pre-fix failures.
"""
import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(os.environ.get('BSSN_SOURCE_DIR', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bssn'))
from bssneqs import BSSN
from nwave import Grid1D, l2norm_mask


class SommerfeldTests(unittest.TestCase):
    def test_outgoing_radial_field(self):
        """u-u_inf=F(|r|-v*t)/|r|^n obeys the same outgoing law at both ends."""
        r = np.r_[np.linspace(-12., -10., 4), np.linspace(10., 12., 4)]
        rho = np.abs(r)
        ng = 2
        boundary = np.r_[np.arange(ng), np.arange(len(r)-ng, len(r))]
        for v in (1., np.sqrt(2.)):
            for n in (0, 1, 2, -2):
                with self.subTest(v=v, falloff=n):
                    # Preserve the existing per-variable asymptote convention.
                    u_inf = r[-1] ** abs(n) if n < 0 else 0.7
                    z = 0.31 * (rho - v * 0.2)
                    f, df = np.sin(z), 0.31 * np.cos(z)
                    u = u_inf + f / rho**n
                    du = np.sign(r) * (df / rho**n - n*f/rho**(n+1))
                    expected = -v * df / rho**n
                    rhs = np.full_like(r, 123.)
                    BSSN.bc_sommerfeld(rhs, u, du, r, n, u_inf, v, ng, True)
                    np.testing.assert_allclose(rhs[boundary], expected[boundary], rtol=1e-12, atol=1e-12)
                    np.testing.assert_array_equal(rhs[ng:-ng], 123.)

    def test_positive_domain_leaves_left_boundary_untouched(self):
        r = np.linspace(10., 12., 8)
        rhs = np.full_like(r, 123.)
        u = 1. + 1. / r
        du = -1. / r**2
        BSSN.bc_sommerfeld(rhs, u, du, r, 1, 1., 1., 2, False)
        np.testing.assert_array_equal(rhs[:-2], 123.)
        np.testing.assert_allclose(rhs[-2:], 0., atol=1e-15)

    def test_lapse_characteristic_speed_is_preserved(self):
        g = Grid1D({'Nx': 21, 'Xmin': -10., 'Xmax': 10., 'NGhost': 3}, cell_centered=True)
        eq = BSSN(g, 1., 2., True)
        self.assertEqual(eq.u_wavespeed[eq.U_ALPHA], np.sqrt(2.))
        np.testing.assert_array_equal(np.delete(eq.u_wavespeed, eq.U_ALPHA), 1.)


class ConstraintGhostTests(unittest.TestCase):
    def test_all_ghost_values_zero_and_interior_unchanged(self):
        # Deliberately nonzero raw H=7/6, M=1 and C=1, including endpoints.
        r = np.r_[np.linspace(-4., -1., 7), np.linspace(1., 4., 7)]
        one, zero = np.ones_like(r), np.zeros_like(r)
        for ng in (2, 3, 4):
            with self.subTest(nghost=ng):
                out = [np.full_like(r, np.nan) for _ in range(3)]
                BSSN.get_constraints(*out, one, one, one, one, one, one,
                                     zero, zero, zero, one, zero, zero, zero, r, ng)
                mask = np.ones_like(r)
                mask[:ng] = mask[-ng:] = 0.
                for values, expected in zip(out, (7./6., 1., 1.)):
                    np.testing.assert_array_equal(values[:ng], 0.)
                    np.testing.assert_array_equal(values[-ng:], 0.)
                    np.testing.assert_allclose(values[ng:-ng], expected)
                    # Excluding ghosts must retain precisely the physical norm.
                    self.assertAlmostEqual(l2norm_mask(values, 0.1, mask),
                                           abs(expected)*np.sqrt(0.1*(len(r)-2*ng)))


if __name__ == '__main__':
    unittest.main()
