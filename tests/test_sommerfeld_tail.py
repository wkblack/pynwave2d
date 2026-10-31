"""Tests for the NewRad-style Sommerfeld tail (bssn/sommerfeld_tail.py).

Run: python -m unittest discover -s tests -p test_sommerfeld_tail.py -v
"""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'bssn'), str(ROOT)]
from bssneqs import BSSN, BSSNOuterBCType
import gauges
import sommerfeld_tail as tail
from nwave import Grid1D, ExplicitFirst642_1D, ExplicitSecond642_1D, FilterApply, RK4


class TailTests(unittest.TestCase):
    def grid(self):
        g = Grid1D({'Nx': 101, 'Xmin': -20., 'Xmax': 20., 'NGhost': 3},
                   cell_centered=True)
        g.set_D1(ExplicitFirst642_1D(g.dx[0]))
        g.set_D2(ExplicitSecond642_1D(g.dx[0]))
        r = g.xi[0]
        n = len(r) // 2
        r[:n] = -r[-1:-n-1:-1]
        return g

    def initialized(self, g, enabled):
        e = BSSN(g, 1., 2., True, sommerfeld_tail=enabled)
        e.initialize(g, {'initial_data': 'Puncture', 'collapsed_lapse': True})
        return e

    def test_telegrapher_auxiliary_uses_same_tail_and_preserves_bulk(self):
        """The telegrapher field gets the tail; interior RHS is untouched."""
        g = self.grid()
        r = g.xi[0]
        base = BSSN(g, 1., 2., True, lapse_gauge=gauges.Telegrapher(c=1.7),
                    sommerfeld_tail=False)
        enabled = BSSN(g, 1., 2., True, lapse_gauge=gauges.Telegrapher(c=1.7),
                       sommerfeld_tail=True)
        for eq in (base, enabled):
            eq.initialize(g, {'initial_data': 'Puncture', 'collapsed_lapse': True})
        u = [x.copy() for x in base.u]
        u[-1][:] = .001 * np.cos(.13 * r)
        reference = [np.zeros_like(x) for x in u]
        actual = [np.zeros_like(x) for x in u]
        base.rhs(reference, u, g)
        enabled.rhs(actual, u, g)
        self.assertEqual(len(actual), 10)
        for a, b in zip(actual, reference):
            np.testing.assert_array_equal(a[3:-3], b[3:-3])
        dx = g.D1.grad(u[-1])
        S = -1.7 * (np.sign(r) * dx + 2 * u[-1] / abs(r))
        sides = ((np.arange(3), 3), (np.arange(len(r) - 3, len(r)), len(r) - 4))
        for boundary, anchor in sides:
            w = (abs(r[anchor]) / abs(r[boundary]))**3
            expected = S[boundary] + w * (reference[-1][anchor] - S[anchor])
            np.testing.assert_allclose(actual[-1][boundary], expected, rtol=2e-14, atol=1e-16)

    def test_default_on_matches_explicit_true(self):
        """Omitting the option is the same as sommerfeld_tail=True."""
        g = self.grid()
        default = BSSN(g, 1., 2., True)
        enabled = self.initialized(g, True)
        self.assertTrue(default.sommerfeld_tail)
        actual = [np.zeros_like(x) for x in enabled.u]
        expected = [np.zeros_like(x) for x in enabled.u]
        default.rhs(actual, enabled.u, g)
        enabled.rhs(expected, enabled.u, g)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(self.initialized(g, False).sommerfeld_tail)

    def test_sponge_does_not_initialize_or_apply_tail(self):
        """An unused default tail must not reject a valid zero-ghost sponge."""
        for ng in (0, 3):
            g = Grid1D({'Nx': 81, 'Xmin': -20., 'Xmax': 20., 'NGhost': ng},
                       cell_centered=True)
            g.set_D1(ExplicitFirst642_1D(g.dx[0]))
            g.set_D2(ExplicitSecond642_1D(g.dx[0]))
            states, initial_rhs = [], []
            for option in ({}, {'sommerfeld_tail': True}, {'sommerfeld_tail': False}):
                eq = BSSN(g, 1., 2., True, outer_bc=BSSNOuterBCType.SPONGE, **option)
                self.assertIsNone(eq._tail_geometry)
                eq.initialize(g, {'initial_data': 'Puncture', 'collapsed_lapse': True})
                rhs = [np.zeros_like(x) for x in eq.u]
                eq.rhs(rhs, eq.u, g)
                initial_rhs.append(np.asarray(rhs))
                rk = RK4(eq, g)
                for _ in range(10):
                    rk.step(eq, g, .01 * g.dx[0])
                self.assertTrue(np.isfinite(eq.u).all())
                states.append(np.asarray(eq.u))
            for candidate in (1, 2):
                np.testing.assert_array_equal(initial_rhs[0], initial_rhs[candidate])
                np.testing.assert_array_equal(states[0], states[candidate])

    def test_invalid_setting(self):
        """Only booleans are accepted."""
        for enabled in (-1, 0, 1, 4, 1.0, 'true', None):
            with self.assertRaises(ValueError):
                self.initialized(self.grid(), enabled)

    def test_exact_power_law_recovery_both_sides_and_gtt(self):
        """An exact R^-3 residual is reproduced on both sides, for X and gtt."""
        r = self.grid().xi[0]
        R = abs(r)
        geometry = tail.geometry(r, 3, True)
        ix = geometry[0].ravel()
        for parity in (1, -1):
            sign = np.ones_like(r) if parity == 1 else np.sign(r)
            x = sign * (.1 + .03 / R)
            dx = -.03 * sign * np.sign(r) / R**2
            S = -(np.sign(r) * dx + x / R)
            # Scaled coefficients make each fitted term measurable.
            E = (20. / R)**3 * sign
            for gtt in (False, True):
                raw = (S + E) * (R**2 if gtt else 1.)
                original = raw.copy()
                raw[ix] = 999.
                tail.apply(raw, x, dx, r, 1., 0., 1., *geometry, spherical_gtt=gtt)
                np.testing.assert_allclose(raw[ix], original[ix], rtol=2e-13, atol=1e-12)
                np.testing.assert_array_equal(raw[3:-3], original[3:-3])

    def test_flat_and_positive_domain_leave_inner_boundary_untouched(self):
        """On a half domain only the outer boundary is changed."""
        r = np.linspace(1., 20., 101)
        geometry = tail.geometry(r, 3, False)
        rhs = np.zeros_like(r)
        rhs[:3] = 123.
        tail.apply(rhs, np.ones_like(r), np.zeros_like(r), r, 1., 1., 1., *geometry)
        np.testing.assert_array_equal(rhs[-3:], 0.)
        np.testing.assert_array_equal(rhs[:3], 123.)

    def test_current_stage_parity_and_exact_interior(self):
        """Uses the supplied RK stage, keeps parity, leaves the interior bitwise."""
        for filtered in (False, True):
            g = self.grid()
            r = g.xi[0]
            if filtered:
                g.num_filters = 1
                g.Filter = [SimpleNamespace(apply_filter=FilterApply.APPLY_DERIVS,
                                            filter=lambda x: .91 * x)]
            base = self.initialized(g, False)
            u = [x.copy() for x in base.u]
            u[1][:] = 1 + .001*np.cos(.13*r)
            u[2][:] = r*r*(1 + .002*np.cos(.17*r))
            u[3][:] = 1e-8*r**4
            u[4][:] = 1e-5*np.cos(.3*r)
            u[5][:] += .0001*np.sin(.2*r)
            u[7][:] = .001*r
            u[8][:] = 1e-7*r**3
            reference = [np.zeros_like(x) for x in u]
            base.rhs(reference, u, g)
            for enabled in (False, True):
                e = self.initialized(g, enabled)
                for x in e.u:
                    x[:] = 999.
                out = [np.zeros_like(x) for x in u]
                e.rhs(out, u, g)
                for m in range(9):
                    np.testing.assert_array_equal(out[m][3:-3], reference[m][3:-3])
                    np.testing.assert_allclose(out[m][:3], e.u_parity[m] * out[m][-1:-4:-1],
                                               atol=5e-10, rtol=0)
                    if not enabled:
                        np.testing.assert_array_equal(out[m], reference[m])
                # Poisoning stored state must not change even boundary RHS.
                fresh = self.initialized(g, enabled)
                expected = [np.zeros_like(x) for x in u]
                fresh.rhs(expected, u, g)
                for a, b in zip(out, expected):
                    np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
