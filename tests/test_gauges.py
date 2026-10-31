"""Regressions for the gauge registry in bssn/gauges.

Run: python -m unittest discover -s tests -p test_gauges.py -v
No evolution is needed.
"""
import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bssn'))
import gauges
from bssneqs import BSSN
from nwave import Grid1D

ALPHA = np.linspace(0.01, 1.2, 50)

# Bona-Masso f(alpha) as documented, and the expected sqrt(f(1)).
LAPSE_CASES = [
    ({'type': '1+log'}, lambda a: 2.0 / a, np.sqrt(2.)),
    ({'type': '1+log', 'k': 1.46}, lambda a: 1.46 / a, np.sqrt(1.46)),
    ({'type': 'N1'}, lambda a: (2 / a) * (1 - a / 2 + a**2 / 3), np.sqrt(5 / 3)),
    ({'type': 'N2'}, lambda a: (2 / a) * (1 - a + a**2 - a**3 / 4), np.sqrt(3 / 2)),
    ({'type': 'N3'},
     lambda a: (2 / a) * (1 - 3 * a / 2 + 2 * a**2 - a**3 + a**4 / 5), np.sqrt(7 / 5)),
    ({'type': 'ShockFull', 'kappa': 0.5}, lambda a: 1 + 0.5 / a**2, np.sqrt(1.5)),
]


class LapseTests(unittest.TestCase):
    def test_alpha2_f_matches_documented_f(self):
        for table, f, _ in LAPSE_CASES:
            with self.subTest(**table):
                lapse, _ = gauges.build_gauges({'lapse': dict(table)})
                np.testing.assert_allclose(
                    lapse.alpha2_f(ALPHA), ALPHA**2 * f(ALPHA), rtol=1e-13)

    def test_speed_inf_is_sqrt_f1(self):
        for table, _, speed in LAPSE_CASES:
            with self.subTest(**table):
                lapse, _ = gauges.build_gauges({'lapse': dict(table)})
                self.assertAlmostEqual(lapse.speed_inf(), speed, places=14)

    def test_default_is_bitwise_2_alpha(self):
        """The default must reproduce the original -2*alpha*K term exactly."""
        lapse, _ = gauges.build_gauges({})
        self.assertEqual(lapse.alpha2_f(ALPHA).tobytes(), (2 * ALPHA).tobytes())
        self.assertEqual(lapse.speed_inf(), np.sqrt(2.))

    def test_bssn_uses_gauge_speed_at_outer_boundary(self):
        g = Grid1D({'Nx': 21, 'Xmin': -10., 'Xmax': 10., 'NGhost': 3}, cell_centered=True)
        lapse, shift = gauges.build_gauges({'lapse': {'type': 'ShockFull', 'kappa': 0.5}})
        eq = BSSN(g, 1., shift, True, lapse_gauge=lapse)
        self.assertEqual(eq.u_wavespeed[eq.U_ALPHA], np.sqrt(1.5))
        np.testing.assert_array_equal(np.delete(eq.u_wavespeed, eq.U_ALPHA), 1.)


class ShiftTests(unittest.TestCase):
    r = np.linspace(-20., 20., 401)

    def eta(self, table, chi=None, g_rr=None, d_chi=None):
        _, shift = gauges.build_gauges({'shift_damping': table})
        ones = np.ones_like(self.r)
        return shift(self.r, ones if chi is None else chi,
                     ones if g_rr is None else g_rr,
                     0 * ones if d_chi is None else d_chi, 1.0)

    def test_default_is_scalar_2(self):
        _, shift = gauges.build_gauges({})
        self.assertEqual(shift(self.r, None, None, None, 1.0), 2.0)

    def test_number_eta_still_accepted_by_bssn(self):
        g = Grid1D({'Nx': 21, 'Xmin': -10., 'Xmax': 10., 'NGhost': 3}, cell_centered=True)
        eq = BSSN(g, 1., 3.0, True)
        self.assertEqual(eq.shift_damping(self.r, None, None, None, 1.0), 3.0)

    def test_schnetter_profile(self):
        e = self.eta({'type': 'Schnetter', 'eta': 2.0, 'R': 5.0})
        self.assertEqual(e[self.r == 0][0], 2.0)
        np.testing.assert_allclose(e[np.isclose(np.abs(self.r), 5.0)], 1.0)

    def test_peaked_lorentz_profile(self):
        e = self.eta({'type': 'peaked_lorentz', 'bg': 1.0, 'peak': 3.0, 'R': 2.0})
        self.assertEqual(e[self.r == 0][0], 4.0)
        np.testing.assert_allclose(e[np.isclose(np.abs(self.r), 2.0)], 2.5)

    def test_eta_mb_on_puncture_data(self):
        """chi = psi^-4, psi = 1 + M/2r: eta = R0 |W'| / (1 - W)^2, W = psi^-2."""
        r = self.r[self.r > 0.5]
        psi = 1 + 0.5 / r
        chi, d_chi = psi**-4, 2 * psi**-5 / r**2
        W, dW = psi**-2, psi**-3 / r**2
        _, shift = gauges.build_gauges({'shift_damping': {'type': 'eta_MB', 'R0': 1.3}})
        np.testing.assert_allclose(
            shift(r, chi, np.ones_like(r), d_chi, 1.0),
            1.3 * dW / (1 - W)**2, rtol=1e-8)


class RegistryTests(unittest.TestCase):
    def test_every_registered_gauge_builds_with_defaults(self):
        for name in gauges.LAPSE_GAUGES:
            with self.subTest(lapse=name):
                lapse, _ = gauges.build_gauges({'lapse': {'type': name}})
                self.assertTrue(np.isfinite(lapse.speed_inf()))
        for name in gauges.SHIFT_DAMPING:
            with self.subTest(shift=name):
                gauges.build_gauges({'shift_damping': {'type': name}})

    def test_bad_input_is_rejected(self):
        bad = {
            'eta given twice': {'eta': 2.0, 'shift_damping': {'type': 'constant'}},
            'unknown lapse type': {'lapse': {'type': '1+lg'}},
            'unknown key': {'lapse': {'type': '1+log', 'kk': 2.0}},
            'missing type': {'shift_damping': {'eta': 2.0}},
            'eta < 0': {'shift_damping': {'type': 'constant', 'eta': -0.5}},
            'Schnetter eta < 0': {'shift_damping': {'type': 'Schnetter',
                                                   'eta': -0.5}},
            'kappa <= 0': {'lapse': {'type': 'ShockFull', 'kappa': 0.0}},
            'k <= 0': {'lapse': {'type': '1+log', 'k': -1.0}},
        }
        for why, params in bad.items():
            with self.subTest(why):
                with self.assertRaises(ValueError):
                    gauges.build_gauges(params)

    def test_old_format_params_still_run(self):
        """Pre-table files: top-level eta, no gauge tables."""
        lapse, shift = gauges.build_gauges({'eta': 3.0, 'Mass': 1.0})
        self.assertEqual(lapse.alpha2_f(ALPHA).tobytes(), (2 * ALPHA).tobytes())
        self.assertEqual(shift(None, None, None, None, 1.0), 3.0)
        with self.assertRaises(ValueError):
            gauges.build_gauges({'eta': -0.5})

    def test_eta_below_one_is_allowed(self):
        for eta in (0.0, 0.5):
            for kind in ('constant', 'Schnetter'):
                with self.subTest(kind=kind, eta=eta):
                    _, shift = gauges.build_gauges(
                        {'shift_damping': {'type': kind, 'eta': eta}})
                    self.assertEqual(shift(0.0, None, None, None, 1.0), eta)

    def test_unknown_type_lists_options(self):
        with self.assertRaisesRegex(ValueError, 'ShockFull'):
            gauges.build_gauges({'lapse': {'type': 'nope'}})


if __name__ == '__main__':
    unittest.main()
