"""Regressions for Laplacian lapse damping and the telegrapher lapse.

Run: python -m unittest discover -s tests -p test_lapse_damping_telegrapher.py -v
"""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bssn'))
import gauges
from bssneqs import BSSN
from nwave import ExplicitFirst642_1D, ExplicitSecond642_1D, Grid1D


def make_bssn(lapse, xmin=-10.0, nx=401, extended=True):
    g = Grid1D({'Nx': nx, 'Xmin': xmin, 'Xmax': 10.0, 'NGhost': 3}, cell_centered=True)
    g.set_D1(ExplicitFirst642_1D(g.dx[0]))
    g.set_D2(ExplicitSecond642_1D(g.dx[0]))
    g.num_filters = 0
    eq = BSSN(g, 1.0, 2.0, extended, lapse_gauge=lapse)
    return g, eq


def flat_state(eq, r, alpha):
    """Flat space in spherical coordinates with a perturbed lapse, beta = 0."""
    u = [np.zeros_like(r) for _ in range(eq.Nu)]
    u[eq.U_CHI][:] = 1.0
    u[eq.U_GRR][:] = 1.0
    u[eq.U_GTT][:] = r**2
    u[eq.U_GT][:] = -2.0 / r
    u[eq.U_ALPHA][:] = alpha
    return u


class LaplacianDampingTests(unittest.TestCase):
    def test_adds_eta_lapse_times_flat_laplacian(self):
        r0 = None
        rhs = {}
        for eta in (0.0, 0.02, 0.04):
            g, eq = make_bssn(gauges.OnePlusLog(eta_lapse=eta))
            r = g.xi[0]
            alpha = 1.0 - 0.1 * np.exp(-((np.abs(r) - 5.0) ** 2))
            u = flat_state(eq, r, alpha)
            dtu = [np.zeros_like(r) for _ in range(eq.Nu)]
            eq.rhs(dtu, u, g)
            rhs[eta] = dtu[eq.U_ALPHA].copy()
            r0 = r
        r = r0
        rho = np.abs(r) - 5.0
        e = np.exp(-rho**2)
        d1 = 0.2 * rho * e * np.sign(r)                       # d_r alpha
        d2 = 0.2 * e * (1.0 - 2.0 * rho**2)                   # d_r^2 alpha
        flat_lap = d2 + 2.0 * d1 / r
        inner = (np.abs(r) > 1.0) & (np.abs(r) < 9.0)
        np.testing.assert_allclose(((rhs[0.02] - rhs[0.0]) / 0.02)[inner], flat_lap[inner], atol=2e-4)
        np.testing.assert_allclose(rhs[0.04] - rhs[0.0], 2.0 * (rhs[0.02] - rhs[0.0]), atol=1e-15)

    def test_default_has_no_damping(self):
        self.assertEqual(gauges.OnePlusLog().eta_lapse, 0.0)
        lapse, _ = gauges.build_gauges({'lapse': {'type': 'N2', 'eta_lapse': 0.01}})
        self.assertEqual(lapse.eta_lapse, 0.01)


class TelegrapherTests(unittest.TestCase):
    def test_rhs_aux_matches_documented_equation(self):
        rng = np.random.default_rng(3)
        n = 50
        alpha, A, K, lap = rng.uniform(0.05, 1.1, n), *rng.normal(size=(3, n))
        chi = rng.uniform(0.0, 1.4, n)
        alpha0 = rng.uniform(0.1, 1.0, n)
        refs = {'none': 0.0, 'unity': 1.0, 'chi': np.sqrt(chi), 'chi_clamped': np.sqrt(np.minimum(chi, 1.0)),
                'initial': alpha0}
        a = np.clip(alpha, 0.0, 1.0)
        gates = {'none': 1.0, 'alpha': a, 'alpha2': a * a,
                 'chi1': 0.5 * (1.0 - np.tanh((chi - 1.2) / 0.05))}
        for ref, aref in refs.items():
            for gate, gval in gates.items():
                with self.subTest(ref=ref, gate=gate):
                    t = gauges.Telegrapher(c=1.7, mu=0.6, mu2_sign=-0.5, mu2_ref=ref, mu2_gate=gate)
                    t.initial_aux(alpha0, np.zeros(n))
                    expected = (1.7**2 * lap - 2 * 0.6 * (A + 2 * alpha * K)
                                - 0.5 * 0.6**2 * gval * (alpha - aref))
                    np.testing.assert_allclose(t.rhs_aux(alpha, A, K, lap, chi), expected,
                                               rtol=1e-12, atol=1e-12)

    def test_bssn_layout_speeds_and_initial_data(self):
        t = gauges.Telegrapher(c=2.5)
        g, eq = make_bssn(t)
        self.assertEqual(eq.Nu, 10)
        self.assertEqual(eq.u_names[eq.U_LAPSE_AUX], 'lapseA')
        self.assertEqual((eq.u_falloff[9], eq.u_inf[9]), (2, 0))
        self.assertEqual(eq.u_wavespeed[eq.U_ALPHA], 2.5)
        self.assertEqual(eq.u_wavespeed[eq.U_LAPSE_AUX], 2.5)
        # EF data has K != 0: A starts on the 1+log equilibrium -2 alpha K.
        eq.initialize(g, {'initial_data': 'EddingtonFinkelstein'})
        np.testing.assert_array_equal(eq.u[9], -2.0 * eq.u[eq.U_ALPHA] * eq.u[eq.U_K])
        self.assertTrue(np.any(eq.u[9] != 0.0))

    def test_rhs_alpha_is_A_and_A_follows_rhs_aux(self):
        t = gauges.Telegrapher(c=1.5, mu=1.0, mu2_sign=-1.0, mu2_ref='unity')
        g, eq = make_bssn(t)
        r = g.xi[0]
        alpha = 1.0 - 0.1 * np.exp(-((np.abs(r) - 5.0) ** 2))
        u = flat_state(eq, r, alpha)
        u[9][:] = 0.3 * np.sin(r)
        dtu = [np.zeros_like(r) for _ in range(eq.Nu)]
        eq.rhs(dtu, u, g)
        inner = slice(3, -3)
        np.testing.assert_array_equal(dtu[eq.U_ALPHA][inner], u[9][inner])
        lap = BSSN.laplacian_alpha(u[0], u[1], u[2], g.D1.grad(alpha), 0 * r, 0 * r,
                                   g.D1.grad(u[2]), g.D2.grad(alpha))
        np.testing.assert_allclose(dtu[9][inner], t.rhs_aux(alpha, u[9], 0 * r, lap, u[0])[inner],
                                   rtol=1e-12, atol=1e-12)

    def test_half_domain_regularity_of_A(self):
        g, eq = make_bssn(gauges.Telegrapher(), xmin=0.0, nx=41, extended=False)
        stage = [np.arange(len(v), dtype=float) + 100 for v in eq.u]
        eq.apply_bcs(stage, g)
        np.testing.assert_array_equal(stage[9][:3], stage[9][5:2:-1])

    def test_defaults_are_frozen_reference_and_alpha2_gate(self):
        """Either alone runs away at the puncture; see gauges/telegrapher.py."""
        t = gauges.build_gauges({'lapse': {'type': 'telegrapher'}})[0]
        self.assertEqual((t.mu2_sign, t.mu2_ref, t.mu2_gate), (-1.0, 'initial', 'alpha2'))

    def test_initial_reference_needs_initial_aux(self):
        t = gauges.Telegrapher(mu2_ref='initial')
        with self.assertRaises(RuntimeError):
            t.rhs_aux(*np.ones((5, 4)))

    def test_bad_input_is_rejected(self):
        bad = [{'c': 0.0}, {'mu': -1.0}, {'mu2_sign': 1.5}, {'mu2_ref': 'zero'},
               {'mu2_gate': 'beta'}, {'gate_width': 0.0}, {'eta_lapse': 0.1}]
        for kw in bad:
            with self.subTest(**kw):
                with self.assertRaises(ValueError):
                    gauges.build_gauges({'lapse': {'type': 'telegrapher', **kw}})
        with self.assertRaises(ValueError):
            gauges.build_gauges({'lapse': {'type': '1+log', 'eta_lapse': -0.1}})


if __name__ == '__main__':
    unittest.main()
