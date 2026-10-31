"""Eddington-Finkelstein initial data: Kerr-Schild outside, puncture-stuffed inside.

Run: python -m unittest discover -s tests -p test_ef_initial_data.py -v
"""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'bssn'), str(ROOT)]
from bssneqs import BSSN
from nwave import Grid1D, ExplicitFirst642_1D, ExplicitSecond642_1D

CONFIG = """Nx=500
Nt=500
NGhost=3
Xmin=0.0
Xmax=20.0
cfl=0.1
eta=2.0
D1=["JT4", "ME6"]
D2=["JT4", "ME6"]
BHMask=true
BHMaskPos=0.0
BHMaskWidth=0.05
Mass=1.0
collapsed_lapse=true
initial_data="EddingtonFinkelstein"
output_interval=500
print_interval=500

[[filters]]
type = "KO6"
apply = "Rhs"
sigma = 0.5
filter_boundary = false
"""


def ef_data(nx, xmin, xmax, **params):
    # Nx + 1 cell centres, as the solver builds them (none at x = 0)
    g = Grid1D({'Nx': nx + 1, 'Xmin': xmin, 'Xmax': xmax, 'NGhost': 3}, cell_centered=True)
    g.set_D1(ExplicitFirst642_1D(g.dx[0]))
    g.set_D2(ExplicitSecond642_1D(g.dx[0]))
    eq = BSSN(g, 1., 2., xmin < 0.)
    eq.initialize(g, {'initial_data': 'EddingtonFinkelstein', **params})
    return g.xi[0], eq


class EFTests(unittest.TestCase):
    def test_exterior_is_kerr_schild(self):
        """Outside the stuffing radius the data are exact, with Gt and beta odd in x."""
        x, eq = ef_data(400, -20., 20.)
        r, m = abs(x), 1.
        H = 2. * m / r
        exact = {
            eq.U_CHI: (1. + H) ** (-1. / 3.),
            eq.U_GRR: (1. + H) ** (2. / 3.),
            eq.U_GTT: r**2 / (1. + H) ** (1. / 3.),
            eq.U_ARR: -4. * m * (1. + H) ** (1. / 6.) * (3. * m + 2. * r)
            / (3. * r * r * (r + 2. * m)),
            eq.U_K: 2 * m * (r + 3. * m) / (r * r * (2. * m + r) * np.sqrt(1. + H)),
            eq.U_GT: -2. * np.sign(x) * (8 * m + 3 * r) * (1. + H) ** (1. / 3.)
            / (3. * (2. * m + r) ** 2),
            eq.U_ALPHA: 1. / np.sqrt(1. + H),
            eq.U_SHIFT: np.sign(x) * H / (1. + H),
        }
        out = r >= 1.8
        for k, v in exact.items():
            np.testing.assert_allclose(eq.u[k][out], v[out], rtol=1e-12, atol=1e-12,
                                       err_msg=eq.u_names[k])

    def test_interior_is_puncture_and_gt_is_consistent(self):
        """Inside R/2 the data are the collapsed puncture; Gt follows the blended metric."""
        x, eq = ef_data(20000, 0., 4.)
        r = abs(x)
        inner = r < 0.9
        chi = (1. + 0.5 / r[inner]) ** -4
        np.testing.assert_allclose(eq.u[eq.U_CHI][inner], chi, rtol=1e-12)
        np.testing.assert_allclose(eq.u[eq.U_ALPHA][inner], np.sqrt(chi), rtol=1e-12)
        for k in (eq.U_ARR, eq.U_K, eq.U_SHIFT):
            np.testing.assert_array_equal(eq.u[k][inner], 0.)
        grr, gtt = eq.u[eq.U_GRR], eq.u[eq.U_GTT]
        gt = np.gradient(grr, r) / (2 * grr**2) - np.gradient(gtt, r) / (grr * gtt)
        shell = (x > 0.9) & (x < 1.8)
        np.testing.assert_allclose(eq.u[eq.U_GT][shell], gt[shell], rtol=1e-5, atol=1e-6)

    def test_stuffing_radius_must_be_inside_horizon(self):
        for radius in (0., -1., 2., 2.5):
            with self.assertRaises(ValueError):
                ef_data(40, 0., 20., ef_stuff_radius=radius)

    def test_evolves_and_keeps_its_mass(self):
        """Unstuffed data went NaN within three steps; this runs 2 M with the mass intact."""
        with tempfile.TemporaryDirectory() as d:
            par = Path(d) / 'params.toml'
            par.write_text(f'output_dir="{d}/out"\n' + CONFIG)
            code = ("import sys; sys.path[:0] = [sys.argv[1] + '/bssn', sys.argv[1]]\n"
                    "import solver; solver.main(sys.argv[2])\n")
            p = subprocess.run([sys.executable, '-c', code, str(ROOT), str(par)],
                               capture_output=True, text=True, timeout=300,
                               env={'MPLBACKEND': 'Agg', 'PATH': ''})
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            with (Path(d) / 'out_0/bssn_constraints.dat').open() as f:
                rows = list(csv.DictReader(f))
            self.assertAlmostEqual(float(rows[-1]['time']), 2., places=6)
            self.assertAlmostEqual(float(rows[-1]['mass_bh']), 1., delta=1e-3)


if __name__ == "__main__":
    unittest.main()
