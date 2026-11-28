"""Regressions for horizon root finding and recovery after a failed search.

Run: python -m unittest discover -s tests -p test_bssn_horizon.py -v
BSSN_TEST_ROOT optionally selects an unmodified source tree as a baseline.
"""
from contextlib import redirect_stdout
import csv
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.interpolate import BarycentricInterpolator

ROOT = Path(os.environ.get("BSSN_TEST_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT / 'bssn'), str(ROOT)]
import bssneqs as b
import solver

CONFIG = """Nx=80
Nt=2
NGhost=3
Xmin=-20.0
Xmax=20.0
cfl=0.1
eta=2.0
D1="E642"
D2="E642"
BHMask=false
Mass=1.0
collapsed_lapse=true
initial_data="Puncture"
output_interval=1
print_interval=1
bhmask_outer_radius=15.0
"""

KO6 = """
[[filters]]
type="KO6"
apply="Rhs"
sigma=0.5
filter_boundary=false
"""


def model_on(x, nghost=0):
    grid = SimpleNamespace(
        xi=[x], shp=[len(x)], nghost=nghost,
        D1=SimpleNamespace(grad=np.zeros_like),
    )
    model = b.BSSN(grid, 1.0, 2.0, x[0] < 0.0, solver.BCType.FUNCTION,
                   b.GBSSNSystem(1, 1, 1), sommerfeld_tail=False)
    return model, grid


def manufactured(x, theta, *, chi=1.0, angular=1.0, nghost=0):
    """Prescribe expansion through K, with both spatial derivatives set to zero."""
    model, grid = model_on(x, nghost)
    model.u[model.U_CHI][:] = chi
    model.u[model.U_GRR][:] = 1.0
    model.u[model.U_GTT][:] = angular
    # The origin itself is outside the search domain.
    model.u[model.U_K][:] = -1.5 * theta / np.where(x == 0.0, 1.0, abs(x))
    return model, grid


def independent_expansion(x, state, radius, anchor):
    """Evaluate physical expansion from independently interpolated fields."""
    left = np.searchsorted(x, anchor) - 1
    stencil = slice(left - 3, left + 5)
    h = x[left + 1] - x[left]
    z = (radius - x[left]) / h
    poly = BarycentricInterpolator((x[stencil] - x[left]) / h,
                                  state[:5, stencil], axis=1, random_state=0)
    chi, grr, gtt, arr, k = poly(z)
    dchi, _, dgtt, _, _ = poly.derivative(z) / h
    return np.sqrt(chi / grr) * (dgtt / gtt - dchi / chi) + arr / grr - 2.0 * k / 3.0


def run_puncture(d, config):
    par = Path(d) / "params.toml"
    par.write_text(config + f'\noutput_dir="{d}/out"\n' + KO6)
    with redirect_stdout(StringIO()):
        solver.main(par)
    with (Path(d) / "out_0/bssn_constraints.dat").open() as file:
        return list(csv.DictReader(file))


class HorizonTests(unittest.TestCase):
    def test_horizon_roots(self):
        x = np.linspace(-2, 2, 17)
        for root in (0.5, 0.51, -0.5, -0.51):
            lo, hi = (0.1, 1.5) if root > 0 else (-1.5, -0.1)
            self.assertAlmostEqual(b.find_ah_zero(x - root, x, lo, hi), root)
        self.assertAlmostEqual(b.find_ah_zero((x - 0.5) * (x - 1.), x, 0.1, 1.5), 1.)
        self.assertIsNone(b.find_ah_zero(np.ones_like(x), x, 0.1, 1.5))
        self.assertIsNone(b.find_ah_zero(x - 0.5, x, 0.6, 1.5))

    def test_clipped_bounds_in_both_directions(self):
        x = np.linspace(-2.0, 2.0, 17)
        for sign in (-1.0, 1.0):
            for root, lo, hi, found in (
                (0.59, 0.60, 1.50, False),  # triage lower bound
                (0.61, 0.30, 0.70, True),   # triage upper bound
                (0.61, 0.60, 1.50, True),
                (0.71, 0.30, 0.70, False),
                (0.61, 0.60, 0.62, True),   # both bounds cut the same cell
                (0.50, 0.50, 1.00, True),
                (1.00, 0.50, 1.00, True),
                (1.00, 1.00, 1.00, True),
            ):
                with self.subTest(sign=sign, root=root, bounds=(lo, hi)):
                    for bounds in ((sign * lo, sign * hi), (sign * hi, sign * lo)):
                        actual = b.find_ah_zero(x - sign * root, x, *bounds)
                        if found:
                            self.assertIsNotNone(actual)
                            self.assertAlmostEqual(actual, sign * root)
                        else:
                            self.assertIsNone(actual)

    def test_outside_crossing_does_not_hide_next_candidate(self):
        x = np.linspace(-2.0, 2.0, 17)
        f = (x - 0.625) * (x - 1.125)
        self.assertAlmostEqual(b.find_ah_zero(f, x, 0.4, 1.5), 1.09375)
        self.assertAlmostEqual(b.find_ah_zero(f, x, 0.4, 1.06), 0.65625)

    def test_nonfinite_samples_do_not_bracket(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                f = np.array([-1.0, invalid, 1.0, 1.0])
                self.assertIsNone(b.find_ah_zero(f, x, 0.0, 3.0))
                f[2] = 0.0
                self.assertEqual(b.find_ah_zero(f, x, 2.0, 3.0), 2.0)

    def test_small_opposite_signs_still_bracket(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        self.assertEqual(b.find_ah_zero(1e-180 * (x - 1.5), x, 0.0, 3.0), 1.5)

    def test_cubic_acceptance_and_fallbacks(self):
        x = np.arange(0.0, 5.0, 0.25)
        root = 0.9
        for sign in (-1.0, 1.0):
            y = x - root
            model, grid = manufactured(x, sign * (y + y**2 + y**3),
                                       chi=1.0 + x**2, angular=2.0 + x**3)
            linear = model.find_horizon(grid, 0.7, "linear")
            self.assertEqual(model.horizon_interpolation_used, (1, False))
            radius, mass, _ = model.find_horizon(grid, 0.7, "cubic")
            self.assertEqual(model.horizon_interpolation_used, (3, False))
            self.assertAlmostEqual(radius, root, places=14)
            self.assertAlmostEqual(mass, 0.5 * np.sqrt((2.0 + root**3) / (1.0 + root**2)),
                                   places=14)
            self.assertGreater(abs(linear[0] - radius), 1e-3)

        theta = x - 0.875
        cell = slice(2, 6)  # [0.5, 0.75, 1.0, 1.25]
        unsafe = theta.copy()
        unsafe[cell] = [-20.0, -0.125, 0.125, 20.0]
        nonfinite = theta.copy()
        nonfinite[2] = np.nan
        overshoot = np.ones_like(x)
        overshoot[cell] = [20.0, 1.0, 1.0, 20.0]
        for label, f, chi, angular in (
            ("nonmonotone expansion", unsafe, 1.0, 1.0),
            ("nonfinite neighbor", nonfinite, 1.0, 1.0),
            ("negative angular interpolant", theta, 1.0, overshoot),
            ("negative chi interpolant", theta, overshoot, 1.0),
            ("negative angular and chi interpolants", theta, overshoot, overshoot),
        ):
            with self.subTest(label=label):
                model, grid = manufactured(x, f, chi=chi, angular=angular)
                linear = model.find_horizon(grid, 0.7, "linear")
                self.assertEqual(model.find_horizon(grid, 0.7, "cubic")[:2], linear[:2])
                self.assertEqual(model.horizon_interpolation_used, (1, True))

    def test_explicit_linear_readouts_match_554ded3_bitwise(self):
        x = np.arange(0.125, 4.0, 0.25)
        y = x - 0.91
        model, grid = manufactured(x, y + 0.5 * y**2, chi=1.0 + x*x,
                                   angular=2.0 + x**3)
        # Captured from 554ded3 using this fixture and the omitted option.
        expected = ("0x1.d02714c3079c1p-1", "0x1.3ac9c698683c3p-1",
                    "c419c62b476f0bb5a987107ad4cccb2173ab52a56abb77a2cf6bd71680da3053")
        radius, mass, theta = model.find_horizon(grid, 0.7, "linear")
        digest = hashlib.sha256(theta.astype("<f8").tobytes()).hexdigest()
        self.assertEqual((float(radius).hex(), float(mass).hex(), digest), expected)

    def test_cubic_crossing_cannot_leave_search_bounds(self):
        x = np.arange(0.0, 5.0, 0.25)
        # The linear root is below 2.8 and the monotone cubic root is above it.
        y = x - 2.802
        model, grid = manufactured(x, y + 0.5 * y**2)
        linear = model.find_horizon(grid, 0.7, "linear")
        self.assertLess(linear[0], 2.8)
        self.assertEqual(model.find_horizon(grid, 0.7, "cubic")[:2], linear[:2])
        self.assertEqual(model.horizon_interpolation_used, (1, True))
        self.assertAlmostEqual(model.find_horizon(grid, 0.75, "cubic")[0], 2.802)
        self.assertEqual(model.horizon_interpolation_used, (3, False))

        # The clipped cell can also move a crossing into the search window.
        y = x - 2.798
        model, grid = manufactured(x, y - 0.5 * y**2)
        self.assertIsNone(model.find_horizon(grid, 0.7, "linear")[0])
        self.assertAlmostEqual(model.find_horizon(grid, 0.7)[0], 2.798)
        self.assertEqual(model.horizon_interpolation_used, (3, False))

    def test_cubic_fallback_keeps_outer_crossing(self):
        x = np.arange(0.0, 5.0, 0.25)
        theta = (x - 0.875) * (x - 1.875)
        # The outer cell has positive geometry at both ends. Its cubic
        # angular metric dips below zero between the two endpoints.
        angular = np.ones_like(x)
        angular[6:10] = [20.0, 1.0, 1.0, 20.0]
        model, grid = manufactured(x, theta, angular=angular)
        linear = model.find_horizon(grid, 0.7, "linear")
        self.assertGreater(linear[0], 1.75)
        self.assertEqual(model.find_horizon(grid, 0.7, "cubic")[:2], linear[:2])
        self.assertEqual(model.horizon_interpolation_used, (1, True))

    def test_positive_active_stencil(self):
        x = np.arange(-1.125, 4.0, 0.25)
        for root, guess, ng, found in (
            (0.02, 0.065, 0, False),  # sign change spans the origin
            (0.20, 0.5, 0, True),    # stencil extends past the origin
            (3.55, 0.95, 2, False),  # sign change uses a ghost endpoint
            (3.30, 0.95, 2, True),   # only the extra neighbor is a ghost
        ):
            with self.subTest(root=root, nghost=ng):
                model, grid = manufactured(x, x - root, nghost=ng)
                radius, mass, _ = model.find_horizon(grid, guess, "cubic")
                if found:
                    self.assertAlmostEqual(radius, root)
                    self.assertEqual(model.horizon_interpolation_used, (1, True))
                else:
                    self.assertEqual((radius, mass), (None, None))
                    self.assertIsNone(model.horizon_interpolation_used)

        # Exact node at the first positive active radius uses only its own metric.
        model, grid = manufactured(x, x - 0.125, angular=np.where(x > 0, 1.0, np.nan))
        self.assertEqual(model.find_horizon(grid, 0.4, "cubic")[:2], (0.125, 0.5))
        self.assertEqual(model.horizon_interpolation_used, (1, False))

    def test_undefined_area_is_a_finder_miss(self):
        x = np.arange(0.0, 5.0, 0.25)
        theta = 0.1 * (x - 0.875)
        self.assertEqual(b.find_ah_zero(theta, x, 0.175, 2.8), 0.875)
        for chi, angular in ((1.0, 2e307), (0.0, 1.0), (np.nan, 1.0)):
            with self.subTest(chi=chi, angular=angular):
                model, grid = manufactured(x, theta, chi=chi, angular=angular)
                self.assertEqual(model.find_horizon(grid, 0.7, "linear")[:2], (None, None))

    def test_negative_geometry_reports_hit_with_warning(self):
        x = np.arange(0.0, 5.0, 0.25)
        for chi, angular, radial, warning in (
            (-1.0, 1.0, 1.0, "nonpositive chi"),
            (1.0, -1.0, 1.0, "nonpositive conformal metric"),
            (1.0, 1.0, -1.0, "nonpositive conformal metric"),
        ):
            with self.subTest(chi=chi, angular=angular, radial=radial):
                model, grid = manufactured(x, x - 0.875, chi=chi, angular=angular)
                model.u[model.U_GRR].fill(radial)
                output = StringIO()
                with redirect_stdout(output):
                    radius, mass, _ = model.find_horizon(grid, 0.7)
                self.assertEqual((radius, mass), (0.875, 0.5))
                self.assertIn(warning, output.getvalue())
                self.assertIn("linear fallback", output.getvalue())
                self.assertEqual("absolute area" in output.getvalue(), radial > 0.0)

    def test_zero_area_keeps_the_radius_and_mask(self):
        x = np.arange(0.0, 5.0, 0.25)
        model, grid = manufactured(x, x - 0.875, angular=np.where(x < 0.875, -1., 1.))
        radius, mass, _ = model.find_horizon(grid, 0.7)
        self.assertEqual((radius, mass), (0.875, 0.0))
        self.assertEqual(solver.update_horizon_tracking(radius, mass, 0.7, 0.7),
                         (0.875, 0.0, 0.875, 0.875))

    def test_cubic_recovers_when_linear_area_is_undefined(self):
        x = np.arange(0.0, 5.0, 0.25)
        chi = np.ones_like(x)
        chi[2:6] = [-2., -1., 1., 4.]
        model, grid = manufactured(x, x - 0.875, chi=chi)
        self.assertIsNone(model.find_horizon(grid, 0.7, "linear")[0])
        output = StringIO()
        with redirect_stdout(output):
            radius, mass, _ = model.find_horizon(grid, 0.7)
        self.assertEqual((radius, mass), (0.875, np.sqrt(2.0)))
        self.assertEqual(model.horizon_interpolation_used, (3, False))
        self.assertIn("linear area undefined", output.getvalue())
        self.assertIn("absolute area", output.getvalue())

    def test_exact_node_geometry_does_not_use_previous_sample(self):
        x = np.arange(0.0, 5.0, 0.25)
        model, grid = manufactured(x, x - 1.0, angular=np.where(x == 0.75, 1e16, 1.0))
        self.assertEqual(model.find_horizon(grid, 0.7)[:2], (1.0, 0.5))

    def test_undefined_outer_area_does_not_hide_inner_crossing(self):
        x = np.arange(0.0, 5.0, 0.25)
        angular = np.where((x >= 1.75) & (x <= 2.0), 2e307, 1.0)
        model, grid = manufactured(x, 0.1 * (x - 0.875) * (x - 1.875), angular=angular)
        output = StringIO()
        with redirect_stdout(output):
            radius, mass, _ = model.find_horizon(grid, 0.7)
        self.assertAlmostEqual(radius, 0.875)
        self.assertAlmostEqual(mass, 0.5)
        self.assertIn("undefined area at crossing", output.getvalue())

    def test_finite_area_survives_intermediate_overflow(self):
        x = np.arange(0.0, 5.0, 0.25)
        model, grid = manufactured(x, 0.01 * (x - 0.875), chi=2e307, angular=2e307)
        self.assertAlmostEqual(model.find_horizon(grid, 0.7, "linear")[1], 0.5)

    def test_window_is_clipped_to_positive_active_grid(self):
        x = np.arange(-0.125, 2.0, 0.25)
        model, grid = manufactured(x, x - 0.875, nghost=1)
        self.assertAlmostEqual(model.find_horizon(grid, 0.7)[0], 0.875)

    def test_recovery_retries_outside_stale_window(self):
        x = np.arange(0.0, 5.0, 0.25)
        model, grid = manufactured(x, x - 1.875)
        self.assertIsNone(model.find_horizon(grid, 0.25)[0])
        output = StringIO()
        with redirect_stdout(output):
            self.assertAlmostEqual(model.find_horizon(grid, 0.25, recover=True)[0], 1.875)
        self.assertIn("recovered outside tracking window", output.getvalue())

    def test_manufactured_convergence_across_phases(self):
        root = 0.9375
        for phase in (0.15, 0.35, 0.65, 0.85):
            with self.subTest(phase=phase):
                errors = []
                for h in (0.125, 0.0625, 0.03125):
                    x = root + h * (np.arange(-int(root / h) - 1, int(3.0 / h) + 1) - phase)
                    model, grid = manufactured(x, np.expm1(x - root))
                    linear = model.find_horizon(grid, 0.7, "linear")[0]
                    cubic = model.find_horizon(grid, 0.7)[0]
                    self.assertEqual(model.horizon_interpolation_used, (3, False))
                    errors.append(np.abs(np.array([linear, cubic]) - root))
                rates = np.array(errors[:-1]) / errors[1:]
                np.testing.assert_allclose(rates, np.broadcast_to([4, 16], (2, 2)), rtol=0.07)

    def test_schwarzschild_initial_radius_and_area(self):
        for phase in (0.15, 0.35, 0.65, 0.85):
            with self.subTest(phase=phase):
                h = 0.0125
                x = 0.5 + h * (np.arange(-36, 244) - phase)
                model, grid = model_on(x)
                grid.D1 = solver.ExplicitFirst642_1D(h)
                model.initial_data_puncture(grid, {"collapsed_lapse": True})
                linear_r, linear_m, _ = model.find_horizon(grid, 0.7, "linear")
                radius, mass, _ = model.find_horizon(grid, 0.7)
                self.assertEqual(model.horizon_interpolation_used, (3, False))
                self.assertLess(abs(radius - 0.5), abs(linear_r - 0.5) / 20.0)
                self.assertLess(abs(mass**2 - 1.0), abs(linear_m**2 - 1.0) / 20.0)
                self.assertAlmostEqual(radius, 0.5, places=6)
                self.assertAlmostEqual(16.0 * np.pi * mass**2, 16.0 * np.pi, places=5)

    def test_horizon_tracking(self):
        good = solver.update_horizon_tracking(0.5, 1., 0.7, 0.)
        self.assertEqual(good, (0.5, 1., 0.5, 0.5))
        failed = solver.update_horizon_tracking(None, None, good[2], good[3])
        self.assertEqual(failed, (0., 0., 0.5, 0.5))
        recovered = solver.update_horizon_tracking(0.51, 1., failed[2], failed[3])
        self.assertEqual(recovered, (0.51, 1., 0.51, 0.51))

    def test_invalid_mass_does_not_advance_tracker(self):
        for mass in (None, np.nan, np.inf, -1.0):
            with self.subTest(mass=mass):
                self.assertEqual(solver.update_horizon_tracking(0.6, mass, 0.5, 0.5),
                                 (0.0, 0.0, 0.5, 0.5))

    def test_solver_recovers_and_reports_mask(self):
        self.check_recovery("None", "None")

    def test_solver_reports_invalid_mass_as_a_miss(self):
        self.check_recovery("0.6", "np.nan")

    def test_solver_seeds_from_mass_and_counts_active_points(self):
        find = b.BSSN.find_horizon
        calls = []

        def record(model, grid, guess, **options):
            radius, mass, theta = find(model, grid, guess, **options)
            calls.append((grid, guess, radius, options))
            return radius, mass, theta

        config = CONFIG.replace("Nx=80", "Nx=800").replace("Mass=1.0", "Mass=8.0")
        with tempfile.TemporaryDirectory() as d, patch.object(b.BSSN, "find_horizon", record):
            rows = run_puncture(d, config)
        grid, seed, radius, options = calls[0]
        self.assertEqual(seed, 4.0)
        self.assertTrue(options["recover"])
        self.assertEqual(rows[0]["horizon_found"], "True")
        self.assertAlmostEqual(radius, 4.0, places=5)

        # Count positive active points strictly inside the radius, as in the mask.
        active = grid.xi[0][grid.nghost : -grid.nghost]
        radius = active[active > 0][320] + 0.01 * grid.dx[0]
        output = StringIO()
        with redirect_stdout(output):
            solver.report_horizon(grid, radius, 1.0)
        self.assertIn("bhpts=321\n", output.getvalue())

    def test_selected_horizon_changes_only_diagnostics(self):
        def run_with_horizon(radius):
            states = []

            def find(model, grid, guess, **options):
                states.append(np.array(model.u).tobytes())
                return radius, 1.0, np.zeros_like(grid.xi[0])

            with tempfile.TemporaryDirectory() as d, patch.object(b.BSSN, "find_horizon", find):
                rows = run_puncture(d, CONFIG.replace("Nx=80", "Nx=800"))
            return rows, states

        near, near_state = run_with_horizon(0.4)
        far, far_state = run_with_horizon(0.6)
        self.assertEqual(near_state, far_state)
        self.assertEqual({float(row["mask_inner_radius"]) for row in near}, {0.4})
        self.assertEqual({float(row["mask_inner_radius"]) for row in far}, {0.6})
        self.assertNotEqual(near[-1]["Ham"], far[-1]["Ham"])

    def test_ko6_puncture_smokes_and_independent_expansion(self):
        base = CONFIG.replace("Nx=80", "Nx=800").replace("Nt=2", "Nt=4")
        compact = (base.replace('"E642"', '["JP6", "ME6"]')
                   .replace("BHMask=false", "BHMask=true\nBHMaskPos=0.0\nBHMaskWidth=0.05"))
        find = b.BSSN.find_horizon

        for label, config in (("E642", base), ("compact", compact)):
            with self.subTest(operator=label):
                samples = []

                def record(model, grid, guess, **options):
                    radius, mass, theta = find(model, grid, guess, **options)
                    samples.append((radius, mass, model.horizon_interpolation_used,
                                    grid.xi[0].copy(), np.array(model.u)))
                    return radius, mass, theta

                with patch.object(b.BSSN, "find_horizon", record):
                    with tempfile.TemporaryDirectory() as d:
                        linear_rows = run_puncture(d, config + 'horizon_interpolation="linear"\n')
                    with tempfile.TemporaryDirectory() as d:
                        cubic_rows = run_puncture(d, config)

                for i, (lin, cub) in enumerate(zip(samples[:5], samples[5:], strict=True)):
                    self.assertEqual(lin[4].tobytes(), cub[4].tobytes())
                    self.assertEqual(lin[2], (1, False))
                    self.assertEqual(cub[2], (3, False))
                    for sample, row in ((lin, linear_rows[i]), (cub, cubic_rows[i])):
                        self.assertEqual(row["horizon_found"], "True")
                        self.assertEqual(float(row["mask_inner_radius"]), sample[0])
                        self.assertAlmostEqual(sample[1], 1.0, places=2)

                # Freeze the last evolved slice and use the same fields and
                # independent interpolant to evaluate expansion at both roots.
                linear, cubic = samples[4][0], samples[-1][0]
                x, state = samples[-1][3:]
                residual_linear = independent_expansion(x, state, linear, linear)
                residual_cubic = independent_expansion(x, state, cubic, linear)
                self.assertLess(abs(residual_cubic), 0.2 * abs(residual_linear))

    def test_coarse_puncture_falls_back(self):
        find = b.BSSN.find_horizon
        used = []

        def record(model, grid, guess, **options):
            result = find(model, grid, guess, **options)
            used.append(model.horizon_interpolation_used)
            return result

        # The initial horizon is bracketed by the first two positive cells.
        with tempfile.TemporaryDirectory() as d, patch.object(b.BSSN, "find_horizon", record):
            rows = run_puncture(d, CONFIG)
        self.assertEqual(used, [(1, True)] * 3)
        self.assertEqual([row["horizon_found"] for row in rows], ["True"] * 3)

    def check_recovery(self, miss_radius, miss_mass):
        with tempfile.TemporaryDirectory() as d:
            par = Path(d) / "params.toml"
            par.write_text(CONFIG + f'output_dir="{d}/out"\n')
            # Injected detections: 0.5, then a failure, then 0.51 only if the
            # search guess survived the failure.
            code = f"""import sys, json
import numpy as np
sys.path[:0] = [sys.argv[1] + '/bssn', sys.argv[1]]
import solver
calls = []
def find(self, g, guess, **options):
    calls.append(float(guess))
    n = len(calls)
    radius = 0.5 if n == 1 else {miss_radius} if n == 2 else 0.51 if guess > 0 else None
    return radius, {miss_mass} if n == 2 else 1., np.zeros_like(g.xi[0])
solver.bssn.BSSN.find_horizon = find
solver.main(sys.argv[2])
print('GUESSES=' + json.dumps(calls))
"""
            p = subprocess.run([sys.executable, '-c', code, str(ROOT), str(par)],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 0, p.stderr)
            guesses = json.loads(p.stdout.split('GUESSES=')[-1])
            self.assertEqual(guesses, [0.5, 0.5, 0.5])
            with (Path(d) / 'out_0/bssn_constraints.dat').open() as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, [
                    'time', 'Ham', 'Mom', 'radius_bh', 'mass_bh', 'horizon_found',
                    'mask_inner_radius', 'Gamcon',
                ])
                rows = list(reader)
            self.assertEqual([r['horizon_found'] for r in rows], ['True', 'False', 'True'])
            self.assertEqual([float(r['mask_inner_radius']) for r in rows], [0.5, 0.5, 0.51])
            self.assertEqual([float(r['radius_bh']) for r in rows], [0.5, 0., 0.51])
            self.assertEqual([float(r['mass_bh']) for r in rows], [1.0, 0.0, 1.0])
            if miss_mass == "np.nan":
                self.assertEqual(p.stdout.count("Horizon(s) found"), 2)
                self.assertIn("Horizon not found; retaining search guess", p.stdout)


if __name__ == "__main__":
    unittest.main()
