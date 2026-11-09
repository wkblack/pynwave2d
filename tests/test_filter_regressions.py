"""Focused regressions; original implementation fails four of these tests."""
import ast
import builtins
import sys
import typing
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'bssn')]
# Unrelated upstream annotation import defect, identical for old/new tests.
builtins.Any = typing.Any
import bssnutils
from nwave import RK4, Filter1D, FilterApply, FilterType, KreissOligerFilterO8_1D

class LinearFilter(Filter1D):
    def __init__(self, rate):
        super().__init__(.1, FilterApply.RHS, FilterType.KREISS_OLIGER_O6, 1)
        self.rate, self.calls = rate, 0
    def filter(self, u):
        self.calls += 1
        return self.rate * u

class Equation:
    Nu = 1
    shp = [9]
    apply_bc = None
    def __init__(self): self.u = [np.linspace(1, 2, 9)]
    def rhs(self, out, u, grid): out[0][:] = 0

class Regressions(unittest.TestCase):
    def test_all_rhs_filters_contribute(self):
        f1, f2 = LinearFilter(-.3), LinearFilter(-.7)
        e = Equation(); initial = e.u[0].copy(); dt = .1
        RK4(e, SimpleNamespace(num_filters=2, Filter=[f1, f2])).step(
            e, SimpleNamespace(num_filters=2, Filter=[f1, f2]), dt)
        self.assertEqual([f1.calls, f2.calls], [4, 4])
        z = -dt
        np.testing.assert_allclose(e.u[0], initial * (1+z+z*z/2+z**3/6+z**4/24), atol=2e-15, rtol=0)

    def test_single_rhs_filter_unchanged(self):
        f = LinearFilter(-.7); e = Equation(); initial = e.u[0].copy()
        g = SimpleNamespace(num_filters=1, Filter=[f]); RK4(e,g).step(e,g,.1)
        z=-.07
        self.assertEqual(f.calls,4)
        np.testing.assert_allclose(e.u[0],initial*(1+z+z*z/2+z**3/6+z**4/24),atol=2e-15,rtol=0)

    def test_vars_selection_executes_real_solver_block(self):
        src=(ROOT/'bssn/solver.py').read_text()
        # Execute the actual selection block, not a duplicate repaired expression.
        block=src[src.index('    filvar = None'):src.index('    for i in range(1, Nt + 1):')]
        import textwrap
        f=SimpleNamespace(get_apply_filter=lambda: FilterApply.APPLY_VARS,get_frequency=lambda: 3)
        ns={'g':SimpleNamespace(num_filters=1,Filter=[f]),'FilterApply':FilterApply}
        exec(compile(ast.parse(textwrap.dedent(block)),'solver filter block','exec'),ns)
        self.assertIs(ns['filvar'], f); self.assertEqual(ns['filter_frequency'],3)

    def test_multiple_vars_filters_rejected(self):
        src=(ROOT/'bssn/solver.py').read_text()
        block=src[src.index('    filvar = None'):src.index('    for i in range(1, Nt + 1):')]
        import textwrap
        f=SimpleNamespace(get_apply_filter=lambda: FilterApply.APPLY_VARS,get_frequency=lambda: 3)
        ns={'g':SimpleNamespace(num_filters=2,Filter=[f,f]),'FilterApply':FilterApply}
        with self.assertRaisesRegex(ValueError, "Multiple Vars filters"):
            exec(compile(ast.parse(textwrap.dedent(block)),'solver filter block','exec'),ns)

    def test_ko8_boundary_reflection_and_quadratic(self):
        x=np.linspace(-2,2,81); f=KreissOligerFilterO8_1D(x[1]-x[0],.5,True)
        rng=np.random.default_rng(1); u=rng.normal(size=x.size)
        np.testing.assert_allclose(f.filter(u[::-1]),f.filter(u)[::-1],atol=1e-13,rtol=1e-14)
        self.assertLess(np.max(np.abs(f.filter(x*x))),1e-12)

    def test_ko_rejects_unsupported_semantics(self):
        x=np.linspace(-1,1,65)
        for kind in ['KO6','KO8']:
            for settings in [{'apply':'Vars','frequency':1},{'apply':'Rhs','frequency':5},{'apply':'Rhs','frequency':0}]:
                with self.subTest(kind=kind,settings=settings):
                    with self.assertRaises(bssnutils.ConfigError):
                        bssnutils.parse_filter_entry(dict(type=kind,**settings),x)
            f=bssnutils.parse_filter_entry({'type':kind,'apply':'Rhs','frequency':1},x)
            self.assertEqual(f.get_apply_filter(),FilterApply.RHS)
            self.assertEqual(f.get_frequency(),1)
if __name__ == '__main__': unittest.main()
