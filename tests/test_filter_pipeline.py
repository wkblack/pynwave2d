"""Biting pipeline/config regressions, without requiring compact derivatives."""
import builtins
import sys
import typing
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'bssn')]
# Upstream has an independent missing typing.Any import, fixed separately.
builtins.Any = typing.Any
import bssnutils
from nwave import RK4, Filter1D, FilterApply, FilterType, KreissOligerFilterO6_1D

class LinearFilter(Filter1D):
    def __init__(self, rate):
        super().__init__(.1, FilterApply.RHS, FilterType.KREISS_OLIGER_O6, 1)
        self.rate, self.calls = rate, 0
    def filter(self, u):
        self.calls += 1
        return self.rate * u

class Equation:
    Nu = 1
    apply_bc = None
    def __init__(self, values=None):
        self.u = [np.linspace(1, 2, 9) if values is None else values.copy()]
        self.shp = list(self.u[0].shape)
    def rhs(self, out, u, grid): out[0][:] = 0

def step(filters, values=None, dt=.1, nsteps=1):
    e=Equation(values);g=SimpleNamespace(num_filters=len(filters),Filter=filters);rk=RK4(e,g)
    for _ in range(nsteps):rk.step(e,g,dt)
    return e.u[0]

class FilterPipeline(unittest.TestCase):
    def test_all_rhs_filters_contribute(self):
        f1, f2 = LinearFilter(-.3), LinearFilter(-.7)
        initial=np.linspace(1,2,9);out=step([f1,f2],initial)
        self.assertEqual([f1.calls, f2.calls], [4, 4])
        z=-.1
        np.testing.assert_allclose(out,initial*(1+z+z*z/2+z**3/6+z**4/24),atol=2e-15,rtol=0)

    def test_single_rhs_filter_unchanged(self):
        f=LinearFilter(-.7);initial=np.linspace(1,2,9);out=step([f],initial)
        z=-.07
        self.assertEqual(f.calls,4)
        np.testing.assert_allclose(out,initial*(1+z+z*z/2+z**3/6+z**4/24),atol=2e-15,rtol=0)

    def test_two_actual_ko_filters_match_single_sum(self):
        x=np.linspace(-1,1,129);dx=x[1]-x[0]
        u=np.exp(-30*x*x)*np.cos(20*np.pi*x)
        # Interior-only KO with tiny timestep; two contributions should sum.
        many=step([KreissOligerFilterO6_1D(dx,.2,False),KreissOligerFilterO6_1D(dx,.3,False)],u,.01*dx,100)
        single=step([KreissOligerFilterO6_1D(dx,.5,False)],u,.01*dx,100)
        np.testing.assert_allclose(many,single,atol=2e-15,rtol=2e-15)
        self.assertTrue(np.isfinite(many).all())
        self.assertLess(np.linalg.norm(many),np.linalg.norm(u))

    def test_explicit_none_is_valid_sentinel(self):
        x=np.linspace(-1,1,65)
        fs=bssnutils.init_filter(x,{'filters':[{'type':'None','apply':'None'}]})
        self.assertEqual(len(fs),1)
        self.assertEqual(fs[0].get_apply_filter(),FilterApply.NONE)
        initial=np.linspace(1,2,9)
        np.testing.assert_array_equal(step(fs,initial),initial)

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

    def test_multiple_vars_rejected_before_solver(self):
        x=np.linspace(-1,1,65)
        f=SimpleNamespace(get_apply_filter=lambda: FilterApply.APPLY_VARS)
        with patch.object(bssnutils,'parse_filter_entry',return_value=f):
            with self.assertRaisesRegex(bssnutils.ConfigError,'Multiple Vars'):
                bssnutils.init_filter(x,{'filters':[{},{}]})
            self.assertEqual(bssnutils.init_filter(x,{'filters':[{}]}),[f])
if __name__=='__main__':unittest.main()
