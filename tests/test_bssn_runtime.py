"""Regression checks for isolated runtime, output, and horizon fixes."""
import os, sys, tempfile, subprocess, unittest
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get('BSSN_TEST_ROOT',Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT/'bssn'),str(ROOT)]
# Baseline import shim only: startup is separately tested without it.
if os.environ.get('BSSN_BASELINE_SHIM'):
    import builtins, typing
    builtins.Any = typing.Any
import bssneqs as b
import bssnutils as util
import solver
from nwave import Grid1D, FilterNone_1D, FilterApply

class RuntimeTests(unittest.TestCase):
    def test_clean_import(self):
        p=subprocess.run([sys.executable,'-c',"import sys;sys.path[:0]="+repr([str(ROOT/'bssn'),str(ROOT)])+";import bssnutils"],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
    def test_masked_writer(self):
        from types import SimpleNamespace
        e=SimpleNamespace(u_names=['u'],c_names=['C'],u=[np.array([2.,4.,6.])],C=[np.array([1.,3.,5.])])
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'a.curve'; util.write_curve_masked(f,0.,np.array([10.,11.,12.]),e,np.array([1.,0.,1.]))
            data=f.read_text(); self.assertIn('1.10000000e+01 0.00000000e+00',data)
            self.assertIn('1.20000000e+01 5.00000000e+00',data)
    def test_none_filter(self):
        fs=util.init_filter(np.linspace(0.,1.,21),{'filters':[{'type':'None','apply':'None'}]})
        self.assertIsInstance(fs[0],FilterNone_1D)
        self.assertEqual(fs[0].get_apply_filter(),FilterApply.NONE)
    def test_stage_regularization(self):
        g=Grid1D({'Nx':21,'NGhost':3,'Xmin':0.,'Xmax':2.},cell_centered=True)
        e=b.BSSN(g,1.,2.,False)
        for v in e.u: v[:]=np.arange(len(v))
        base=[v.copy() for v in e.u]; stage=[v.copy()+100 for v in e.u]
        e.apply_bcs(stage,g)
        for original,current in zip(base,e.u): np.testing.assert_array_equal(current,original)
        for j in range(9):
            sign=-1 if j in (e.U_GT,e.U_SHIFT,e.U_GB) else 1
            np.testing.assert_array_equal(stage[j][:3],sign*stage[j][5:2:-1])
            np.testing.assert_array_equal(stage[j][3:],base[j][3:]+100)
    def test_extended_unchanged(self):
        g=Grid1D({'Nx':21,'NGhost':3,'Xmin':-2.,'Xmax':2.},cell_centered=True)
        e=b.BSSN(g,1.,2.,True); stage=[np.arange(len(v),dtype=float)+100 for v in e.u]; before=[v.copy() for v in stage]
        e.apply_bcs(stage,g)
        for a,z in zip(before,stage): np.testing.assert_array_equal(a,z)
    def test_horizon_roots(self):
        x=np.linspace(-2,2,17)
        for root in (.5,.51,-.5,-.51):
            lo,hi=(.1,1.5) if root>0 else (-1.5,-.1)
            self.assertAlmostEqual(b.find_ah_zero(x-root,x,lo,hi),root)
        self.assertAlmostEqual(b.find_ah_zero((x-.5)*(x-1.),x,.1,1.5),1.)
        self.assertIsNone(b.find_ah_zero(np.ones_like(x),x,.1,1.5))
        self.assertIsNone(b.find_ah_zero(x-.5,x,.6,1.5))
    def test_horizon_tracking(self):
        good=solver.update_horizon_tracking(.5,1.,.7,0.)
        self.assertEqual(good,(.5,1.,.5,.5))
        failed=solver.update_horizon_tracking(None,None,good[2],good[3])
        self.assertEqual(failed,(0.,0.,.5,.5))
        recovered=solver.update_horizon_tracking(.51,1.,failed[2],failed[3])
        self.assertEqual(recovered,(.51,1.,.51,.51))
    def test_nonfinite_every_field(self):
        for j in range(9):
            for bad in (np.nan,np.inf,-np.inf):
                u=[np.ones(10) for _ in range(9)];u[j][4]=bad
                with self.assertRaises(FloatingPointError):solver.require_finite_state(u,7)
        solver.require_finite_state([np.ones(10) for _ in range(9)],7)
        code="import sys;sys.path[:0]="+repr([str(ROOT/'bssn'),str(ROOT)])+";import solver,numpy as np;solver.require_finite_state([np.array([np.inf])],7)"
        p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0)

if __name__=='__main__':unittest.main()
