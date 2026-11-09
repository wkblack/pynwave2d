"""Independent analytic actual-RHS regressions, including B-driver coupling."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'bssn'), str(ROOT)]
from bssneqs import BSSN
from nwave import Grid1D, ExplicitFirst642_1D, ExplicitSecond642_1D, FilterApply

class GammaRHSTests(unittest.TestCase):
    def check_rhs(self, filtered):
        g=Grid1D({'Nx':101,'Xmin':-5.,'Xmax':5.,'NGhost':3},cell_centered=True)
        r=g.xi[0]; dr=g.dx[0]; g.set_D1(ExplicitFirst642_1D(dr));g.set_D2(ExplicitSecond642_1D(dr))
        core=np.abs(r)<4.5
        g.num_filters = int(filtered)
        if filtered:
         g.Filter = [SimpleNamespace(apply_filter=FilterApply.APPLY_DERIVS, filter=lambda x: .9*x)]
        records=[]
        for strength in (0.7,-0.7):
         for slope in (-.13,0.,.13):
          base=BSSN(g,1.,2.,True,c_G=0.); damp=BSSN(g,1.,2.,True,c_G=strength)
          # Full angular metric, including r², with analytic polynomial derivatives.
          a=1+.1*r*r; b=r*r*(1+.03*r*r)
          ap=.2*r;bp=2*r+.12*r**3
          G=ap/(2*a*a)-bp/(a*b)
          for stage in (1.,1.4):
           u=[np.zeros_like(r) for _ in range(9)]
           u[0][:]=.9;u[1][:]=a;u[2][:]=b;u[5][:]=G+stage*.07*np.sin(r);u[6][:]=1.;u[7][:]=slope*stage*r
           # Poison stored state/diagnostics: correct stage code must ignore both.
           for arr in damp.u: arr[:]=123.
           for arr in damp.C: arr[:]=456.
           d0=[np.zeros_like(r) for _ in range(9)];d1=[np.zeros_like(r) for _ in range(9)]
           base.rhs(d0,u,g);damp.rhs(d1,u,g)
           C=(.1*G if filtered else 0.)+stage*.07*np.sin(r)
           div=(.9 if filtered else 1.)*slope*stage*(1+r*(ap/(2*a)+bp/b))
           want=-strength*C*div
           err=[]
           for n in range(9):
            expected=want if n in (5,8) else np.zeros_like(r)
            delta=d1[n]-d0[n]
            np.testing.assert_allclose(delta[core],expected[core],atol=5e-12,rtol=1e-10)
            err.append(float(np.max(np.abs(delta[core]-expected[core]))))
           records.append({'cG':strength,'slope':slope,'stage':stage,'max_error_by_field':err,'min_radius_checked':float(np.min(np.abs(r[core])))})

    def test_raw_stage_rhs(self):
        self.check_rhs(False)

    def test_filtered_stage_rhs(self):
        self.check_rhs(True)

    def test_default_equals_explicit_zero(self):
        g=Grid1D({'Nx':101,'Xmin':-5.,'Xmax':5.,'NGhost':3},cell_centered=True)
        r=g.xi[0]
        g.set_D1(ExplicitFirst642_1D(g.dx[0]))
        g.set_D2(ExplicitSecond642_1D(g.dx[0]))
        a=1+.1*r*r; b=r*r*(1+.03*r*r)
        u=[np.zeros_like(r) for _ in range(9)]
        u[0][:]=.9; u[1][:]=a; u[2][:]=b
        u[5][:]=.2*r/(2*a*a)-(2*r+.12*r**3)/(a*b)+.07*np.sin(r)
        u[6][:]=1.; u[7][:]=.13*r
        results=[]
        for kwargs in ({}, {'c_G':0.}):
            eq=BSSN(g,1.,2.,True,**kwargs)
            out=[np.zeros_like(r) for _ in range(9)]
            eq.rhs(out,u,g)
            results.append(out)
        np.testing.assert_array_equal(results[0],results[1])

    def test_invalid_strength(self):
        g=Grid1D({'Nx':21,'Xmin':-5.,'Xmax':5.,'NGhost':3},cell_centered=True)
        for value in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError): BSSN(g,1.,2.,True,c_G=value)

if __name__ == '__main__': unittest.main()
