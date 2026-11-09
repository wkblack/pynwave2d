"""Manufactured reference-Ricci, scaling, and fresh-stage CAHD regression tests."""
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'bssn'), str(Path(__file__).resolve().parents[1])]
from hamiltonian import hamiltonians, chi_adjustment, diagnostic_fields
from bssneqs import BSSN
from nwave import Grid1D, ExplicitFirst642_1D, ExplicitSecond642_1D, FilterApply


class HamiltonianTests(unittest.TestCase):
    def test_reference_divergence_flat_manufactured(self):
        r=np.array([-3.,-.1,.1,3.]); one=np.ones_like(r); zero=0*one
        for eps in (0., .07):
            hg,hb=hamiltonians(one,one,r*r,zero,zero,-2/r+eps*r,
                              zero,zero,2*r,zero,zero,2*one,2/r**2+eps,r)
            np.testing.assert_allclose(hg,0,atol=3e-13)
            np.testing.assert_allclose(hb,3*eps,atol=3e-13)

    def test_nontrivial_metric_connection_surface_and_injection(self):
        r=np.array([-2.,-.1,.1,2.]); a=1+.1*r*r;b=r*r*(1+.03*r*r)
        da=.2*r;db=2*r+.12*r**3;d2a=np.full_like(r,.2);d2b=2+.36*r*r
        gm=da/(2*a*a)-db/(a*b)
        dgm=d2a/(2*a*a)-da**2/a**3-d2b/(a*b)+da*db/(a*a*b)+db**2/(a*b*b)
        chi=.7+.02*r*r
        args=[chi,a,b,.01*r,.03*r,gm,.04*r,da,db,np.full_like(r,.04),d2a,d2b,dgm,r]
        hg,hb=hamiltonians(*args)
        np.testing.assert_allclose(hb,hg,atol=1e-13)
        args[5]=gm+.07*np.sin(r);args[12]=dgm+.07*np.cos(r)
        hg2,hb2=hamiltonians(*args)
        np.testing.assert_array_equal(hg,hg2)
        np.testing.assert_allclose(hb2-hg2,chi*.07*(np.cos(r)+2*np.sin(r)/r),atol=1e-13)

    def test_conversion_scaling_and_no_alpha(self):
        chi=np.array([.3,.8]);hb=np.array([-2.,3.])
        base=chi_adjustment(chi,hb,.016,.05,.1)
        np.testing.assert_allclose(base,.032*chi*hb)
        np.testing.assert_array_equal(chi_adjustment(chi,hb,.016,.10,.1),2*base)
        np.testing.assert_array_equal(chi_adjustment(chi,hb,.016,.05,.2),base/2)
        # dt scales with CFL*h: fixed-h per-step increment independent of CFL.
        np.testing.assert_allclose(base*.005,chi_adjustment(chi,hb,.016,.05,.2)*.01)

    def test_actual_rhs_stage_poison_and_only_chi(self):
        g=Grid1D({'Nx':101,'Xmin':-5.,'Xmax':5.,'NGhost':3},cell_centered=True)
        r=g.xi[0];h=g.dx[0];g.set_D1(ExplicitFirst642_1D(h));g.set_D2(ExplicitSecond642_1D(h))
        core=np.abs(r)<4.5
        for cG in (0.,.5):
            base=BSSN(g,1.,2.,True,c_G=cG)
            for cfl in (.1,.2):
                damp=BSSN(g,1.,2.,True,c_G=cG,c_H=.016,cfl=cfl)
                for stage in (1.,1.4):
                    u=[np.zeros_like(r) for _ in range(9)]
                    u[0][:]=.8+.01*r*r;u[1][:]=1+.1*r*r;u[2][:]=r*r*(1+.03*r*r)
                    u[3][:]=.002*np.cos(r);u[4][:]=.003*np.cos(r)
                    u[5][:]=-.5/r+stage*.02*np.sin(r);u[6][:]=.7+.01*np.cos(r);u[7][:]=stage*.04*r
                    for x in damp.u+damp.C: x[:]=np.nan
                    d0=[np.zeros_like(r) for _ in u];d1=[x.copy() for x in d0]
                    base.rhs(d0,u,g);damp.rhs(d1,u,g)
                    hg,hb=diagnostic_fields(u,g)
                    want=4*.016*h/cfl*u[0]*hb
                    np.testing.assert_allclose((d1[0]-d0[0])[core],want[core],atol=2e-12,rtol=1e-11)
                    for n in range(1,9): np.testing.assert_array_equal(d1[n],d0[n])
                    self.assertTrue(np.any(np.abs(hb[abs(r)<.5])>0))

    def test_filtered_derivatives_and_composed_second_derivative(self):
        g=Grid1D({'Nx':101,'Xmin':-5.,'Xmax':5.,'NGhost':3},cell_centered=True)
        r=g.xi[0];h=g.dx[0];g.set_D1(ExplicitFirst642_1D(h));g.set_D2(ExplicitSecond642_1D(h))
        g.num_filters=1
        g.Filter=[SimpleNamespace(apply_filter=FilterApply.APPLY_DERIVS,filter=lambda x:.9*x)]
        u=[np.zeros_like(r) for _ in range(9)]
        u[0][:]=.8+.01*r*r;u[1][:]=1+.1*r*r;u[2][:]=r*r*(1+.03*r*r)
        u[5][:]=-.5/r+.02*np.sin(r);u[6][:]=.7;u[7][:]=.04*r
        for have_d2 in (False, True):
            base=BSSN(g,1.,2.,True,c_G=.5,have_d2=have_d2)
            damp=BSSN(g,1.,2.,True,c_G=.5,c_H=.016,cfl=.1,have_d2=have_d2)
            for x in damp.u+damp.C: x[:]=np.nan
            d0=[np.zeros_like(r) for _ in u];d1=[x.copy() for x in d0]
            base.rhs(d0,u,g);damp.rhs(d1,u,g)
            dc,da,db,dG=[g.D1.grad(.9*u[i]) for i in (0,1,2,5)]
            d2c,d2a,d2b=([g.D2.grad(.9*u[i]) for i in (0,1,2)] if have_d2
                         else [g.D1.grad(x) for x in (dc,da,db)])
            _,hb=hamiltonians(*u[:6],dc,da,db,d2c,d2a,d2b,dG,r)
            core=np.abs(r)<4.5
            np.testing.assert_allclose((d1[0]-d0[0])[core],(4*.016*h/.1*u[0]*hb)[core],atol=2e-12)
            for i in range(1,9): np.testing.assert_array_equal(d1[i],d0[i])

    def test_validation(self):
        g=Grid1D({'Nx':21,'Xmin':-2.,'Xmax':2.,'NGhost':3},cell_centered=True)
        for cH,cfl in ((-1.,.1),(np.nan,.1),(.016,None),(.016,0),(.016,np.inf)):
            with self.assertRaises(ValueError): BSSN(g,1.,2.,True,c_H=cH,cfl=cfl)
        BSSN(g,1.,2.,True)  # old constructor still works, no CFL required when off.

if __name__=='__main__': unittest.main()
