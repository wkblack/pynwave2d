"""Manufactured Gamma identities and explicitly measured diagnostic regressions."""
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bssn'))
from gamma import gamma_constraint, shift_divergence, reference_shift_divergence, diagnostic_summary


class GammaTests(unittest.TestCase):
    def test_flat_connection_not_zero_but_constraint_zero(self):
        r = np.array([-3., -.1, .1, 3.])
        one = np.ones_like(r)
        np.testing.assert_allclose(gamma_constraint(one, r*r, -2/r, 0*one, 2*r), 0., atol=1e-14)
        for slope in (-.3, 0., .3):
            np.testing.assert_allclose(shift_divergence(one, r*r, slope*r, 0*one, 2*r, slope*one), 3*slope)

    def test_nontrivial_metric_signed_constraint_including_puncture_region(self):
        r = np.array([-2., -.1, .1, 2.])
        a, b = 1+.1*r*r, r*r*(1+.03*r*r)
        da, db = .2*r, 2*r+.12*r**3
        wanted = .07*np.sin(r)
        Gamma = da/(2*a*a)-db/(a*b)+wanted
        C = gamma_constraint(a,b,Gamma,da,db)
        np.testing.assert_allclose(C, wanted, atol=2e-14)
        np.testing.assert_allclose(C, -C[::-1], atol=2e-14)
        for slope in (-.13,.13):
            div = shift_divergence(a,b,slope*r,da,db,np.full_like(r,slope))
            expected = slope*(1+r*(da/(2*a)+db/b))
            np.testing.assert_allclose(div,expected)
            np.testing.assert_allclose(-.7*C*div, -(-.7*C[::-1]*div[::-1]))

    def test_reference_divergence_ignores_determinant_drift(self):
        r = np.array([-2., -.025, .025, 2.])
        beta = .13*r + .02*r**3
        dbeta = .13 + .06*r*r
        div = reference_shift_divergence(beta, dbeta, r)
        np.testing.assert_allclose(div, .39 + .10*r*r)
        a, b = 1+.1*r*r, r*r*(1+.03*r*r)
        metric_div = shift_divergence(a, b, beta, .2*r, 2*r+.12*r**3, dbeta)
        self.assertGreater(np.max(np.abs(div-metric_div)), .01)
        np.testing.assert_allclose(div, div[::-1])
        with self.assertRaisesRegex(ValueError, "excluding r=0"):
            reference_shift_divergence(np.array([0.]), np.array([.13]), np.array([0.]))

    def test_diagnostics_measure_and_health_no_clipping(self):
        r=np.array([-3.,-2.,-1.,1.,2.,3.])
        a=np.ones_like(r); b=r*r; beta=.2*r
        u=[a.copy(),a,b,a*0,a*0,-2/r+.5,a,beta,a*0]
        derivatives={id(a):a*0,id(b):2*r,id(beta):a*.2}
        g=SimpleNamespace(xi=[r],dx=[1.],nghost=1,D1=SimpleNamespace(grad=lambda x:derivatives[id(x)]))
        u[0][2]=-.01
        row=diagnostic_summary(u,g,.5,.1,1,2)
        self.assertAlmostEqual(row['C_signed_rms_dr_shell'],.5)
        self.assertAlmostEqual(row['C_signed_l2_dr_shell'],1.)
        self.assertAlmostEqual(row['chi_min_physical'],-.01)
        self.assertEqual(row['chi_nonpositive_count_physical'],1)
        self.assertAlmostEqual(row['divbeta_min_physical'],.6)
        self.assertAlmostEqual(row['cG_divbeta_dt_max_physical'],.03)
        for shell in ((2,1),(-1,2),(5,6),(0,np.nan)):
            with self.assertRaises(ValueError): diagnostic_summary(u,g,.5,.1,*shell)

if __name__ == '__main__': unittest.main()
