"""The new named variant must not silently redefine the old JP6 selector."""
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'bssn'),str(ROOT)]
import bssnutils
from nwave import DerivType, CFDSolve
from nwave.cd import _coeffs_D1_JTP6, _coeffs_D2_JTP6, _coeffs_D1_JP6_JT4, _coeffs_D2_JP6_JT4


class JP6JT4SelectionTests(unittest.TestCase):
    def test_existing_ids_and_selectors_are_preserved(self):
        self.assertEqual(DerivType.D1_JP6.value,5)
        self.assertEqual(DerivType.D2_JP6.value,6)
        self.assertEqual(bssnutils.get_d1_type('JP6'),DerivType.D1_JP6)
        self.assertEqual(bssnutils.get_d2_type('JP6'),DerivType.D2_JP6)
        self.assertEqual(bssnutils.get_d1_type('JP6_JT4'),DerivType.D1_JP6_JT4)
        self.assertEqual(bssnutils.get_d2_type('JP6_JT4'),DerivType.D2_JP6_JT4)

    def test_original_jp6_coefficients_unchanged(self):
        # Preserve old behavior, including its known closure defect; the new
        # method is an explicit opt-in selector rather than a hidden change.
        for order,old,new in [(1,_coeffs_D1_JTP6,_coeffs_D1_JP6_JT4),(2,_coeffs_D2_JTP6,_coeffs_D2_JP6_JT4)]:
            if order==1:
                a,b,c=17/57,-1/114,30/19
                p0,p1=[1.,8.,6.],[a,1.,a,b]
                q0,q1=[-43/12,-20/3,9.,4/3,-1/12],[-c/2,0.,c/2]
            else:
                a,b,c=12/97,-1/194,120/97
                p0,p1=[1.,11/2,-131/4],[a,1.,a,b]
                q0,q1=[177/16,-507/8,783/8,-201/4,81/16,-3/8],[c,-2*c,c]
            P,Q=old();PN,QN=new()
            for actual,expected in [(P[0],[b,a,1.,a,b]),(Q[0],q1),(P[1][0],p0),(P[1][1],p1),(Q[1][0],q0),(Q[1][1],q1)]:
                np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(PN[0],P[0]);np.testing.assert_array_equal(QN[0],Q[0])
            # Calling the variant must not mutate later original calls.
            afterP,afterQ=old()
            np.testing.assert_array_equal(afterP[1][0],p0);np.testing.assert_array_equal(afterQ[1][0],q0)

    def test_parameter_factory_dispatch_and_masks(self):
        x=np.linspace(-2.,2.,97)
        for backend in ('LUSOLVE','SCIPY'):
            for mask in (False,True):
                selection=['JP6_JT4','ME642'] if mask else 'JP6_JT4'
                p={'D1':selection,'D2':selection,'DerivSolveMethod':backend,'BHMask':mask,'BHMaskPos':0.,'BHMaskWidth':.05}
                D,DD=bssnutils.init_derivative_operators(x,p)
                self.assertEqual(D.dtype,DerivType.D1_JP6_JT4)
                self.assertEqual(DD.dtype,DerivType.D2_JP6_JT4)
                self.assertEqual(D.method,getattr(CFDSolve,backend))
                self.assertEqual(DD.method,getattr(CFDSolve,backend))
                np.testing.assert_allclose(D.grad(x),1.,atol=1e-10,rtol=0.)
                np.testing.assert_allclose(DD.grad(x*x),2.,atol=1e-8,rtol=0.)


if __name__=='__main__':
    unittest.main()
