"""JP6 one-sided boundary-adjacent consistency and smooth refinement tests."""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nwave import NCompactDerivative, DerivType, CFDSolve
class JP6Closure(unittest.TestCase):
    def test_polynomial_consistency(self):
        x=np.linspace(-1,1,65)
        for degree,kind in [(1,DerivType.D1_JP6),(2,DerivType.D2_JP6)]:
            op=NCompactDerivative.deriv(x,kind,CFDSolve.SCIPY)
            for p in range(degree,degree+6):
                with self.subTest(derivative=degree,power=p):
                    exact=p*x**(p-1) if degree==1 else p*(p-1)*x**(p-2)
                    np.testing.assert_allclose(op.grad(x**p),exact,rtol=0,atol=1e-8)
    def test_exponential_refinement(self):
        for kind in [DerivType.D1_JP6,DerivType.D2_JP6]:
            errors=[]
            for n in [17,33,65]:
                x=np.linspace(-1,1,n);op=NCompactDerivative.deriv(x,kind,CFDSolve.SCIPY)
                errors.append(np.max(abs(op.grad(np.exp(x))-np.exp(x))))
            with self.subTest(kind=kind):
                self.assertGreater(errors[0]/errors[1],40)
                self.assertGreater(errors[1]/errors[2],40)
if __name__=='__main__':unittest.main()
