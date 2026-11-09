from pathlib import Path
import sys, os, unittest
import numpy as np
root=Path(os.environ.get('SOURCE', Path(__file__).resolve().parents[1])).resolve()
sys.path[:0]=[str(root),str(root/'bssn')]
from bssneqs import BSSN
class Boundary(unittest.TestCase):
 def apply(self,out,u,du,r,v,ext=True):
  if hasattr(BSSN,'bc_spherical_gtt'): BSSN.bc_spherical_gtt(out,u,du,r,v,2,ext)
  else:BSSN.bc_sommerfeld(out,u,du,r,-2,1.,v,2,ext)
 def test_flat_background(self):
  r=np.r_[np.linspace(-21.,-20.,4),np.linspace(20.,21.,4)];out=np.zeros_like(r)
  self.apply(out,r*r,2*r,r,1.)
  np.testing.assert_allclose(out,0.,atol=1e-13)
 def test_outgoing_normalized_perturbation(self):
  r=np.r_[np.linspace(-21.,-20.,4),np.linspace(20.,21.,4)];R=abs(r)
  ix=[0,1,6,7]
  for v in (1.,np.sqrt(2)):
   f=np.sin(.37*(R-v*.2));df=.37*np.cos(.37*(R-v*.2))
   u=r*r+R*f;du=2*r+np.sign(r)*(f+R*df);expected=-v*R*df
   out=np.full_like(r,123.);self.apply(out,u,du,r,v)
   np.testing.assert_allclose(out[ix],expected[ix],rtol=1e-12,atol=1e-12)
   np.testing.assert_array_equal(out[2:-2],123.)
 def test_positive_domain(self):
  r=np.linspace(20.,21.,8);out=np.full_like(r,123.)
  self.apply(out,r*r,2*r,r,1.,False)
  np.testing.assert_array_equal(out[:-2],123.);np.testing.assert_allclose(out[-2:],0.,atol=1e-13)
if __name__=='__main__':unittest.main()
