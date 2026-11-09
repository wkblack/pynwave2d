"""Independent regression tests; BSSN_TEST_ROOT selects an unmodified baseline."""
import os, sys, tempfile, subprocess, unittest, builtins, typing, csv, json
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get("BSSN_TEST_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT/'bssn'), str(ROOT)]
# Other independent PR branches still have the upstream missing Any import.
# The separate upstream-cleanups PR supplies that import; this PR does not duplicate it.
builtins.Any = typing.Any
import bssneqs as b
import bssnutils as util
import solver
from nwave import Grid1D

CONFIG = """Nx=80
Nt=2
NGhost=3
Xmin=-20.0
Xmax=20.0
cfl=.1
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
""".replace("cfl=.1", "cfl=0.1")

class RuntimeTests(unittest.TestCase):
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

if __name__=="__main__": unittest.main()
