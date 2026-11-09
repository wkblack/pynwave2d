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
    def test_masked_writer(self):
        from types import SimpleNamespace
        e=SimpleNamespace(u_names=['u'],c_names=['C'],u=[np.array([2.,4.,6.])],C=[np.array([1.,3.,5.])])
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'a.curve'; util.write_curve_masked(f,0.,np.array([10.,11.,12.]),e,np.array([1.,0.,1.]))
            data=f.read_text(); self.assertIn('1.10000000e+01 0.00000000e+00',data)
            self.assertIn('1.20000000e+01 5.00000000e+00',data)

    def test_nonfinite_every_field(self):
        for j in range(9):
            for bad in (np.nan,np.inf,-np.inf):
                u=[np.ones(10) for _ in range(9)];u[j][4]=bad
                with self.assertRaises(FloatingPointError):solver.require_finite_state(u,7)
        solver.require_finite_state([np.ones(10) for _ in range(9)],7)
        code="import sys,builtins,typing;builtins.Any=typing.Any;sys.path[:0]="+repr([str(ROOT/'bssn'),str(ROOT)])+";import solver,numpy as np;solver.require_finite_state([np.array([np.inf])],7)"
        p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0)
    def test_solver_rejects_injected_inf(self):
        with tempfile.TemporaryDirectory() as d:
            par=Path(d)/"params.toml"
            # No diagnostic interval occurs before the sole injected step.
            par.write_text(CONFIG.replace("Nt=2", "Nt=1").replace("interval=1", "interval=10")+f'output_dir="{d}/out"\n')
            code="""import sys,builtins,typing,numpy as np
builtins.Any=typing.Any
sys.path[:0]=[sys.argv[1]+'/bssn',sys.argv[1]]
import solver
class BadStep:
 def __init__(self,e,g): pass
 def step(self,e,g,dt): e.u[8][10]=np.inf
solver.RK4=BadStep
solver.main(sys.argv[2])
"""
            p=subprocess.run([sys.executable,'-c',code,str(ROOT),str(par)],capture_output=True,text=True,timeout=60)
            self.assertNotEqual(p.returncode,0,p.stdout)
            self.assertIn('Non-finite evolved field 8 at step 1',p.stderr)

if __name__=="__main__": unittest.main()
