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
    def test_solver_recovers_and_reports_mask(self):
        with tempfile.TemporaryDirectory() as d:
            par=Path(d)/"params.toml"; par.write_text(CONFIG+f'output_dir="{d}/out"\n')
            code="""import sys,builtins,typing,json,numpy as np
builtins.Any=typing.Any
sys.path[:0]=[sys.argv[1]+'/bssn',sys.argv[1]]
import solver
calls=[]
def find(self,g,guess):
 calls.append(float(guess)); n=len(calls)
 radius=.5 if n==1 else None if n==2 else .51 if guess>0 else None
 return radius,None if radius is None else 1.,np.zeros_like(g.xi[0])
solver.bssn.BSSN.find_horizon=find
solver.main(sys.argv[2]);print('GUESSES='+json.dumps(calls))
"""
            p=subprocess.run([sys.executable,'-c',code,str(ROOT),str(par)],capture_output=True,text=True,timeout=60)
            self.assertEqual(p.returncode,0,p.stderr)
            guesses=json.loads(p.stdout.split('GUESSES=')[-1]);self.assertEqual(guesses,[.7,.5,.5])
            with (Path(d)/'out_0/bssn_constraints.dat').open() as f: rows=list(csv.DictReader(f))
            self.assertEqual([r['horizon_found'] for r in rows],['True','False','True'])
            self.assertEqual([float(r['mask_inner_radius']) for r in rows],[.5,.5,.51])
            self.assertEqual([float(r['radius_bh']) for r in rows],[.5,0.,.51])

if __name__=="__main__": unittest.main()
