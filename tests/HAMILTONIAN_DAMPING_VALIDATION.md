# Reference-metric Hamiltonian damping: experimental, default off

Base: `f52cf8f27ed518ef284dcb4c5a1c854bef014b6c` (`experiment/gamma-damping`).
A live read of all upstream open PRs and wkblack fork branches on 2026-10-30
found no H/G branch overlap; PR9 gauge-registry remained `eae993cb`.
This isolated experiment does **not** integrate PR9: compatibility is untested.
No publication or merge is claimed.

## Definition

The supplied CAHD notes define the **NRPy reference-metric** BSSN Hamiltonian,
not the existing masked geometric diagnostic. With a=g_rr and b=g_theta_theta
(including r²), define

```
C = Gamma - a'/(2*a²) + b'/(a*b)
Gamma_metric' = a''/(2*a²) - a'²/a³ - b''/(a*b)
                + a'*b'/(a²*b) + b'²/(a*b²)
H_B_ref = H_g + chi * (Gamma' - Gamma_metric' + 2*C/r)
chi_rhs += 4*c_H*(h/CFL)*chi*H_B_ref
```

The 2C/r term is the spherical reference divergence. It is required by the
attachment's reference-metric convention. **Native pynwave's trace-free Ricci
uses a raw-coordinate connection convention**, whose off-constraint difference
is not the same. We leave that original evolution unchanged and label the new
H_B_ref explicitly. H_B_ref and H_g agree on the continuum connection-constraint
surface; discrete derivatives need not satisfy continuum chain identities.

H_g uses the existing unmasked geometric formula. No horizon/puncture mask,
chi floor, clipping, extra alpha factor, M damping or changes to the other RHS
fields are introduced. h is the actual `g.dx[0]`; CFL is the configured `cfl`,
not a hidden wave-speed multiplier. Solver dt=CFL*h.

## Interface

Optional TOML `c_H = 0.0` defaults off. An exploratory on-value is `.016`, not
a validated safe recommendation. `c_G` remains independent. BSSN construction
accepts `c_H=0.0,cfl=None`; active H damping requires positive finite `cfl`.
c_H must be finite/nonnegative. No claim that the notes' 3D bound applies to
this 1D coupled system.

RHS uses current RK-stage derivatives under the existing derivative-filter
policy and native D2 / composed-D1 setting. It never reads stored self.u/self.C.
The exact zero branch leaves archived Gamma-only RHS arithmetic unchanged.

New separate `hamiltonian_diagnostics.dat` records H_g and H_B_ref RMS/L2/Linf
on the same fixed coordinate-dr shell as Gamma diagnostics; no AH mask moves
that shell. `hamiltonian_NNNN.curve` stores `H_g_raw`,`H_B_ref_raw` without masks.
These diagnostics use raw **unfiltered** states, intentionally distinguishable
from the RHS source when derivative filtering is enabled.
`H_diffusion_step_*_physical = 8*c_H*chi/a` is the local principal parabolic
coefficient times dt/h² under dt=CFL*h; multiply by a dimensionless discrete
Laplacian symbol for a mode's negative-real RK argument. This is not a full
coupled or boundary stability bound. Negative/nonfinite fields remain visible.
Legacy output formats are unchanged.

## Tests before physical trials

`python -m unittest discover -s tests -p test_hamiltonian_damping.py -v`
passes six tests, covering:

- flat spherical C=0 and injected C=eps*r, giving H_B_ref-H_g=3*chi*eps;
- nontrivial metric on-constraint equality and sinusoidal injected C;
- chi conversion factor four, positive sign, h/CFL scaling and CFL-independent
  fixed-h per-step increment;
- eight actual RHS cases over c_G, CFL and distinct stages, stored state and
  constraints poisoned with NaNs, unmasked near-puncture source;
- derivative filtering with both native D2 and D1D1: only chi RHS changes;
- parameter rejection and backward-compatible off constructor.

Separate archived-source test: c_G=.5 nine-field RHS saved from frozen f52cf8f
and new default H0 agree **bitwise**. Evidence outside the repo is in
`../hgamma-validation/zero_gamma_guard.py`, `before.npy`, `after.npy`, `zero.log`.

The catch-all `unittest discover` also loads legacy script test_derivs2D.py
which depends on a cwd-relative params2D.json and fails from workspace root.
This is a test harness issue, not a new numerical failure; focused regression
modules avoid treating the legacy plotting/script entry points as unit tests.

Independent mathematical review: `../h-damping-derivation/DERIVATION.md`.
Physical comparison results are recorded separately; implementation tests do
not establish that H+G improves over H alone.

## Completed bounded trial screen (2026-10-30)

Full 14-run report and reproducibility artifacts: `../hgamma-trials/REPORT.md`,
`source_manifest.json`, `run_receipts.json`, `summary.json`, `pairs.json`,
`signed_profiles.json`, and `matched_5M.png`. Physical runs used implementation
`d3d189e`, E642 D1/D2, Nx=800, [-20,20], CFL=.1, fixed 1<|r|<10 shell.
The primary exploratory regime was KO6 sigma=0; sigma=.5 was a separate screen.
No further code change or coefficient optimization is implied by this summary.

At 5M, H=.016 plus G=.5 versus H=.016 alone lowered H_g by **7.15%**, H_B_ref
by **9.42%**, and Gamma magnitude by **2.73%**, while **raising final M by 8.56%**.
Chi remained positive at every saved sample (cadence .2M; not proof between
samples). Ratios of time-integrated **squared** L2 norms over 0–5M were
H_g `.923749`, H_B_ref `.792281`, M `.986643`, Gamma magnitude `.930567`.
The slight integrated-M reduction does not erase the final-M penalty.
G=1 lost chi positivity at the saved 5M endpoint and is not a healthy candidate.

The off-control H/H_B_ref developed a sharp rise after approximately 4.4M,
which drives the large apparent 5M improvement versus off. Earlier H_B_ref
was often worse with H adjustment: at 2M H=.016/off gave H_g `.565137` but
H_B_ref `2.099080`. Signed profiles expose strong cancellation between H_g
and H_B_ref-H_g, so the two Hamiltonians must not be conflated. The shorter
H=.008 check showed the same qualitative tradeoff, not an optimized solution.
All three KO=.5 trials had negative chi by the first saved .2M sample;
improved exterior-shell norms do not repair that puncture defect.

H0 reproduced all 26 archived pre-H 5M legacy snapshots and Gamma CSV byte
for byte. Independent final mathematical review and six H tests passed;
all 37 focused regression tests passed. Archived nine-field RHS equality was
also checked bitwise at both G0 and G=.5.

These data do **not** establish a universal H+G improvement or a healthy
production regime. This is one resolution and a bounded duration, not a
convergence or long-time stability study. H_B_ref remains the explicit
reference-metric adjustment added to frozen native raw-coordinate TF Ricci;
old gauge API is retained and PR9 compatibility remains untested. Defaults
remain off, the branch remains local/unpublished, and no M damping was added.
