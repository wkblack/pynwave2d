# BSSN boundary and constraint-ghost regressions

Run from the repository root with NumPy, SciPy, Numba and pentapy available:

```sh
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python -m unittest discover -s tests -p test_bssn_boundaries.py -v
```

The tests check the exact outgoing field
`u - u_inf = F(|r| - v*t) / |r|**n`, for speeds 1 and sqrt(2),
falloff powers 0, 1, 2, and -2, at both boundaries. They also check the
positive-only-domain behavior, unchanged per-field speeds, all constraint
ghost cells for NGhost = 2, 3, 4, and preserved interior values/masked norms.
`BSSN_SOURCE_DIR=/absolute/source/tree` runs the same regressions against an
archived source version. No `bssnutils` import or compatibility shim is needed
for these unit tests.

## Short matched evolution comparison

Compared upstream `1063d45`, each fix separately, and both fixes together.
All four runs used the same puncture initial data, collapsed lapse, mass 1,
eta 2, domain [-20, 20], Nx 800, NGhost 3, CFL .1, E642 first/second
operators, BHMask false, KO6 RHS dissipation sigma .5 with boundary filtering
disabled, 400 steps (t = 2), output interval 400, print interval 20, and
`bhmask_outer_radius = 15`. All runs ended normally with finite saved values.
The resulting cell-centered arrays contain 808 points including ghosts.
All four runs also emit the same `invalid value encountered in sqrt` warning
from the horizon-finder calculation of `1/sqrt(chi[i])`. Finite saved arrays
do not imply positive chi everywhere or trustworthy horizon diagnostics;
this shared warning is not repaired or explained by these boundary fixes.

The untouched solver could not import `bssnutils` because its `Any` annotation
has no import. All four evolutions therefore used the **same test-only shim**:
`builtins.Any = typing.Any` before loading `solver.py` with `runpy.run_path`.
This supplies only the missing annotation name; no numerical source or
settings are changed by the shim. The unrelated missing import is not fixed
in this change. The direct-launch failure was retained in validation logs.

Maximum parity residuals in final saved `.curve` values (even alpha/H,
odd M/Gamma constraint):

| Variant | alpha | H | M | Gamma constraint |
|---|---:|---:|---:|---:|
| Upstream | .007254715 | .419538124 | .018359944 | .0718465883 |
| Boundary only | 1.0e-9 | .179341686 | .00176675466 | .0718465883 |
| Ghost only | .007254715 | .419538124 | .018359944 | .005871181 |
| Both | 1.0e-9 | 2.121e-6 | 1.1115e-6 | 2.277e-7 |

The final rightmost H/M/Gamma constraint values changed from
(-.179341686, .00176675466, -.0718465883) to exactly zero with the ghost fix.
Both pairs with/without the ghost fix have exactly identical saved evolved
fields and non-ghost constraint values. The ghost correction is diagnostic,
not a change to the evolution.

All four solver-reported masked H/M norms and horizon diagnostics are
identical: final H = .0137539349705, M = .0216507771821, horizon radius
.646038544168, mass 1.00033851863. These norms exclude the outer boundary.
This is a short symmetry/implementation check, **not** evidence of improved
interior accuracy, convergence, or long-time stability. In particular,
the full-array H norm including the outer region rises from .476638 to
.549874 for both fixes; the corrected outgoing law restores symmetry but
does not eliminate boundary truncation residuals.

The four unit-test methods pass after both fixes. Against the upstream tree,
there are six outgoing-field subtest failures and three ghost-mask subtest
failures. Boundary-only leaves the three ghost failures; ghost-only leaves
the six outgoing-field failures. Thus each regression isolates its fix.
