# JP6_REPAIRED (opt-in, experimental)

Select `D1 = "JP6_REPAIRED"`, `D2 = "JP6_REPAIRED"`,
`DerivSolveMethod = "LUSOLVE"` and `BHMask = false` in the BSSN configuration.
No operator or tail defaults change. Original `JP6` is retained unchanged, including its known boundary
consistency defect. This is distinct from the separately selectable `JP6_JT4` option
(fourth-order boundary / sixth-order interior).

## Definition and support

This pair is the frozen **resolved_v1 D1 + multigrid29 D2** sixth-order boundary
repair, with original JP6 compact interiors. It is independently derived and
optimized, not a claim to reproduce canonical published JP6 boundary rows.
Both derivatives require at least **56 allocated points**, including ghost
cells, on a finite increasing uniform grid. Boundary support is not truncated
on small grids: construction raises `ValueError` for N < 56.

**Unmasked LUSOLVE** is the optimized path. Unmasked `DerivSolveMethod = "SCIPY"`
is also available as a generic wide-banded reference backend with the same P/Q,
not as a performance alternative. BH blending, PENTAPY, D_INV, D_LU and NONE are
explicitly rejected. No silent lower-order fallback or unqualified
masked operation is supplied. Input vectors must be finite real float64.

`NCompactDerivative` retains the actual candidate banded `P` and `Qb`, lazy
full `Q`, and truthful method metadata. Under LUSOLVE, its `solver_kind`
identifies the cached structured application: rank-eight Woodbury correction over the original pentadiagonal LU
for D1, or four-boundary-variable Schur elimination over a pentadiagonal interior
LU for D2. There is no hidden N-by-N setup or unused wide-band LU factorization.
The new correction kernels preserve the frozen non-fastmath arithmetic and
do not drop small coefficients. The original D1 base multiplication and the
generic SCIPY path retain the repository's existing fastmath banded matvec.
Floating results across other exact solve algorithms need not be
bitwise identical. SCIPY reports `scipy-wide-banded-reference`, uses the existing
generic banded solve, and does not install a structured solver or factor wide LU
at construction. Its repeated applications are slower; selecting LUSOLVE never
silently switches to this backend.

## Provenance and tests

Frozen research installer SHA256:
`51407c6b953a82f5e6f43b456bbb345cf9b8cf3d0e12f0c0d3a7b1d24d9c32d3`.
D1 exact-rational JSON SHA256:
`3373e6826c6e57ef77179678dc3192fcb0f92719658442464beb62e7d9142d1e`.
D2 parameter JSON SHA256:
`896e272184061e692d132f645cff3a00af1f98e19da06e415bf78a245474682d`.
D2 generator SHA256:
`2baebba29c68b746fd174e853927d506a2b9f0ce3edd0d80846f8a033137defb`.

Runtime imports literal binary64 coefficients, not SymPy. Exact rational fixtures
in `tests/fixtures/jp6_repaired_exact.json` independently verify D1 polynomial
moments through degree 6 and D2 through degree 7. D2 boundary equations retain
the research normalization before conversion into runtime literal data. Tests
bind these literals to the rational fixtures, include all allocated rows (not
just the physical interval), reflection, resolved smooth refinement, banded
references/residuals, minimum size, unsupported modes and sparse setup.

## Scope of evidence and limitations

The frozen pair passed the tested matched scalar boundary problems and short
nonlinear cases. With KO6 sigma 0.5 it remained finite through 50M at both tested
resolutions with tails off/on, where original JP6 with the tail failed earlier.
Early negative chi remains shared: finite completion is not a positivity or
physical-admissibility certificate. Tail-off late Gamma residual is roughly
11–12% above inconsistent original JP6, but comparable to clean JT4/E642 controls.
There is no claim of a universal stability theorem, improvement at every time,
frequency or boundary condition, or bitwise equality across different solvers.
The 56-point closure has a substantial pre-asymptotic region; sixth-order moment
identities do not guarantee clean sixth-order rates on under-resolved wavelengths.

### Runtime comparisons: different denominators

The historical approximately 0–5% median whole-solver overhead compared the
optimized research pair with **original JP6**, including its known boundary
consistency defect. It is not a timing claim against clean JT operators.

In matched warm production workloads (N800/1600, R20, 1M horizon, CFL 0.1,
KO6 sigma 0, tail off/on), the repaired pair is **13–21% slower than JT6 D1
with the same repaired D2**, **20–27% slower than JT6/JT4**, and **31–37% slower
than JT4/JT4**. These are median paired whole-solver ratios, not a universal
latency bound. Different accuracy and boundary behavior must be considered
alongside speed; this pair is not presented as the fastest option.

A separate nine-pair production-versus-frozen-optimized-research check measures
only the cost of the normal production API and guards. Its median ratios range
from 0.989 to 1.020; all paired bootstrap 95% upper bounds are below 1.10 in the
tested blocks (coarse tail-on upper bound 1.0991). This incremental result must
not be multiplied with selected historical timings and described as a newly
measured cumulative confidence interval.

Measured environment: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and Numba 0.68.1.
No compatibility or performance guarantee is inferred for untested versions,
machines or configurations. The unmasked operator remains opt-in, N >= 56;
`BHMask = false` is required. Experimental masked support is a separate branch,
not a new default or part of this candidate.

### Targeted integration and gauge-pulse checks

A separate local integration with the current PR10–13 heads passes 81 tests.
NGhost 2/3/4 tail-enabled runs reach 2M with finite constraints and positive
saved physical chi/metric. The main-based operator remains independent of those
unmerged PRs; retest the eventual merged base.

With a parity-correct lapse/shift pulse that reaches the boundary, the final
pair shows expected RK4 temporal refinement at CFL 0.05/0.1/0.2 over the common
healthy interval. Fine N1600 pulse/control runs reach 8M with positive observed
RK states. Coarse N800 pulse and unperturbed controls fail around 7.3–7.6M for
both repaired and clean JT6-reference methods; reducing CFL does not remove
that failure. Both methods develop sizeable shared boundary constraints upon
pulse arrival, without improvement across the two tested resolutions. The
repaired closure has larger early smooth-signal residuals. These tests do not
certify constraint-preserving gauge boundaries or a universal CFL limit.

An origin-safe analytic flat-space RHS fixture passes to roundoff with exact
derivatives. Numerical LU/SCIPY checks satisfy the same P/Q equations, retain
floating cancellation floors, and show approach toward sixth-order derivative
convergence on the tested reciprocal. This is a consistency fixture, not a
flat-space evolution stability proof; the Mass0 TOML policy is unchanged.

## Choosing between the two opt-in alternatives

| Selector | Boundary / interior order | Application | Scope |
|---|---|---|---|
| `JP6_REPAIRED` | sixth / sixth | structured LUSOLVE or generic SCIPY | unmasked; at least 56 allocated points |
| `JP6_JT4` | fourth / sixth | existing compact backend machinery | joint JT4 first two boundary rows, original JP6 interior |

For the hybrid, set both `D1` and `D2` to `"JP6_JT4"`. LU/SciPy and
existing masked factory paths have regression coverage; that does not certify
every possible mask, companion or PDE boundary treatment. In particular,
legacy companion-operator defects are not repaired by selecting a new background.
See [the hybrid validation notes](../tests/JP6_JT4_VALIDATION.md).

The two alternatives preserve their separately tested coefficients. Neither
selector replaces original `JP6`, changes existing enum values, or becomes
the default. The fourth-order hybrid trades boundary order for a simpler
closure; do not infer that it has the same accuracy, cost, or stability
properties as the sixth-order repaired pair. Experimental masked support
for `JP6_REPAIRED` is not included.
