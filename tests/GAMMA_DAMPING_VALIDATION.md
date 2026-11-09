# Experimental Gamma diagnostics and CCD adjustment

## Frozen base / scope

Local `experiment/gamma-damping`, base `288e869`: integration-clean `77497d4`
plus PR #1 source/test boundary parity and ghost fixes (local cherry-picks
`9b58fe7`, `288e869`). No experimental BC, CFD/JP6, Vars filtering, or chi
floor. Upstream main checked 2026-10-30: `1063d45`; PR #9 gauge-registry head
`eae993cb` changes the gauge API and is deliberately NOT partially integrated.
Port and rerun gauge coupling tests when that independent PR is integrated.
The upstream missing `typing.Any` uses an identical test-only import shim in
controls and experiments; that unrelated source change belongs to PR #2.

## Interface and equations

Optional top-level TOML keys (all defaults shown):

```toml
c_G = 0.0
gamma_shell_inner = 1.0
gamma_shell_outer = 10.0
```

With `a=g_rr`, `b=g_theta_theta` (including r squared),

```
C = Gamma_evolved - a'/(2 a^2) + b'/(a b)
divbeta = beta' + 2 beta/r  # fixed reference determinant (NRPy convention)
delta(dtGamma) = -c_G C divbeta
```

This is a constraint adjustment, not universally positive damping: negative
`c_G*divbeta` is anti-damping. No added 2/3 correction: the base RHS already
uses the metric-computed Gamma in its undifferentiated shift couplings.
The adjustment is applied before the B-driver reads `dtGamma`. Boundary
conditions and RHS filters then act in their original order.

Current-stage data and existing RHS derivative policy are used: if derivative
filtering is enabled, derivatives are of filtered stage fields, with raw stage
metric factors for C, exactly as the other RHS terms. In divbeta, only beta'
is differentiated/filtered; 2 beta/r uses the raw stage beta. No `self.C` or `self.u` reuse,
inner mask, or positivity repair. `c_G == 0` bypasses the entire adjustment,
preserving original RHS arithmetic (including avoiding zero times NaN).

## New outputs (legacy formats untouched)

`gamma_diagnostics.dat`: initial and every print interval. Fixed shell
`inner <= abs(r) <= outer`, physical cells only, both mirrored halves counted.
Coordinate measure `dr`, NOT spherical volume or proper volume:

- Signed C and sqrt(a)*abs(C): mean, RMS=sqrt(mean(value^2)),
  L2=sqrt(dr sum(value^2)), and Linfinity on shell.
- Physical-cell extrema of chi, a, b, full spherical divbeta, c_G divbeta dt;
  count chi<=0, state/diagnostic finite flags, anti-damping cell fraction.
- Explicit coefficient, shell edges, cell count, dr in every row.

`gamma_STEP.curve`: initial and normal output interval; signed raw C, scalar
magnitude, divbeta on ALL grid points including ghosts, no hidden masks.
Diagnostic derivatives use D1 on raw evolved state (not derivative-filtered
surrogates); this distinction is intentional and relevant when using derivative
filters. Negative a yields NaN scalar magnitude rather than a hidden repair.
No hidden horizon mask; legacy H/M diagnostics retain their existing behavior.

## Evidence and required interpretation

`test_gamma_damping.py`: exact flat spherical connection has C=0 despite
Gamma=-2/r; beta=s*r gives full divbeta=3s; nonconstant manufactured metric and
odd signed C pass on both sides and within abs(r)<0.5; shell measure, signed
health extrema and configuration validation tested. Independent actual RHS
tests additionally check stage freshness, sign, parity, gauge coupling and
zero-control arithmetic. Matched physical evolutions are reported separately.

A lower C alone is not enough: require no degradation of H/M, metric/chi
health, horizon behavior and resolution/CFL sensitivity. Existing puncture
chi negativity under KO6 is not repaired here. A short run is exploratory,
not a claim of stable long-time damping or a publication-ready physics result.

### Local verification on 2026-10-30

- 12 existing BSSN boundary/horizon/RK/runtime tests passed.
- 6 filter-pipeline and 6 KO8 tests passed.
- 3 manufactured helper/diagnostic tests passed.
- Independent unfiltered and filtered actual-RHS checks: 24 manufactured cases,
  including both coefficient signs, positive/negative/zero divergence and two
  stage states. Only Gamma and B change; maximum error about 1.6e-13.
- Independent archived base288e869 versus c_G=0: all nine actual RHS arrays
  exactly equal. Reusable actual-RHS regressions are in test_gamma_rhs.py.

### First coupled puncture screen: HOLD, not a safe production setting

Matched experiments use the frozen source `764adce`, E642, CFL0.1, mass1,
[-20,20], Nx800 (also Nx1600), fixed exterior shell 1<abs(r)<10. Archived
base and c_G=0 legacy curves/CSV are byte-identical through2M.
At2M c_G=1 lowers scalar Gamma L2 by1.74% with KO6 sigma0.5, but KO runs
already have negative chi and damping worsens that minimum. Fine-grid benefit
is only0.416%; no-KO benefit0.372%, so efficacy is resolution/KO sensitive.
In the extended5M no-KO comparison, c_G=1 lowers exterior C by4.4% yet
creates negative chi by saved time4.8M (off remains positive) and increases
momentum norm by22% at5M. These are adverse coupled-physics results, despite
correct unit tests. c_G remains default0; no coefficient is recommended.
Raw divergence changes sign over substantial portions of the grid, so this
term can locally anti-damp. No positivity repair was used. Full raw outputs,
configs, time-series ratios and logs are in the sibling local artifact
`gamma-trials/REPORT.md`; smaller-coefficient controls are analyzed there.

Final bounded no-KO 5M comparison (ratios versus same c_G=0 control):

| c_G | C L2 ratio | H L2 ratio | M L2 ratio | minimum chi through 5M |
|---:|---:|---:|---:|---:|
|0|1|1|1|2.964545e-6|
|0.25|0.988254|0.950466|1.053919|2.239081e-6|
|0.5|0.977013|0.901368|1.108745|7.757416e-7|
|1|0.955992|0.804457|1.220110|-6.033283e-6|

Both smaller coefficients retain positive sampled chi only over this bounded
window, while worsening momentum by5.39%/10.87%. They are not established
safe settings or overall improvements. No further parameter sweep or physics
PR was undertaken. PR #9 compatibility is untested; all numbers refer only
to frozen base288e869 plus implementation764adce, not the gauge-registry API.


## Reference-determinant correction (2026-10-30)

The original trials above used metric-derived divergence. The corrected CCD
uses beta' + 2 beta/r, matching NRPy's fixed conformal determinant. Only the
CCD addition and its rate diagnostics change; native GBSSN metric/Gamma/gauge
terms remain unchanged. Old scientific trial numbers do not qualify the new
operator. The old metric-derived helper remains solely for comparisons.

Reference fork: wkblack/nrpy drpy e8d956896d2ad9db594459640677f50d28f163ba,
BSSN_quantities.py determinant convention and BSSN_RHSs.py CCD_C term. The
c_G coefficient is unchanged. c_H remains phi-normalized (chi factor four).

An off-determinant polynomial test distinguishes the two divergences. Actual
RHS tests check two stage states, both damping/shift signs, zero shift, poisoned
stored diagnostics, Gamma and B-driver increments, and unchanged other fields.
For derivative filter factor .9 and raw beta=s*r, expected divergence is 2.9*s,
not 2.7*s. Diagnostics always use raw-state derivatives as before. The solver's
cell-centered puncture grid excludes r=0; the helper raises ValueError rather
than silently imposing a regular-origin limit on puncture data.

Overlap check at 23:33 UTC: upstream main 0065bfdf includes the independent gauge
API and merged earlier fixes, but no H/C damping. No upstream changes were
partially integrated; this remains a frozen-base local experiment.

Validation: all 38 focused and integration regressions pass in the existing
pynwave test venv, using the unchanged test-only typing.Any import shim. An
archived pre-correction be0ca3a comparison verifies bitwise equality of all nine
RHS fields (108 cells) for c_G=0. git diff --check passes. No evolution result is
claimed by these implementation tests.

## Corrected-operator numerical screen and positivity limit (2026-10-30)

All results below use source 79b1799, with no further production changes.
Independent review passes 24 manufactured actual-RHS cases (maximum error
1.55e-13), including derivative filtering and B-driver coupling, and confirms
full nine-field zero-damping equivalence.

Matched puncture trials use collapsed lapse, eta=2, E642, [-20,20], CFL=.1,
KO=0 and coordinate-dr norms over 1<|r|<10. At H=.016, G=.5 still raises final
M by 8.54% at 5M despite reducing H and C. Correcting the divergence is not by
itself an efficacy solution.

Increasing H to .032 gives a transient joint benefit for G=.5 versus H alone:
at 5M, [H_g,H_B,M,|C|] ratios are [.968261,.874312,.952238,.956801] at N800
and [.941157,.805581,.901750,.931904] at N1600. Integrated squared norms also
improve, but this does not establish sustained stability. First sampled chi
loss is earlier with G=.5: 5.2M versus 5.6M at N800; 5.6M versus 5.9M at N1600.
The .032 coefficient is exploratory in 1D, not a recommended 3D default.

The requested extension target was .7 of full grid width: L=40M, target=28M.
Every one of 16 distinct configurations loses sampled chi positivity between
4.8M and 5.9M, including all-off (5.4M). N800 first-loss times are:

| c_H | c_G=0 | .25 | .5 | 1 |
|---|---:|---:|---:|---:|
| 0 | 5.4 | — | — | — |
| .008 | 5.4 | 5.2 | 5.2 | — |
| .016 | 5.4 | 5.2 | 5.2 | 4.8 |
| .024 | 5.4 | 5.4 | 5.2 | — |
| .032 | 5.6 | 5.4 | 5.2 | — |

N800 preceding positive samples are .2M earlier; fine-grid samples are .1M
apart. These are sampled brackets, not exact crossing times. All fields remain finite, and
both conformal metric components remain positive, at first chi loss. In all ten
newly instrumented N800 cases, the chi minimum first occurs at |r|=.025M.
Locations were not originally logged for the six reused configurations.

The first N1600 extension monitor watched an incorrect output suffix and did
not stop at chi loss; the solver's nonfinite-state guard later stopped both
runs. Corrected monitor path logic passes replay tests, and all ten subsequent
fresh-prefix remaining-case checks stopped at the first observed chi loss.
All 153 previously saved curve hashes per fine case match through 5M after
recomputation. Late invalid samples are retained only as failure evidence.

There were 31 timed launches; none remain active. The bounded search does not
yield a stable production C coefficient. Baseline puncture positivity must be
understood before interpreting short exterior-norm gains as overall success.
No floor, M damper, gauge/BC rewrite, automatic further optimization or
publication follows. Defaults remain off. Full reproducible report and tables
are in the sibling gamma-corrected-trials/REPORT.md and positivity_table.json;
implementation review is in gamma-corrected-review/REVIEW.md.
