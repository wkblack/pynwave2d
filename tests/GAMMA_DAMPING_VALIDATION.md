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
divbeta = beta' + beta (a'/(2 a) + b'/b)
delta(dtGamma) = -c_G C divbeta
```

This is a constraint adjustment, not universally positive damping: negative
`c_G*divbeta` is anti-damping. No added 2/3 correction: the base RHS already
uses the metric-computed Gamma in its undifferentiated shift couplings.
The adjustment is applied before the B-driver reads `dtGamma`. Boundary
conditions and RHS filters then act in their original order.

Current-stage data and existing RHS derivative policy are used: if derivative
filtering is enabled, derivatives are of filtered stage fields, with raw stage
metric factors, exactly as the other RHS terms. No `self.C` or `self.u` reuse,
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
