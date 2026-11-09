# Filter pipeline/config validation

Base: upstream main1063d45 (verified2026-10-30). This branch is independent of
PR #1 and contains no compact-derivative or Gamma changes.

Changes:
- Apply every RHS-filter contribution at every RK4 stage, instead of only the
  last configured filter.
- Return the explicit no-filter sentinel for `type=None`, rather than a Python
  None that later crashes the filter pipeline.
- Reject KO apply modes other than Rhs and frequencies other than1; the KO
  constructors previously silently overrode these requests.
- Reject multiple Vars filters until composition semantics are defined, rather
  than silently ignoring all but the first.

Not included: the single-Vars dispatch method-call fix. Activating the tested
JTT8 setup produced nonfinite data; it needs separate numerical qualification.
No compact-filter development is attempted here.

Run `python tests/test_filter_pipeline.py` with NumPy/Numba/SciPy/pentapy.
The test harness supplies the same test-only typing.Any import compatibility
shim for both trees because upstream has an unrelated missing import.

Six regression methods pass here. Original source fails nine assertions/subtests
plus one error. Tests include a nonzero linear ODE with analytic RK4 amplification,
call counters, two real KO6 filters versus one with summed strength over100
steps, decay/finiteness, explicit None through RK4, invalid KO settings and
multiple-Vars rejection. Single-RHS behavior is retained.

Matched BSSN control:800points,±20,E642,CFL.1,KO6.5,400steps to2M.
All six saved curves and constraint CSV are byte-identical to upstream.
This supports nonregression of this measured unaffected path, not universal
physical-health/stability: both still have the known negative inner chi and
horizon warning. Production parser intentionally now rejects invalid settings.
