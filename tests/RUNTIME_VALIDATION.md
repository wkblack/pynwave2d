# Local runtime fixes: evidence and limits

Base: `918dcfb` (the boundary-fix PR tree). This isolated worktree is not published and adds no Gamma damping, positivity clipping, or stencil changes.

## Priorities and proposed changes

1. **High: RK boundary state on half domains.** `BSSN.apply_bcs(u,g)` now updates its argument, not the main stored state. A stage-array regression checks correct even/odd ghost parity, unchanged main state and unchanged physical stage cells. The extended-domain path remains a no-op. This repairs RK semantics; it is not yet a full half-domain convergence qualification.
2. **High: non-finite failure detection.** Inspect all nine evolved fields for NaN/Inf before initial output and after every complete step. Raise `FloatingPointError`, which produces nonzero CLI exit. This deliberately does not treat negative chi as a repaired or acceptable physical state; positivity handling is a separate unresolved problem.
3. **Medium-high: horizon root and recovery.** Accept exact grid-point roots, selecting the outermost found root as before. Preserve a separate last-valid positive search guess across detection failures. An invalid explicitly supplied initial guess falls back to 1.0 in `find_horizon`.
4. **Blocking usability: missing Any import.** Import `typing.Any`; direct Python 3.12 startup now works without the test shim.
5. **Lower-use crashes:** masked writer uses the supplied x array instead of undefined rr; explicit None filtering returns the same no-filter sentinel used when filters are omitted.

## Explicit horizon-mask design choice

A failed current detection is still recorded as radius/mass zero in the existing CSV schema, with a warning. The search guess and diagnostic inner mask retain their last valid values; before any valid detection the mask is zero and the solver's initial search guess remains 0.7. The mask is **not** evidence that the horizon remains at its former location. This prevents a finder failure from silently changing the norm's integration region, but is a diagnostic-policy choice deserving user review. A future schema should include finder validity and mask radius separately rather than overloading radius zero.

## Regression tests that bite

Run `python tests/test_bssn_runtime.py` using the existing dependency environment. Eight test methods pass on fixed code. Against the original tree, with an import shim only to enable the other tests:

- Five tests of pre-existing behavior fail: direct clean import, masked writer, explicit None filter, exact-grid horizon roots, and stage-array regularization.
- The unaffected extended-domain boundary test passes.
- Two tests exercise newly factored helpers; their absence on the old tree is **not counted as evidence of a bug**.

The original four boundary/ghost regression methods also pass after these changes.

Additional end-to-end fault injection through actual `solver.main`:

- Inject an Inf in B_r at RK step 1, before any diagnostic interval: original exits 0; fixed exits 1 before diagnostic/output handling. The fixed helper separately rejects NaN, +Inf and -Inf in each of all nine fields (27 cases).
- Supply horizon detections .5, failure, then a recoverable .51. Original guesses [.7,.5,0] and mask radii [.5,0,0]; fixed guesses [.7,.5,.5] and masks [.5,.5,.51]. The final recovery is denied by the mock finder only when the guess has collapsed to zero.

## Matched evolution non-regression

An 800-point extended-domain puncture run, ±20M, E642, CFL .1, KO6 sigma .5, runs 400 steps to 2M. Original uses the previously disclosed `Any` import shim; fixed launches directly. All six saved field/constraint curve files and the full constraint/horizon CSV are byte-identical (7/7 files). This is strong evidence of no changes on this exercised path, not proof of no degradation for every configuration or long-time evolution. Both retain the known near-puncture negative-chi/horizon square-root warning.

## Reproduction artifacts

Sibling `runtime-evidence/` contains baseline/fixed TOMLs and logs, complete outputs, `comparison.json` SHA256 receipts, `inject_nonfinite.py`, `inject_horizon_failure.py`, injected-case TOMLs/logs, and regression logs. Baseline regression invocation sets `BSSN_TEST_ROOT` to the original checkout and `BSSN_BASELINE_SHIM=1`; the direct-import test runs a clean subprocess without the shim. All executions used explicit 60–180 second timeouts and single numerical threads.
