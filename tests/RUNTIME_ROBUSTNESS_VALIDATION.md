# Runtime robustness validation

Fixes two defects: the masked curve writer references undefined `rr`, and the solver detects only NaNs in grr, then exits successfully. The writer now uses its supplied x coordinates. The solver checks all evolved arrays for NaN/Inf at initialization and after each completed step, before diagnostics/output, and raises an exception with field and step.

`python tests/test_bssn_runtime_robustness.py`: 3 methods pass. Two tests exercise pre-existing public behavior and fail on original: masked writer, and Inf injected into B_r through the actual solver/RK interface at step 1. Original returns process status 0; fixed returns nonzero with the specific field/step message. The additional helper test covers 27 nonfinite combinations (nine fields, NaN/+Inf/−Inf); helper absence on the original is not counted as independent bug evidence. The complete unchanged-path CSV is also byte-identical.

This improves failure reporting, not the underlying evolution or chi positivity. A NaN arising within an intermediate RK stage may trigger another exception first; only initialization and completed-step states are explicitly inspected.

## Scope and dependencies

Independent branch from upstream `1063d45ff3338827cd6efea0185b965b55ba2eb4`, verified still main on 2026-10-30. No changes from PR #1 are duplicated. Upstream's missing `typing.Any` import is already covered by PR #2; the tests use an explicit test-only import shim on both original and fixed trees until that lands. No other PR is required for these source changes. No Gamma damping, compact-stencil changes, or positivity clipping.

## Unaffected-path non-regression

A matched 800-point extended-domain puncture evolution, E642, KO6 sigma .5, CFL .1, ±20M, 400 steps to 2M completed on original and this branch. All six curve snapshots are byte-identical. The original five columns of the full constraint/horizon CSV are identical. This proves equivalence on that exercised path, not universal or long-time accuracy. The pre-existing negative-chi/sqrt warning remains.

## Evidence

Local full artifacts: `/workspace/scratch/e1a39ef4befa/runtime-evidence/independent/`, containing per-branch TOMLs, stdout/stderr, original/fixed test logs, curve/CSV outputs and `comparison.json`. Tests can select the original checkout with `BSSN_TEST_ROOT=/path/to/original`. Use the normal project dependency environment and explicit timeouts.
