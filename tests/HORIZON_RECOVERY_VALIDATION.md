# Horizon root and recovery validation

Changes: accept exact grid-point roots while searching outermost first; retain a separate last-valid positive search guess across a failed detection; guard explicitly invalid input guesses. Both radial signs and off-grid roots are tested.

`python tests/test_bssn_horizon.py`: 3 methods pass. The original exact-grid-root test fails. An end-to-end solver test injects detections .5, failure, then a recoverable .51: original search guesses [.7,.5,0] cannot recover; fixed guesses [.7,.5,.5] do recover. The helper-only test is additional coverage, not independent original-bug evidence.

## Explicit diagnostic policy and schema

On a failed current detection, legacy radius/mass values remain zero, but the constraint mask retains its last valid inner radius instead of silently expanding to the origin. A warning is printed. Two appended CSV columns, `horizon_found` and `mask_inner_radius`, explicitly expose detection validity and the actual integration mask. Existing first five columns are retained in their original order. Before any successful detection the mask is zero and initial search guess remains .7. The injected sequence checks validity True/False/True, masks .5/.5/.51, and reported radii .5/0/.51.

Retaining a stale mask is a deliberate diagnostic continuity choice, not evidence of a horizon at that location. CSV readers insisting on exactly five columns will need updating. Original five columns in the unaffected-path evolution are identical; the entire CSV is necessarily different because the two metadata columns were added.

## Scope and dependencies

Independent branch from upstream `1063d45ff3338827cd6efea0185b965b55ba2eb4`, verified still main on 2026-10-30. No changes from PR #1 are duplicated. Upstream's missing `typing.Any` import is already covered by PR #2; the tests use an explicit test-only import shim on both original and fixed trees until that lands. No other PR is required for these source changes. No Gamma damping, compact-stencil changes, or positivity clipping.

## Unaffected-path non-regression

A matched 800-point extended-domain puncture evolution, E642, KO6 sigma .5, CFL .1, ±20M, 400 steps to 2M completed on original and this branch. All six curve snapshots are byte-identical. The original five columns of the full constraint/horizon CSV are identical. This proves equivalence on that exercised path, not universal or long-time accuracy. The pre-existing negative-chi/sqrt warning remains.

## Evidence

Local full artifacts: `/workspace/scratch/e1a39ef4befa/runtime-evidence/independent/`, containing per-branch TOMLs, stdout/stderr, original/fixed test logs, curve/CSV outputs and `comparison.json`. Tests can select the original checkout with `BSSN_TEST_ROOT=/path/to/original`. Use the normal project dependency environment and explicit timeouts.
