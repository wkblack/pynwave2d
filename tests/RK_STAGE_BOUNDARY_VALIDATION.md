# RK stage boundary validation

`BSSN.apply_bcs(u,g)` formerly mutated `self.u`, ignoring the RK intermediate state supplied as u. This changes only that array reference for all nine regularity operations.

`python tests/test_bssn_rk_boundary.py`: 2 methods pass. The half-domain test fails on original, then checks fixed even/odd ghost parity, unchanged main state, and unchanged physical cells of the temporary stage. The extended-domain no-op test passes on both versions. The complete unchanged-path CSV is byte-identical as well.

The positive-half-domain stage semantics are now correct; a full half-domain convergence study is still not claimed. This defect does not affect the default extended-domain branch.

## Scope and dependencies

Independent branch from upstream `1063d45ff3338827cd6efea0185b965b55ba2eb4`, verified still main on 2026-10-30. No changes from PR #1 are duplicated. Upstream's missing `typing.Any` import is already covered by PR #2; the tests use an explicit test-only import shim on both original and fixed trees until that lands. No other PR is required for these source changes. No Gamma damping, compact-stencil changes, or positivity clipping.

## Unaffected-path non-regression

A matched 800-point extended-domain puncture evolution, E642, KO6 sigma .5, CFL .1, ±20M, 400 steps to 2M completed on original and this branch. All six curve snapshots are byte-identical. The original five columns of the full constraint/horizon CSV are identical. This proves equivalence on that exercised path, not universal or long-time accuracy. The pre-existing negative-chi/sqrt warning remains.

## Evidence

Local full artifacts: `/workspace/scratch/e1a39ef4befa/runtime-evidence/independent/`, containing per-branch TOMLs, stdout/stderr, original/fixed test logs, curve/CSV outputs and `comparison.json`. Tests can select the original checkout with `BSSN_TEST_ROOT=/path/to/original`. Use the normal project dependency environment and explicit timeouts.
