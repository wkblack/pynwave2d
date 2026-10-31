# JP6_JT4: a separately selectable hybrid

For the new JP6_JT4 selector, the first two allocated boundary rows use the existing JT4 P **and** Q
boundary equations: the one-sided JT4 row at index zero and its centered
row at index one, reflected at the opposite boundary. Every JP6 interior
coefficient remains unchanged. This is a hybrid with **fourth-order boundary
accuracy and sixth-order interior accuracy**, not canonical published JP6
boundary coefficients. Boundary-dominated/global maximum error may be fourth
order. Existing JP6 coefficients and behavior, including its known defect, are unchanged. No tail, gauge, BSSN equation, filter, or default setting changes here.

The original second-row closure dropped an exterior P beta coefficient while
retaining the interior Q stencil, failing even D1 linear / D2 quadratic
consistency (exact row residuals 1/114 and 1/97). An earlier Q-only sixth-order
moment repair fixed consistency but introduced a Dirichlet-wave instability.
This joint-row repair supersedes that recommendation; the earlier experiment
and its data are retained separately.

## Regression coverage

Run `python -m unittest discover -s tests -p 'test_jp6_jt4*.py' -v`.
The six boundary tests check both boundary rows and the interior defining polynomial
moments, all allocated rows (including evolved Sommerfeld cells), physical
rows, LU/SciPy backends, BH masks, fourth-order boundary and sixth-order
interior smooth refinement, reflection parity, Dirichlet-wave spectra/RK4,
and a radial-wave proxy with three evolved Sommerfeld outer cells both with
and without a single-anchor residual tail. Exact external-cell function
values separate derivative consistency from ghost extrapolation.

In the frozen equivalent replacement-candidate tests, restoring original coefficients makes 33 assertions fail. Substituting the
previous Q-only repair fails the Dirichlet-wave test, despite passing its
polynomial and relevant radial-wave tests. Thus consistency alone is not the
stability gate.

## Stability and validation scope

Matched model tests at N32/64/128 cover strong-inflow advection, Dirichlet
diffusion/waves, and three-cell radiative radial waves with/without tail.
Joint JT4 rows remove the tested O(1/h) growth and Dirichlet-wave instability
of the Q-only repair. This is **not a universal energy-stability proof**:
strong single-endpoint inflow still has JT4-like finite growth (maximum real
eigenvalue about 0.297, 0.298, 0.306 on the unit domain as N doubles), plus
non-normal transient amplification. Comparisons use matched boundary
formulations and norms; do not read them as unconditional stability for
arbitrary PDEs, boundaries or timesteps.

Separate four-case puncture BSSN tests at N800/1600 through 2M, with and
without the single-anchor tail, remain healthy and show all sampled outer
H/M/C norm gains with bitwise near-zone equality. These short tests do not
certify long-time puncture evolution. Production-to-research snapshot
comparison and full-suite receipts are maintained outside the source patch.

P boundary rows and conditioning change; Q support is no wider than the
original first-row support and the JP6 interior bands are retained. No runtime
speed claim is made. The independent ME right-boundary parity defect is not
repaired by this change.

## Selector non-regression

Three additional tests check unchanged enum IDs/old selectors and exact legacy coefficient arrays, explicit new-name parsing, and parameter-factory dispatch with/without BH masks under LU/SciPy. The existing banded-versus-dense construction test now includes both new derivative types and masked pairs. Independent source/array/gradient/snapshot checks compare the new name against the frozen replacement candidate6d353944; numerical equivalence does not imply the old JP6 option was repaired.
