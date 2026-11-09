# JP6 boundary-adjacent closure candidate

This is a **local consistency-repair candidate**, not a claim of a globally
nondegrading BSSN discretization or a published upstream fix.

The existing JP6 row 1 truncates the implicit pentadiagonal left-hand side to
four derivative samples, while retaining the centered three-point right-hand
side. Even D1(x) and D2(x²) therefore fail consistency by O(1). The mirrored
row has the same problem. Row 0 is already sixth-order consistent.

The candidate preserves the implicit matrix and the complete interior stencil.
For derivative order d, choose RHS coefficients q_j on nodes j=0..(d+5), solving
exact rational moment equations

    sum_j q_j j^k = sum_i p_i d^d(x^k)/dx^d evaluated at x=i

for k=0..d+5 and the existing row-1 implicit coefficients p_i. Thus D1 matches
monomials through degree 6, D2 through degree 7. Right boundaries are reflected
with the existing parity. This uniquely determines the new RHS coefficients;
it is not recovery of a verified original-author intended closure.

Tests assert polynomial consistency and refinement of exp(x) on 17/33/65 points.
A short 800-point, 2M spherical BSSN trial with both JP6 derivatives preserves
saved inner-shell norms and masked diagnostics while changing boundary data.
Full H L2 falls .52970→.51763, but M and Gamma L2 increase slightly
(.11459→.11529 and .08096→.08153). This does **not** satisfy a blanket
no-degradation criterion. The remaining spherical-background boundary defect
and negative inner chi also limit interpretation.

Before promotion: independent closure review; coupled boundary/energy or spectral
stability assessment; smooth evolution convergence; longer/multi-resolution
BSSN tests inside justified trust windows. Keep distinct from the clear filter
implementation fixes.
