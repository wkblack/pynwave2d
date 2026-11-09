# KO8 endpoint typo validation

Base: upstream main1063d45 (verified2026-10-30), independent of PR #1.
Only the left endpoint KO8 boundary stencil changes. The previous expression
repeated u[5] and omitted u[0]/u[2]. The new endpoint is the exact reflection
of the existing right endpoint: coefficients [-1,4,-6,4,-1] on u[0] through u[4].
No other boundary row or interior coefficient changes. No compact derivatives.

Run `python tests/test_ko8_boundary.py`.
Six regression methods pass; original source fails five assertions/subtests.
Tests cover low-polynomial annihilation (degrees0..2), arbitrary-data reflection,
linearity, exact reflected endpoint formula, unchanged boundary-off/interior
paths, and100RK4 steps of active KO8 dissipation: finite values, decaying L2,
and even parity. This is a dissipation experiment, not a full BSSN stability proof.

Matched BSSN unaffected control:800points,±20,E642,CFL.1,KO6.5,400steps to2M.
All six saved curves and constraint CSV are byte-identical to upstream. Both
retain unrelated negative-inner-chi and horizon-warning limitations. No claim
of eighth-order boundary accuracy or universally nondegrading simulations.
