"""
Telegrapher lapse: a second-order-in-time gauge that evolves the auxiliary
field A = d_t alpha,

    d_t alpha = A + beta d_r alpha
    d_t A     = c^2 D^2 alpha - 2 mu (A + 2 alpha K)
                + s mu^2 g (alpha - alpha_ref) + beta d_r A.

The lapse propagates at speed c and is damped at rate mu.  The -2 alpha K
term makes A = -2 alpha K (1+log with k = 2) the equilibrium, which is also
the initial value of A.  This gauge is experimental.

mu2_sign (s): with s = -1 the K-decoupled roots are lambda = -mu +- i c k, so
every wavenumber travels at exactly c (the distortionless, Heaviside case);
s = +1 is dispersive and s = 0 drops the mass term.  Values in between scale
the term continuously.

mu2_ref (alpha_ref): the mass term must act on a perturbation, not the bare
lapse: on the bare lapse the k = 0 mode collapses alpha to zero on a time
1/mu (s = -1) or grows without bound (s = +1).
    "none"         0 (the bare lapse; for comparison only)
    "unity"        1
    "chi"          sqrt(chi), the puncture's trumpet lapse profile
    "chi_clamped"  sqrt(min(chi, 1)); chi overshoots 1 near the puncture
                   during evolution, after which sqrt(chi) would drive alpha up
    "initial"      alpha at t = 0 (equal to sqrt(chi) for collapsed puncture data;
                   default)

mu2_gate (g): a spatial weight on the mass term, 1 in the wave zone and small
near the puncture, where the chi > 1 runaway lives.
    "none"    1
    "alpha"   alpha, clipped to [0, 1]
    "alpha2"  alpha^2, clipped (default)
    "chi1"    a tanh step in chi, off above chi = 1 + 8 gate_width; centred at
              1 + 4 gate_width so the wave zone (chi -> 1) sees ~1, not 0.5

The defaults are the only combination that stays healthy for a puncture.
With s = -1 the mass term pushes A up wherever alpha < alpha_ref; at the
puncture alpha -> 0, so it fights the lapse collapse, and with a co-evolving
reference sqrt(chi) the growth of chi feeds the drive back (chi runs away and
the run fails near 20-30 M).  The alpha^2 gate removes the term where alpha
collapses, and the frozen reference cuts the feedback: either alone still
runs away (the frozen reference fails near 50 M; the gate alone completes
100 M with chi growth still accelerating).
"""

import numpy as np

from .lapse import LapseGauge
from .registry import register_lapse

MU2_REFS = ("none", "unity", "chi", "chi_clamped", "initial")
MU2_GATES = ("none", "alpha", "alpha2", "chi1")


@register_lapse("telegrapher")
class Telegrapher(LapseGauge):
    n_aux = 1
    aux_names = ["lapseA"]

    def __init__(self, c=1.5, mu=1.0, mu2_sign=-1.0, mu2_ref="initial",
                 mu2_gate="alpha2", gate_width=0.05):
        if c <= 0.0:
            raise ValueError("telegrapher: c must be positive")
        if mu <= 0.0:
            raise ValueError("telegrapher: mu must be positive")
        if not -1.0 <= mu2_sign <= 1.0:
            raise ValueError("telegrapher: mu2_sign must lie in [-1, 1]")
        if mu2_ref not in MU2_REFS:
            raise ValueError(f"telegrapher: mu2_ref must be one of {MU2_REFS}")
        if mu2_gate not in MU2_GATES:
            raise ValueError(f"telegrapher: mu2_gate must be one of {MU2_GATES}")
        if gate_width <= 0.0:
            raise ValueError("telegrapher: gate_width must be positive")
        self.c = c
        self.mu = mu
        self.mu2_sign = mu2_sign
        self.mu2_ref = mu2_ref
        self.mu2_gate = mu2_gate
        self.gate_width = gate_width
        self._alpha_initial = None

    def __repr__(self):
        args = ", ".join(f"{k}={v}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{self.name}({args})"

    def speed_inf(self):
        return self.c

    def initial_aux(self, alpha, K):
        """Initial A: the 1+log equilibrium -2 alpha K (zero for K = 0 data)."""
        self._alpha_initial = alpha.copy()
        return -2.0 * alpha * K

    def reference(self, alpha, chi):
        if self.mu2_ref == "none":
            return 0.0
        if self.mu2_ref == "unity":
            return 1.0
        if self.mu2_ref == "chi":
            return np.sqrt(np.maximum(chi, 1.0e-14))
        if self.mu2_ref == "chi_clamped":
            return np.sqrt(np.clip(chi, 1.0e-14, 1.0))
        if self._alpha_initial is None:
            raise RuntimeError("telegrapher: mu2_ref = 'initial' needs initial_aux() first")
        return self._alpha_initial

    def gate(self, alpha, chi):
        if self.mu2_gate == "none":
            return 1.0
        if self.mu2_gate == "alpha":
            return np.clip(alpha, 0.0, 1.0)
        if self.mu2_gate == "alpha2":
            return np.clip(alpha, 0.0, 1.0) ** 2
        w = self.gate_width
        return 0.5 * (1.0 - np.tanh((chi - (1.0 + 4.0 * w)) / w))

    def rhs_aux(self, alpha, A, K, lap_alpha, chi):
        """d_t A without the advection term, which the solver adds."""
        return (self.c**2 * lap_alpha
                - 2.0 * self.mu * (A + 2.0 * alpha * K)
                + self.mu2_sign * self.mu**2 * self.gate(alpha, chi)
                * (alpha - self.reference(alpha, chi)))
