"""
Shift damping: the eta in the Gamma-driver  d_t B = d_t Gt + ... - eta B.

A damping class is called as  eta(r, chi, g_rr, d_chi, M)  and returns a
scalar or an array over the grid; rhs_gauge accepts either.
"""

import numpy as np

from .registry import register_shift


class ShiftDamping:
    def __call__(self, r, chi, g_rr, d_chi, M):
        raise NotImplementedError

    def __repr__(self):
        args = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{self.name}({args})"


@register_shift("constant")
class ConstantEta(ShiftDamping):
    """eta everywhere."""

    def __init__(self, eta=2.0):
        if eta < 0.0:
            raise ValueError("eta must be non-negative")
        self.eta = eta

    def __call__(self, r, chi, g_rr, d_chi, M):
        return self.eta


@register_shift("Schnetter")
class Schnetter(ShiftDamping):
    """eta = eta / (1 + (r/R)^2): radial falloff over the scale R."""

    def __init__(self, eta=2.0, R=50.0):
        if eta < 0.0:
            raise ValueError("eta must be non-negative")
        self.eta = eta
        self.R = R

    def __call__(self, r, chi, g_rr, d_chi, M):
        return self.eta / (1.0 + (r / self.R) ** 2)


@register_shift("eta_MB")
class EtaMB(ShiftDamping):
    """
    Mueller & Bruegmann (2010):  eta = R0 |d_r W| / (sqrt(g_rr) (1 - W)^2),
    with W = sqrt(chi); the conformal metric has g^rr = 1/g_rr in 1D.

    Stiff at an under-resolved puncture: d_chi / (2 sqrt(chi)) is large where
    chi -> 0, and the damping term -eta B is stable under RK4 only for
    eta dt <~ 2.8.  With default params, Nx = 2001 (dt = 0.01) blows up within
    20 steps (eta ~ 700 at r = 0.05); Nx = 10001 (dt = 0.002) runs, eta ~ 14.
    """

    def __init__(self, R0=1.31241):
        self.R0 = R0

    def __call__(self, r, chi, g_rr, d_chi, M):
        eps = 1.0e-10
        W = np.sqrt(np.maximum(chi, eps))
        dW = d_chi / (2.0 * W)
        denom = np.sqrt(np.maximum(g_rr, eps)) * (1.0 - W + eps) ** 2
        return self.R0 * np.abs(dW) / denom


@register_shift("peaked_lorentz")
class PeakedLorentz(ShiftDamping):
    """eta = bg + peak / (1 + (r/R)^2): a Lorentzian bump on the puncture."""

    def __init__(self, bg=2.0, peak=2.0, R=1.31241):
        self.bg = bg
        self.peak = peak
        self.R = R

    def __call__(self, r, chi, g_rr, d_chi, M):
        return self.bg + self.peak / (1.0 + (r / self.R) ** 2)
