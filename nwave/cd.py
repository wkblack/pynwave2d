import numpy as np
from scipy.linalg import solve_banded, inv
from enum import Enum
from .bandedLUSolve import *
from .utils import *
from .types import *
try:
    import pentapy as pp
except ImportError:
    pp = None
from numba import njit

CompactFirstDerivatives = [
    DerivType.D1_KP4,
    DerivType.D1_ME44,
    DerivType.D1_ME642,
    DerivType.D1_JT4,
    DerivType.D1_JT6,
    DerivType.D1_JP6,
    DerivType.D1_JP6_JT4,
    DerivType.D1_DSQ6A,
    DerivType.D1_DSQ6B,
    DerivType.D1_BYUP6,
    DerivType.D1_Wm6,
    DerivType.D1_DSQ6B_LEFT,
    DerivType.D1_DSQ6B_RIGHT,
    DerivType.D1_BYU_C6_Op10,
    DerivType.D1_BYU_P4_Op18,
    DerivType.D1_C6_4,
    DerivType.D1_C4_1,
    DerivType.D1_A4_3
]
CompactSecondDerivatives = [
    DerivType.D2_ME44,
    DerivType.D2_ME642,
    DerivType.D2_JT4,
    DerivType.D2_JT6,
    DerivType.D2_JP6,
    DerivType.D2_JP6_JT4,
    DerivType.D2_DSQ6A,
    DerivType.D2_DSQ6B,
    DerivType.D2_BYUP6,
    DerivType.D2_2C4_1,
    DerivType.D2_2C6_1,
    DerivType.D2_2A_1
]
CompactExplicitDerivatives = [
    DerivType.D1_ME44,
    DerivType.D2_ME44,
    DerivType.D1_ME642,
    DerivType.D2_ME642,
]


class NCompactDerivative:
    def __init__(
        self, N: int, dtype: DerivType, method: CFDSolve, Pmat, Qmat, denom, mask=None
    ):
        """
        Initialize the NCompactDerivative class.
        """
        if dtype in CompactExplicitDerivatives:
            if method != CFDSolve.D_LU and method != CFDSolve.D_INV:
                print(
                    f"Method {method} not supported for explicit derivatives of type {dtype}.  Using D_LU instead."
                )
            self.method = CFDSolve.D_LU
        else:
            self.method = method

        self.N = N
        self.dtype = dtype
        self.Pbands = Pmat[1]
        self.Qbands = Qmat[1]
        self.P = Pmat[0]
        self.Qb = Qmat[0]  # banded storage; the full Q is built only on demand
        self._Q = None
        self.denom = denom
        self.lu_factorization = None
        self.D = None
        self.overwrite = True
        self.checkf = True
        self.mask = mask
        self.HAVE_ADVECTIVE_DERIV = False

        if self.method == CFDSolve.LUSOLVE:
            # factor the P matrix for LU solve
            self.lu_factorization = lu_banded(
                self.Pbands,
                self.P,
                overwrite_ab=self.overwrite,
                check_finite=self.checkf,
            )
        elif self.method == CFDSolve.D_INV:
            if dtype in CompactExplicitDerivatives:
                self.D = self.Q
            else:
                nb = self.Pbands[0]
                A = banded_to_full(self.P, nb, nb, self.N, self.N)
                Ainv = inv(A)
                self.D = np.matmul(Ainv, self.Q)
        elif self.method == CFDSolve.D_LU:
            if dtype in CompactExplicitDerivatives:
                self.D = self.Q
            else:
                self.D = solve_banded(self.Pbands, self.P, self.Q)
        elif self.method == CFDSolve.SCIPY or method == CFDSolve.PENTAPY:
            # no additional setup required for these methods
            pass
        else:
            raise ValueError(f"Compact derivative method = {method} not implemented")

    @classmethod
    def bh_deriv(
        cls,
        r,
        dtype0: DerivType,
        dtypeBH: DerivType,
        method: CFDSolve,
        rbh,
        delta_r,
    ):
        """
        Factory method to create combined derivative for BH
        """
        N = len(r)
        dr = r[1] - r[0]
        if dtype0 in CompactFirstDerivatives:
            denom = 1 / dr
        elif dtype0 in CompactSecondDerivatives:
            denom = 1 / (dr * dr)
        else:
            raise ValueError(f"Unsupported first derivative type: {dtype0}")

        mask0 = 1.0
        maskBH = 0.0
        r0 = rbh - 2 * delta_r
        r1 = rbh - delta_r
        r2 = rbh + delta_r
        r3 = rbh + 2 * delta_r

        bg_mask = generalized_transition_profile(
            r, mask0, maskBH, r0, r1, r2, r3, method="tanh"
        )

        Pmat, Qmat = build_bh_derivative(N, dtype0, dtypeBH, bg_mask)
        return cls(N, dtype0, method, Pmat, Qmat, denom, mask=bg_mask)

    @classmethod
    def deriv(cls, r, dtype: DerivType, method: CFDSolve):
        """
        Factory method to create an instance based on the derivative type.
        """
        N = len(r)
        dr = r[1] - r[0]
        if dtype in CompactFirstDerivatives:
            denom = 1 / dr
        elif dtype in CompactSecondDerivatives:
            denom = 1 / (dr * dr)
        else:
            raise ValueError(f"Unsupported first derivative type: {dtype}")

        Pmat, Qmat = build_derivative(N, dtype)
        return cls(N, dtype, method, Pmat, Qmat, denom)

    def get_type(self):
        return self.dtype

    def get_method(self):
        return self.method

    def get_P(self) -> np.ndarray:
        """
        Get the P matrix.
        """
        return self.P

    @property
    def Q(self) -> np.ndarray:
        """
        Full N x N Q matrix, built from the banded storage on first use and
        cached.  Only the D_INV, D_LU and PENTAPY methods need it.
        """
        if self._Q is None:
            self._Q = banded_to_full(
                self.Qb, self.Qbands[0], self.Qbands[1], self.N, self.N
            )
        return self._Q

    def get_Q(self) -> np.ndarray:
        """
        Get the Q matrix.
        """
        return self.Q

    def get_Qb(self) -> np.ndarray:
        """
        Get the Q matrix.
        """
        return self.Qb

    def get_Pbands(self) -> tuple:
        """
        Get the P bands.
        """
        return self.Pbands

    def get_Qbands(self) -> tuple:
        """
        Get the Q bands.
        """
        return self.Qbands

    def get_D(self) -> np.ndarray:
        """
        Get the D matrix.
        """
        if self.D is None:
            raise ValueError("D matrix is not initialized")
        return self.D

    def get_mask(self) -> np.ndarray:
        """
        Get the mask array.
        """
        if self.mask is None:
            raise ValueError("Mask is not initialized")
        return self.mask

    def grad(self, f: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the function f using the derivative matrices.
        """
        if len(f) != self.N:
            raise ValueError(f"Input array length {len(f)} does not match N {self.N}")

        if self.method == CFDSolve.LUSOLVE:
            # DGBMV is about 4 times slower than numpy matmul
            # rhs = dgbmv(self.N, self.N, self.Qbands[0], self.Qbands[1], self.denom, self.Q, f)
            # rhs = np.matmul(self.Q, f) * self.denom
            rhs = banded_matvec(
                self.N, self.Qbands[0], self.Qbands[1], self.Qb, f, self.denom
            )
            dxf = lu_solve_banded(
                self.lu_factorization, rhs, overwrite_b=True, check_finite=self.checkf
            )
        elif self.method == CFDSolve.D_INV:
            dxf = _matvec_numba(self.D, f, self.denom)
        elif self.method == CFDSolve.D_LU:
            dxf = _matvec_numba(self.D, f, self.denom)
        elif self.method == CFDSolve.SCIPY:
            # Use Numba-accelerated banded matrix-vector multiply for rhs
            # rhs0 = np.matmul(self.Q, f) * self.denom
            rhs = banded_matvec(
                self.N, self.Qbands[0], self.Qbands[1], self.Qb, f, self.denom
            )
            dxf = solve_banded(self.Pbands, self.P, rhs)
        elif self.method == CFDSolve.PENTAPY:
            if pp is None:
                raise ImportError("pentapy is required for CFDSolve.PENTAPY but is not installed.")
            rhs = np.matmul(self.Q, f) * self.denom
            return np.asarray(
                pp.solve(self.P, rhs, is_flat=True, index_row_wise=False, solver=1)
            )
        else:
            raise ValueError(
                f"Compact derivative method = {self.method} not implemented"
            )

        return dxf


@njit(fastmath=True, cache=True)
def _matvec_numba(D, f, alpha):
    return np.dot(D, f) * alpha


def _coeffs_D1_JTT4():
    Lcoeff = np.array([0.25, 1.0, 0.25])
    Rcoeff = np.array([-3.0 / 4.0, 0, 3.0 / 4.0])
    Lbcoeff0 = np.array([1.0, 3.0])
    Rbcoeff0 = np.array([-17.0, 9.0, 9.0, -1.0]) / 6.0
    Lbands = (1, 1)
    Rbands = (1, 1)
    Lmat = [Lcoeff, [Lbcoeff0], Lbands, 1]
    Rmat = [Rcoeff, [Rbcoeff0], Rbands, -1]
    Rbands = (1, 1)
    return Lmat, Rmat


def _coeffs_D2_JTT4():
    alpha = 1.0 / 10
    a = 6.0 / 5
    Lcoeff = np.array([alpha, 1.0, alpha])
    Rcoeff = np.array([a, -2 * a, a])

    Lbcoeff0 = np.array([1.0, 10.0])
    Rbcoeff0 = np.array([145 / 12, -76 / 3, 29 / 2, -4 / 3, 1 / 12])
    Lbands = (1, 1)
    Rbands = (1, 1)
    Lmat = [Lcoeff, [Lbcoeff0], Lbands, 1]
    Rmat = [Rcoeff, [Rbcoeff0], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_JTT6():
    alpha = 1 / 3
    Lcoeff = np.array([alpha, 1.0, alpha])
    Rcoeff = np.array([-1 / 36, -14 / 18, 0, 14 / 18, 1 / 36])

    Lbcoeff0 = np.array([1.0, 5.0])
    Lbcoeff1 = np.array([1 / 8, 1.0, 3 / 4])

    Rcoeffb0 = np.array([-197 / 60, -5 / 12, 5, -5 / 3, 5 / 12, -1 / 20])
    Rcoeffb1 = np.array([-43 / 96, -5 / 6, 9 / 8, 1 / 6, -1 / 96])
    Lbands = (1, 1)
    Rbands = (2, 2)
    Lmat = [Lcoeff, [Lbcoeff0, Lbcoeff1], Lbands, 1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1], Rbands, -1]
    return Lmat, Rmat


def _coeffs_D2_JTT6():
    """
    Tridiagonal sixth-order second derivative (Tyler 2007, BYU MS thesis,
    "Analysis and Implementation of High-Order Compact Finite Difference
    Schemes": interior Table 2, boundary rows Tables 25 and 27).
    """
    alpha = 2 / 11
    a = 12 / 11
    b = 3 / 11
    Lcoeff = np.array([alpha, 1.0, alpha])
    Rcoeff = np.array([b / 4, a, -2 * a - b / 2, a, b / 4])

    Lcoeffb0 = np.array([1.0, 126 / 11])
    Lcoeffb1 = np.array([2 / 11, 1.0, -131 / 22])
    Rcoeffb0 = np.array(
        [13097 / 990, -2943 / 110, 573 / 44, 167 / 99, -18 / 11, 57 / 110, -131 / 1980]
    )
    Rcoeffb1 = np.array([177 / 88, -507 / 44, 783 / 44, -201 / 22, 81 / 88, -3 / 44])
    Lbands = (1, 1)
    Rbands = (2, 2)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1], Lbands, 1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_JTP6():
    alpha = 17.0 / 57.0
    beta = -1.0 / 114.0
    Lcoeff = np.array([beta, alpha, 1.0, alpha, beta])
    a = 30.0 / 19
    Rcoeffs = np.array([-a / 2, 0.0, a / 2])
    Lcoeffb0 = np.array([1.0, 8.0, 6.0])
    Lcoeffb1 = np.array([alpha, 1.0, alpha, beta])
    Rcoeffb0 = np.array([-43 / 12, -20 / 3, 9, 4 / 3, -1 / 12])
    Rcoeffb1 = np.array([-a / 2, 0.0, a / 2])
    Lbands = (2, 2)
    Rbands = (1, 1)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1], Lbands, 1]
    Rmat = [Rcoeffs, [Rcoeffb0, Rcoeffb1], Rbands, -1]
    return Lmat, Rmat


def _coeffs_D2_JTP6():
    alpha = 12 / 97
    beta = -1 / 194
    a = 120 / 97
    Lcoeff = np.array([beta, alpha, 1.0, alpha, beta])
    Rcoeff = np.array([a, -2 * a, a])

    Lcoeffb0 = np.array([1.0, 11 / 2, -131 / 4])
    Lcoeffb1 = np.array([alpha, 1.0, alpha, beta])
    Rcoeffb0 = np.array([177 / 16, -507 / 8, 783 / 8, -201 / 4, 81 / 16, -3 / 8])
    Rcoeffb1 = np.array([a, -2 * a, a])
    Lbands = (2, 2)
    Rbands = (1, 1)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1], Lbands, 1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_JP6_JT4():
    """
    JP6 interior with the two JT4 boundary rows (fourth order at the edge).
    JP6's own second row drops the left beta term, so it is not consistent.
    """
    P, Q = _coeffs_D1_JTP6()
    P4, Q4 = _coeffs_D1_JTT4()
    P[1] = [P4[1][0], P4[0]]
    Q[1] = [Q4[1][0], Q4[0]]
    return P, Q


def _coeffs_D2_JP6_JT4():
    """Second-derivative counterpart of _coeffs_D1_JP6_JT4."""
    P, Q = _coeffs_D2_JTP6()
    P4, Q4 = _coeffs_D2_JTT4()
    P[1] = [P4[1][0], P4[0]]
    Q[1] = [Q4[1][0], Q4[0]]
    return P, Q


def _coeffs_D1_DSQ6A():
    """
    Coefficients for Boris Daszuta's Q_3 first derivative, optimized for eta = 1.
    See B. Daszuta, "Spectrally-tuned compact finite-diﬀerence schemes with domain
        decomposition and applications to numerical relativity", arXiv:2302.14858v1.
    See Table XII.
    """
    Lcoeff = np.array([0.37987923, 1.0, 0.37987923])
    Rcoeff = np.array(
        [
            0.0023272948,
            -0.052602255,
            -0.78165660,
            0.0,
            0.78165660,
            0.052602255,
            -0.0023272948,
        ]
    )
    Lcoeffb0 = np.array([1.0, 0.0])
    Lcoeffb1 = np.array([0.0, 1.0, 0.0])
    Lcoeffb2 = np.array([0.0, 0.0, 1.0, 0.0])
    Rcoeffb0 = np.array([-1.5, 2.0, -0.5])
    Rcoeffb1 = np.array([-0.5, 0.0, 0.5])
    Rcoeffb2 = np.array([1 / 12, -2.0 / 3, 0.0, 2.0 / 3, -1 / 12])
    Lbands = (1, 1)
    Rbands = (3, 3)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1, Lcoeffb2], Lbands, -1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1, Rcoeffb2], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D2_DSQ6A():
    """
    Coefficients for Boris Daszuta's Q_3 second derivative, optimized for eta = 1.
    See B. Daszuta, "Spectrally-tuned compact finite-diﬀerence schemes with domain
        decomposition and applications to numerical relativity", arXiv:2302.14858v1.
    See Table XIII.
    """
    Lcoeff = np.array([0.24246603, 1.0, 0.24246603])
    Rcoeff = np.array(
        [
            -0.0037062571,
            0.14095923,
            0.95445144,
            -2.1834080,
            0.95445144,
            0.14095923,
            -0.0037062571,
        ]
    )
    Lcoeffb0 = np.array([1.0, 0.0])
    Lcoeffb1 = np.array([0.0, 1.0, 0.0])
    Lcoeffb2 = np.array([0.0, 0.0, 1.0, 0.0])
    Rcoeffb0 = np.array([2.0, -5.0, 4.0, -1.0])
    Rcoeffb1 = np.array([1.0, -2.0, 1.0])
    Rcoeffb2 = np.array([-1, 16, -30, 16, -1]) / 12.0
    Lbands = (1, 1)
    Rbands = (3, 3)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1, Lcoeffb2], Lbands, 1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1, Rcoeffb2], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_DSQ6B():
    """
    Coefficients for Boris Daszuta's Q_3 first derivative, optimized for eta = 4*pi/5.
    See B. Daszuta, "Spectrally-tuned compact finite-diﬀerence schemes with domain
        decomposition and applications to numerical relativity", arXiv:2302.14858v1.
    See Table XII.
    """
    Lcoeff = np.array([0.41825851, 1.0, 0.41825851])
    Rcoeff = np.array(
        [
            0.0042462587,
            -0.073071204,
            -0.78485488,
            0.0,
            0.78485488,
            0.073071204,
            -0.0042462587,
        ]
    )
    Lcoeffb0 = np.array([1.0, 0.0])
    Lcoeffb1 = np.array([0.0, 1.0, 0.0])
    Lcoeffb2 = np.array([0.0, 0.0, 1.0, 0.0])
    Rcoeffb0 = np.array([-1.5, 2.0, -0.5])
    Rcoeffb1 = np.array([-0.5, 0.0, 0.5])
    Rcoeffb2 = np.array([1 / 12, -2.0 / 3, 0.0, 2.0 / 3, -1 / 12])
    Lbands = (1, 1)
    Rbands = (3, 3)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1, Lcoeffb2], Lbands, -1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1, Rcoeffb2], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D2_DSQ6B():
    """
    Coefficients for Boris Daszuta's Q_3 second derivative, optimized for eta = 4*pi/5.
    See B. Daszuta, "Spectrally-tuned compact finite-diﬀerence schemes with domain
        decomposition and applications to numerical relativity", arXiv:2302.14858v1.
    See Table XIII.
    """
    Lcoeff = np.array([0.28533501, 1.0, 0.28533501])
    Rcoeff = np.array(
        [
            -0.0063260285,
            0.19240201,
            0.85799622,
            -2.0881444,
            0.85799622,
            0.19240201,
            -0.0063260285,
        ]
    )
    Lcoeffb0 = np.array([1.0, 0.0])
    Lcoeffb1 = np.array([0.0, 1.0, 0.0])
    Lcoeffb2 = np.array([0.0, 0.0, 1.0, 0.0])
    Rcoeffb0 = np.array([2.0, -5.0, 4.0, -1.0])
    Rcoeffb1 = np.array([1.0, -2.0, 1.0])
    Rcoeffb2 = np.array([-1, 16, -30, 16, -1]) / 12.0
    Lbands = (1, 1)
    Rbands = (3, 3)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1, Lcoeffb2], Lbands, 1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1, Rcoeffb2], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_ME44():
    Lcoeff = np.array([1.0])
    Rcoeff = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0
    Lcoeffb0 = np.array([1.0])
    Lcoeffb1 = np.array([1.0])
    Rcoeffb0 = np.array([-25, 48, -36, 16, -3]) / 12.0
    Rcoeffb1 = np.array([-3.0, -10.0, 18.0, -6.0, 1.0]) / 12.0
    Lbands = (0, 0)
    Rbands = (2, 2)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1], Lbands, -1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D2_ME44():
    Lcoeff = np.array([1.0])
    Rcoeff = np.array([-1, 16, -30, 16, -1]) / 12.0
    Lcoeffb0 = np.array([1.0])
    Lcoeffb1 = np.array([1.0])
    Rcoeffb0 = np.array([45, -154, 214, -156, 61, -10]) / 12.0
    Rcoeffb1 = np.array([10, -15, -4, 14, -6, 1]) / 12.0
    Lbands = (0, 0)
    Rbands = (2, 2)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1], Lbands, 1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_ME642():
    Lcoeff = np.array([1.0])
    Rcoeff = np.array([-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]) / 60.0
    Lcoeffb0 = np.array([1.0])
    Lcoeffb1 = np.array([1.0])
    Lcoeffb2 = np.array([1.0])
    Rcoeffb0 = np.array([-1.5, 2.0, -0.5])
    Rcoeffb1 = np.array([-0.5, 0.0, 0.5])
    Rcoeffb2 = np.array([1 / 12, -2.0 / 3, 0.0, 2.0 / 3, -1 / 12])
    Lbands = (0, 0)
    Rbands = (3, 3)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1, Lcoeffb2], Lbands, -1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1, Rcoeffb2], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D2_MEQ642():
    Lcoeff = np.array([1.0])
    Rcoeff = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0]) / 180.0
    Lcoeffb0 = np.array([1.0])
    Lcoeffb1 = np.array([1.0])
    Lcoeffb2 = np.array([1.0])
    Rcoeffb0 = np.array([2.0, -5.0, 4.0, -1.0])
    Rcoeffb1 = np.array([1.0, -2.0, 1.0])
    Rcoeffb2 = np.array([-1, 16, -30, 16, -1]) / 12.0
    Lbands = (0, 0)
    Rbands = (3, 3)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1, Lcoeffb2], Lbands, 1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1, Rcoeffb2], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_KP4():
    """
    #  This is the 4th-order compact operator defined in Wu and Kim (2024).
    #  The terms are defined in Table 3.
    """
    alpha = 0.5862704032801503
    beta = 9.549533555017055e-2
    Lcoeff = np.array([beta, alpha, 1.0, alpha, beta])

    a1 = 0.6431406736919156
    a2 = 0.2586011023495066
    a3 = 7.140953479797375e-3
    Rcoeff = np.array([-a3, -a2, -a1, 0.0, a1, a2, a3])

    # i = 0
    alpha01 = 43.65980335321481
    beta02 = 92.40143116322876

    b01 = -86.92242000231872
    b02 = 47.58661913475775
    b03 = 57.30693626084370
    b04 = -13.71254216556246
    b05 = 2.659826729790792
    b06 = -0.2598929200600359

    # i = 1
    alpha10 = 0.08351537442980239
    alpha12 = 1.961483362670730
    beta13 = 0.8789761422182460

    b10 = -0.3199960780333493
    b12 = 0.07735499170041915
    b13 = 1.496612372811008
    b14 = 0.2046919801608821
    b15 = -0.02229717539815850
    b16 = 0.001702365014746567

    # i = 2
    beta20 = 0.008073091519768687
    alpha21 = 0.2162434143850924
    alpha23 = 1.052242062502679
    beta24 = 0.2116022463346598

    b20 = -0.03644974757120792
    b21 = -0.4997030280694729
    b23 = 0.7439822445654316
    b24 = 0.5629384925762924
    b25 = 0.01563884275691290
    b26 = -0.0003043666146108995

    b00 = -(b01 + b02 + b03 + b04 + b05 + b06)
    b11 = -(b10 + b12 + b13 + b14 + b15 + b16)
    b22 = -(b20 + b21 + b23 + b24 + b25 + b26)

    Lcoeffb0 = np.array([1.0, alpha01, beta02])
    Lcoeffb1 = np.array([alpha10, 1.0, alpha12, beta13])
    Lcoeffb2 = np.array([beta20, alpha21, 1.0, alpha23, beta24])

    Rcoeffb0 = np.array([b00, b01, b02, b03, b04, b05, b06])
    Rcoeffb1 = np.array([b10, b11, b12, b13, b14, b15, b16])
    Rcoeffb2 = np.array([b20, b21, b22, b23, b24, b25, b26])
    Lbands = (2, 2)
    Rbands = (3, 3)

    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1, Lcoeffb2], Lbands, -1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1, Rcoeffb2], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_BYUP6(optcoeffs1):
    beta = optcoeffs1[1]
    alpha0 = (1.0 + 12.0 * beta) / 3.0  # default value for alpha
    alpha = alpha0 + optcoeffs1[0]
    a06 = optcoeffs1[2]
    a16 = optcoeffs1[3]

    a1 = (-2.0 * (-7.0 + 12.0 * beta)) / 9.0
    a2 = (1.0 + 114.0 * beta) / 9.0
    a00 = (-227.0 + 600.0 * a06) / 60.0
    a01 = (-65.0 + 1644.0 * a06) / 6.0
    a02 = (-5.0 * (-7.0 + 75.0 * a06)) / 3.0
    a03 = (-10.0 * (-1.0 + 60.0 * a06)) / 3.0
    a04 = (5.0 * (-1.0 + 120.0 * a06)) / 12.0
    a05 = (1.0 - 300.0 * a06) / 30.0
    a10 = (-247.0 + 35460.0 * a16) / 900.0
    a11 = (-19.0 - 2592.0 * a16) / 12.0
    a12 = (1.0 - 1755.0 * a16) / 3.0
    a13 = (13.0 + 5760.0 * a16) / 9.0
    a14 = (1.0 + 1620.0 * a16) / 12.0
    a15 = (-1.0 - 4320.0 * a16) / 300.0
    gamma01 = -10.0 * (-1.0 + 12.0 * a06)
    gamma02 = -10.0 * (-1.0 + 30.0 * a06)
    gamma10 = (1.0 - 180.0 * a16) / 15.0
    gamma12 = 2.0 * (1.0 + 270.0 * a16)
    gamma13 = (2.0 * (1.0 + 720.0 * a16)) / 3.0

    # boundary elements for P matrix for 1st derivative
    P1DiagBoundary = [[1.0, gamma01, gamma02], [gamma10, 1.0, gamma12, gamma13]]

    # diagonal elements for P matrix for 1st derivative
    P1DiagInterior = [beta, alpha, 1.0, alpha, beta]

    # boundary elements for Q matrix for 1st derivative
    Q1DiagBoundary = [
        [a00, a01, a02, a03, a04, a05, a06],
        [a10, a11, a12, a13, a14, a15, a16],
    ]

    # diagonal elements for Q matrix for 1st derivative
    Q1DiagInterior = [-a2 / 4.0, -a1 / 2.0, 0.0, a1 / 2.0, a2 / 4.0]

    Lmat = [P1DiagInterior, P1DiagBoundary, (2, 2), 1]
    Rmat = [Q1DiagInterior, Q1DiagBoundary, (2, 2), -1]
    return Lmat, Rmat


def _coeffs_D2_BYUP6(optcoeffs2):
    beta = optcoeffs2[1]
    alpha0 = (2.0 * (1.0 + 62.0 * beta)) / 11.0  # default value for alpha
    alpha = alpha0 + optcoeffs2[0]
    a07 = optcoeffs2[2]
    a17 = optcoeffs2[3]

    a1 = (-12.0 * (-1.0 + 26.0 * beta)) / 11.0
    a2 = (3.0 * (1.0 + 194.0 * beta)) / 11.0
    a00 = (48241.0 - 549180.0 * a07) / 900.0
    a01 = (16389.0 - 262820.0 * a07) / 25.0
    a02 = (3.0 * (-10453.0 + 162000.0 * a07)) / 20.0
    a03 = (43622.0 - 667035.0 * a07) / 45.0
    a04 = (-2529.0 + 37220.0 * a07) / 20.0
    a05 = (-3.0 * (-143.0 + 1890.0 * a07)) / 25.0
    a06 = (-1169.0 + 9720.0 * a07) / 900.0
    a10 = (753829.0 - 24461820.0 * a17) / 1114560.0
    a11 = (57209.0 + 4218420.0 * a17) / 20640.0
    a12 = (-58367.0 - 8677836.0 * a17) / 8256.0
    a13 = (172793.0 + 92712420.0 * a17) / 55728.0
    a14 = (4453.0 - 7381500.0 * a17) / 8256.0
    a15 = (-391.0 + 2318580.0 * a17) / 20640.0
    a16 = (529.0 - 15888780.0 * a17) / 1114560.0
    gamma01 = (-36.0 * (-17.0 + 235.0 * a07)) / 5.0
    gamma02 = (-27.0 * (-113.0 + 1740.0 * a07)) / 5.0
    gamma10 = (23.0 - 1620.0 * a17) / 688.0
    gamma12 = (5.0 * (467.0 + 8028.0 * a17)) / 688.0
    gamma13 = (2659.0 - 1702980.0 * a17) / 3096.0

    # boundary elements for P matrix for 2nd derivative
    P2DiagBoundary = [[1.0, gamma01, gamma02], [gamma10, 1.0, gamma12, gamma13]]

    # diagonal elements for P matrix for 2nd derivative
    P2DiagInterior = [beta, alpha, 1.0, alpha, beta]

    # boundary elements for Q matrix for 2nd derivative
    Q2DiagBoundary = [
        [a00, a01, a02, a03, a04, a05, a06, a07],
        [a10, a11, a12, a13, a14, a15, a16, a17],
    ]

    # diagonal elements for Q matrix for 2nd derivative
    t1 = -2.0 * (a1 / 1.0 + a2 / 4.0)
    Q2DiagInterior = [-a2 / 4.0, -a1 / 1.0, t1, a1 / 1.0, a2 / 4.0]

    Lmat = [P2DiagInterior, P2DiagBoundary, (2, 2), 1]
    Rmat = [Q2DiagInterior, Q2DiagBoundary, (2, 2), 1]
    return Lmat, Rmat


def _coeffs_D1_Wm6():
    Lcoeff = np.array([829 / 1200, 1.0, 0.0])
    Rcoeff = np.array(
        [
            29 / 12000,
            -877 / 12000,
            -24271 / 14400,
            151 / 80,
            -307 / 2400,
            -167 / 36000,
            29 / 24000,
        ]
    )

    # JTT4 coefficients for boundary conditions
    Lbcoeff0 = np.array([1.0, 3.0])
    Rbcoeff0 = np.array([-17.0, 9.0, 9.0, -1.0]) / 6.0
    Lbcoeff1 = np.array([0.25, 1.0, 0.25])
    Rbcoeff1 = np.array([-3.0 / 4.0, 0, 3.0 / 4.0])

    # V_m2, O(h^5) coefficients for boundary conditions
    Lbcoeff2 = np.array([0, 2 / 3, 1.0, 0.0])
    Rbcoeff2 = np.array([-3 / 36, -44 / 36, 1, 12 / 36, -1 / 36])

    Lmat = [Lcoeff, [Lbcoeff0, Lbcoeff1, Lbcoeff2], (1, 1), 1]
    Rmat = [Rcoeff, [Rbcoeff0, Rbcoeff1, Rbcoeff2], (3, 3), -1]
    return Lmat, Rmat


def _coeffs_D1_X6_RIGHT():
    Lcoeff = np.array([829 / 1200, 1.0, 0.0])
    Rcoeff = np.array(
        [
            29 / 12000,
            -877 / 12000,
            -24271 / 14400,
            151 / 80,
            -307 / 2400,
            -167 / 36000,
            29 / 24000,
        ]
    )

    # JTT4 coefficients for boundary conditions
    Lbcoeff0 = np.array([1.0, 3.0])
    Rbcoeff0 = np.array([-17.0, 9.0, 9.0, -1.0]) / 6.0
    Lbcoeff1 = np.array([0.25, 1.0, 0.25])
    Rbcoeff1 = np.array([-3.0 / 4.0, 0, 3.0 / 4.0])

    # V_m2, O(h^5) coefficients for boundary conditions
    Lbcoeff2 = np.array([0, 2 / 3, 1.0, 0.0])
    Rbcoeff2 = np.array([-3 / 36, -44 / 36, 1, 12 / 36, -1 / 36])

    Lmat = [Lcoeff, [Lbcoeff0, Lbcoeff1, Lbcoeff2], (1, 1), 1]
    Rmat = [Rcoeff, [Rbcoeff0, Rbcoeff1, Rbcoeff2], (3, 3), -1]
    return Lmat, Rmat


def _coeffs_D1_DSQ6B_LEFT():
    """
    Coefficients for the left side of the first derivative operator in the DSQ6B scheme.
    """
    Lcoeff = np.array([0.0, 1.0, 0.71391655])
    Rcoeff = np.array(
        [
            0.0044106875,
            -0.035642870,
            0.147435560,
            -0.54741579,
            -0.79751467,
            1.0738269,
            0.16644143,
            -0.011541239,
            0.0,
        ]
    )

    Lcoeffb0 = np.array([1.0, 0.0])
    Lcoeffb1 = np.array([0.0, 1.0, 0.0])
    Lcoeffb2 = np.array([0.0, 0.0, 1.0, 0.0])
    Lcoeffb3 = np.array([0.0, 0.0, 0.0, 1.0, 0.0])

    Rcoeffb0 = np.array([-1.0, 4.0, -3.0]) / 2.0
    Rcoeffb1 = np.array([-1.0, 0.0, 1.0]) / 2.0
    Rcoeffb2 = np.array([-1.0, 6.0, -18.0, 10.0, 3.0]) / 12.0
    Rcoeffb3 = np.array([-1.0, 6.0, -18.0, 10.0, 3.0]) / 12.0

    Lbands = (1, 1)
    Rbands = (4, 4)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1, Lcoeffb2, Lcoeffb3], Lbands, -1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1, Rcoeffb2, Rcoeffb3], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_DSQ6B_RIGHT():
    """
    Coefficients for the left side of the first derivative operator in the DSQ6B scheme.
    """
    Lcoeff = np.array([0.71391655, 1.0, 0.0])
    Rcoeff = np.array(
        [
            0.0,
            0.011541239,
            -0.16644143,
            -1.0738269,
            0.79751467,
            0.54741579,
            -0.147435560,
            0.035642870,
            -0.0044106875,
        ]
    )

    Lcoeffb0 = np.array([1.0, 0.0])
    Lcoeffb1 = np.array([0.0, 1.0, 0.0])
    Lcoeffb2 = np.array([0.0, 0.0, 1.0, 0.0])
    Lcoeffb3 = np.array([0.0, 0.0, 0.7139165, 1.0])

    Rcoeffb0 = np.array([-3.0, 4.0, -1.0]) / 2.0
    Rcoeffb1 = np.array([0.0, -3.0, 4.0, -1.0]) / 2.0
    Rcoeffb2 = np.array([0.0, -3.0, -10.0, 18.0, -6.0, 1.0]) / 12.0
    Rcoeffb3 = Rcoeff[1:]
    Lbands = (1, 1)
    Rbands = (4, 4)
    Lmat = [Lcoeff, [Lcoeffb0, Lcoeffb1, Lcoeffb2, Lcoeffb3], Lbands, -1]
    Rmat = [Rcoeff, [Rcoeffb0, Rcoeffb1, Rcoeffb2, Rcoeffb3], Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_BYU_C6_Op10(D_coeffs):
    """
    Coefficients for the first derivative operator in the BYU C6 Op10 scheme.
    """
    alpha0 = 0.4963477370017426
    alpha = alpha0 + D_coeffs[0]

    beta = 0.04075360091710232
    a1 = 0.723439643221641
    a2 = 0.15683084734860173
    gamma01 = 6.41607857818298
    gamma02 = 3.624117867379136
    gamma10 = 0.05045368184220923
    gamma12 = 1.7003969127106167
    gamma13 = 0.3354126691761755
    a00 = -3.3853431556471
    a01 = -3.762810726379067
    a02 = 6.624117866996977
    a03 = 0.541372622221082
    a04 = -0.0173366074130172
    a10 = -0.24136315021013707
    a11 = -1.5974102131837702
    a12 = 0.8455199507089511
    a13 = 0.9803800752434986
    a14 = 0.012873337440911644

    # boundary elements for P matrix for 1st derivative
    Lcoeffb = [
        np.array([1.0, gamma01, gamma02]),
        np.array([gamma10, 1.0, gamma12, gamma13]),
    ]

    # diagonal elements for P matrix for 1st derivative
    Lcoeff = np.array([beta, alpha, 1.0, alpha, beta])

    # boundary elements for Q matrix for 1st derivative
    Rcoeffb = [np.array([a00, a01, a02, a03, a04]), np.array([a10, a11, a12, a13, a14])]

    # diagonal elements for Q matrix for 1st derivative
    Rcoeff = np.array([-a2, -a1, 0.0, a1, a2])
    Lbands = (2, 2)
    Rbands = (2, 2)
    Lmat = [Lcoeff, Lcoeffb, Lbands, -1]
    Rmat = [Rcoeff, Rcoeffb, Rbands, 1]
    return Lmat, Rmat


def _coeffs_D1_BYU_P4_Op18(D_coeffs):
    alpha0 = 0.5770460201292381
    alpha = alpha0 + D_coeffs[0]

    beta = 0.08900836845092142
    a1 = 0.6517078440898839
    a2 = 0.24815882627035918
    a3 = 0.006009630649852392
    gamma01 = 7.382379779647204
    gamma02 = 6.118391304701528
    gamma10 = 0.06106149690079044
    gamma12 = 2.6424801172618326
    gamma13 = 1.3710069602113883
    gamma20 = 0.014504838045567564
    gamma21 = 0.2617273578999154
    gamma23 = 1.110302482205355
    gamma24 = 0.3070536064807964
    a00 = -3.4387560411379563
    a01 = -6.120972374567166
    a02 = 7.819283685673539
    a03 = 2.0327058980606223
    a04 = -0.3578340204155545
    a05 = 0.07704263923889951
    a06 = -0.011469786857726549
    a10 = -0.25047387349211636
    a11 = -1.7715514547125182
    a12 = -0.5198462724609013
    a13 = 2.2536920432047496
    a14 = 0.318588110606521
    a15 = -0.0328153209921653
    a16 = 0.0024067678049133558
    a20 = -0.058823348930518396
    a21 = -0.5249167357886487
    a22 = -0.714498395784144
    a23 = 0.5832981939847537
    a24 = 0.6747270495592782
    a25 = 0.0425803899581001
    a26 = -0.0023671529988773697

    # boundary elements for P matrix for 1st derivative
    P1DiagBoundary = [
        np.array([1.0, gamma01, gamma02]),
        np.array([gamma10, 1.0, gamma12, gamma13]),
        np.array([gamma20, gamma21, 1.0, gamma23, gamma24]),
    ]

    # diagonal elements for P matrix for 1st derivative
    P1DiagInterior = np.array([beta, alpha, 1.0, alpha, beta])

    # boundary elements for Q matrix for 1st derivative
    Q1DiagBoundary = [
        np.array([a00, a01, a02, a03, a04, a05, a06]),
        np.array([a10, a11, a12, a13, a14, a15, a16]),
        np.array([a20, a21, a22, a23, a24, a25, a26]),
    ]

    # diagonal elements for Q matrix for 1st derivative
    Q1DiagInterior = np.array([-a3, -a2, -a1, 0.0, a1, a2, a3])

    Lmat = [P1DiagInterior, P1DiagBoundary, (2, 2), -1]
    Rmat = [Q1DiagInterior, Q1DiagBoundary, (3, 3), 1]
    return Lmat, Rmat

# The following operators were added by Luke
def _coeffs_D1_C6_4():
    alpha0 = 0.4963477370017426
    alpha = alpha0
    beta = 0.04075360091710232
    a1 = 0.723439643221641
    a2 = 0.15683084734860173
    gamma01 =  - 1.2310441501678786
    gamma02 =  - 7.846566219762
    gamma10 = 0.16123050668544858
    gamma12 = 0.4741493035858441
    gamma13 =  - 0.038978437534430796
    a00 =  - 2.4294528173078316
    a01 = 10.25691429353829
    a02 =  - 4.84656624284059
    a03 =  - 3.282188754146896
    a04 = 0.3012935069894299
    a10 =  - 0.5431362438346228
    a11 =  - 0.5240000610825333
    a12 = 1.0747761362453676
    a13 =  - 0.0014084866408688862
    a14 =  - 0.006231344687109832

    # boundary elements for P matrix for 1st derivative
    P1DiagBoundary = [
        np.array([1.0, gamma01, gamma02]),
        np.array([gamma10, 1.0, gamma12, gamma13]),
    ]

    # diagonal elements for P matrix for 1st derivative
    P1DiagInterior = np.array([beta, alpha, 1.0, alpha, beta])

    # boundary elements for Q matrix for 1st derivative
    Q1DiagBoundary = [
        np.array([a00, a01, a02, a03, a04]),
        np.array([a10, a11, a12, a13, a14])
    ]

    # diagonal elements for Q matrix for 1st derivative
    Q1DiagInterior = np.array([-a2, -a1, 0.0, a1, a2])

    Lmat = [P1DiagInterior, P1DiagBoundary, (2, 2), -1]
    Rmat = [Q1DiagInterior, Q1DiagBoundary, (2, 2), 1]
    return Lmat, Rmat

def _coeffs_D1_C4_1():
    alpha0 = 0.5362930888943283
    alpha = alpha0

    beta = 0.05831696520179083
    a1 = 0.6899191224266673
    a2 = 0.20234546583472587
    gamma01 = 5.887245740617931
    gamma02 = 3.4740401606206683
    gamma10 = 0.0481407122009377
    gamma12 = 1.7713756159896565
    gamma13 = 0.4570840961794273
    a00 =  - 3.26564142176947
    a01 =  - 3.222064890928744
    a02 = 5.830868610926915
    a03 = 0.705737236775412
    a04 =  - 0.04889953500022839
    a10 =  - 0.24076885710110105
    a11 =  - 1.593145513766301
    a12 = 0.6699517191280673
    a13 = 1.1260081070772001
    a14 = 0.03795454466215297

    # boundary elements for P matrix for 1st derivative
    P1DiagBoundary = [
        np.array([1.0, gamma01, gamma02]),
        np.array([gamma10, 1.0, gamma12, gamma13]),
    ]

    # diagonal elements for P matrix for 1st derivative
    P1DiagInterior = np.array([beta, alpha, 1.0, alpha, beta])

    # boundary elements for Q matrix for 1st derivative
    Q1DiagBoundary = [
        np.array([a00, a01, a02, a03, a04]),
        np.array([a10, a11, a12, a13, a14])
    ]

    # diagonal elements for Q matrix for 1st derivative
    Q1DiagInterior = np.array([-a2, -a1, 0.0, a1, a2])

    Lmat = [P1DiagInterior, P1DiagBoundary, (2, 2), -1]
    Rmat = [Q1DiagInterior, Q1DiagBoundary, (2, 2), 1]
    return Lmat, Rmat

def _coeffs_D1_A4_3():
    alpha0 = 0.5770460201292381
    alpha = alpha0 

    beta = 0.08900836845092142
    a1 = 0.6517078440898839
    a2 = 0.24815882627035918
    a3 = 0.006009630649852392
    gamma01 = 11.720569036874554
    gamma02 = 15.554462579626016
    gamma10 = 0.1081072833795741
    gamma12 = 0.9879415284429277
    gamma13 =  - 0.019091049862053604
    gamma20 = 0.001594503696184088
    gamma21 = 0.11896661542190605
    gamma23 = 1.4712607824914257
    gamma24 = 0.39065672002582436
    a00 =  - 3.869256849728272
    a01 =  - 15.34137988312291
    a02 = 12.883637155318173
    a03 = 7.717258677339867
    a04 =  - 1.6841269756473085
    a05 = 0.32933821713218175
    a06 =  - 0.035470348869858045
    a10 =  - 0.3964964225533291
    a11 =  - 1.0420535477054953
    a12 = 1.1470798670545719
    a13 = 0.3494082667125378
    a14 =  - 0.06737997031063933
    a15 = 0.010504184266465403
    a16 =  - 0.0010623774453651048
    a20 =  - 0.008283084846223839
    a21 =  - 0.3750015455221021
    a22 =  - 1.2054305952790025
    a23 = 0.6249651589579047
    a24 = 0.9236932467174234
    a25 = 0.04166802585994469
    a26 =  - 0.0016112058874788778

    # boundary elements for P matrix for 1st derivative
    P1DiagBoundary = [
        np.array([1.0, gamma01, gamma02]),
        np.array([gamma10, 1.0, gamma12, gamma13]),
        np.array([gamma20, gamma21, 1.0, gamma23, gamma24])
    ]

    # diagonal elements for P matrix for 1st derivative
    P1DiagInterior = np.array([beta, alpha, 1.0, alpha, beta])

    # boundary elements for Q matrix for 1st derivative
    Q1DiagBoundary = [
        np.array([a00, a01, a02, a03, a04, a05, a06]),
        np.array([a10, a11, a12, a13, a14, a15, a16]),
        np.array([a20, a21, a22, a23, a24, a25, a26])
    ]

    # diagonal elements for Q matrix for 1st derivative
    Q1DiagInterior = np.array([-a3, -a2, -a1, 0.0, a1, a2, a3])

    Lmat = [P1DiagInterior, P1DiagBoundary, (2, 2), -1]
    Rmat = [Q1DiagInterior, Q1DiagBoundary, (3, 3), 1]
    return Lmat, Rmat


def _coeffs_D2_2C4_1():
    alpha0 = 0.39755582066409934
    alpha = alpha0

    beta = 0.021185375111255712
    a1 = 0.5207872376311244
    a2 = 0.32917378847989637
    gamma01 = 8.684466815062526
    gamma02 = 5.2334494385813395
    gamma10 =  - 0.2908610497433836
    gamma12 = 19.151728922470173
    gamma13 = 3.7205127814515646
    a00 = 10.44130712725852
    a01 =  - 16.162845440315607
    a02 = 0.758609811076265
    a03 = 5.206088189802421
    a04 =  - 0.24315968780464542
    a10 =  - 1.8376982037450174
    a11 = 27.629938588220078
    a12 =  - 48.28224588801183
    a13 = 21.025468826343424
    a14 = 1.4645366771933166

    # boundary elements for P matrix for 1st derivative
    P1DiagBoundary = [
        np.array([1.0, gamma01, gamma02]),
        np.array([gamma10, 1.0, gamma12, gamma13])
    ]

    # diagonal elements for P matrix for 1st derivative
    P1DiagInterior = np.array([beta, alpha, 1.0, alpha, beta])

    # boundary elements for Q matrix for 1st derivative
    Q1DiagBoundary = [
        np.array([a00, a01, a02, a03, a04]),
        np.array([a10, a11, a12, a13, a14])
    ]

    t1 = -2.0 * (a1 + a2)

    # diagonal elements for Q matrix for 1st derivative
    Q1DiagInterior = np.array([a2, a1, t1, a1, a2])

    Lmat = [P1DiagInterior, P1DiagBoundary, (2, 2), 1]
    Rmat = [Q1DiagInterior, Q1DiagBoundary, (2, 2), 1]
    return Lmat, Rmat

def _coeffs_D2_2C6_1():
    alpha0 = 0.3552519833863128
    alpha = alpha0 

    beta = 0.015385256590721305
    a1 = 0.6545272676086319
    a2 = 0.27168680308635906
    gamma01 = 8.389045509463868
    gamma02 = 4.997784937575292
    gamma10 = 0.07252339307985996
    gamma12 = 2.274638288052355
    gamma13 = 0.3624606194793401
    a00 = 10.190142972214264
    a01 =  - 15.984695932338362
    a02 = 1.2000604108003015
    a03 = 4.793395086593271
    a04 =  - 0.19890253725298698
    a10 = 0.9084349875219685
    a11 = 0.8584685172041276
    a12 =  - 4.316393176132551
    a13 = 2.423640850564837
    a14 = 0.12584882084157425

    # boundary elements for P matrix for 1st derivative
    P1DiagBoundary = [
        np.array([1.0, gamma01, gamma02]),
        np.array([gamma10, 1.0, gamma12, gamma13])
    ]

    # diagonal elements for P matrix for 1st derivative
    P1DiagInterior = np.array([beta, alpha, 1.0, alpha, beta])

    # boundary elements for Q matrix for 1st derivative
    Q1DiagBoundary = [
        np.array([a00, a01, a02, a03, a04]),
        np.array([a10, a11, a12, a13, a14])
    ]

    t1 = -2.0 * (a1 + a2)

    # diagonal elements for Q matrix for 1st derivative
    Q1DiagInterior = np.array([a2, a1, t1, a1, a2])

    Lmat = [P1DiagInterior, P1DiagBoundary, (2, 2), 1]
    Rmat = [Q1DiagInterior, Q1DiagBoundary, (2, 2), 1]
    return Lmat, Rmat

def _coeffs_D2_2A_1():
    alpha0 = 0.4917593294116515
    alpha = alpha0 
    beta = 0.052107304747970096
    a1 = 0.2456096922476715
    a2 = 0.42112296480247163
    a3 = 0.017514635206853885
    gamma01 = 12.591315362048217
    gamma02 = 15.084784378502926
    gamma10 = 0.053356389606600524
    gamma12 = 2.354755322703094
    gamma13 = 0.6916927010503097
    gamma20 = 0.01459587668090503
    gamma21 = 0.3251836724513594
    gamma23 = 0.6353132625404602
    gamma24 = 0.09603512518086209
    a00 = 12.874331545332872
    a01 =  - 7.791236348900909
    a02 =  - 25.746221106082196
    a03 = 24.030231590355
    a04 =  - 4.139878085038264
    a05 = 0.8841382958263313
    a06 =  - 0.11136589149817189
    a10 = 0.8369729348468714
    a11 = 1.1485958676212564
    a12 =  - 4.349701163379025
    a13 = 1.8878205985135512
    a14 = 0.49399689902406285
    a15 =  - 0.01746788736710351
    a16 =  - 0.00021724928766294311
    a20 = 0.24876323785361718
    a21 = 0.6483433806079487
    a22 =  - 1.4133854015241178
    a23 =  - 0.06478123358362578
    a24 = 0.5309016728653824
    a25 = 0.051785630955101844
    a26 =  - 0.0016272871743160185

    # boundary elements for P matrix for 1st derivative
    P1DiagBoundary = [
        np.array([1.0, gamma01, gamma02]),
        np.array([gamma10, 1.0, gamma12, gamma13]),
        np.array([gamma20, gamma21, 1.0, gamma23, gamma24])
    ]

    # diagonal elements for P matrix for 1st derivative
    P1DiagInterior = np.array([beta, alpha, 1.0, alpha, beta])

    # boundary elements for Q matrix for 1st derivative
    Q1DiagBoundary = [
        np.array([a00, a01, a02, a03, a04, a05, a06]),
        np.array([a10, a11, a12, a13, a14, a15, a16]),
        np.array([a20, a21, a22, a23, a24, a25, a26])
    ]

    t1 = -2.0 * (a1 + a2 + a3)

    # diagonal elements for Q matrix for 1st derivative
    Q1DiagInterior = np.array([a3, a2, a1, t1, a1, a2, a3])

    Lmat = [P1DiagInterior, P1DiagBoundary, (2, 2), 1]
    Rmat = [Q1DiagInterior, Q1DiagBoundary, (3, 3), 1]
    return Lmat, Rmat


def build_derivative(N: int, dtype: DerivType):
    """
    Build the banded matrix for the specified derivative type.

    Parameters:
    N (int): Size of the matrix.
    dtype (DerivType): Type of derivative to build.

    Returns:
    np.ndarray: Banded matrix for the specified derivative.
    """
    if dtype == DerivType.D1_JT4:
        Pmat, Qmat = _coeffs_D1_JTT4()
    elif dtype == DerivType.D2_JT4:
        Pmat, Qmat = _coeffs_D2_JTT4()
    elif dtype == DerivType.D1_JT6:
        Pmat, Qmat = _coeffs_D1_JTT6()
    elif dtype == DerivType.D2_JT6:
        Pmat, Qmat = _coeffs_D2_JTT6()
    elif dtype == DerivType.D1_JP6:
        Pmat, Qmat = _coeffs_D1_JTP6()
    elif dtype == DerivType.D2_JP6:
        Pmat, Qmat = _coeffs_D2_JTP6()
    elif dtype == DerivType.D1_JP6_JT4:
        Pmat, Qmat = _coeffs_D1_JP6_JT4()
    elif dtype == DerivType.D2_JP6_JT4:
        Pmat, Qmat = _coeffs_D2_JP6_JT4()
    elif dtype == DerivType.D1_KP4:
        Pmat, Qmat = _coeffs_D1_KP4()
    elif dtype == DerivType.D1_BYUP6:
        Pmat, Qmat = _coeffs_D1_BYUP6([0.0, 0.0, 0.0, 0.0])
    elif dtype == DerivType.D2_BYUP6:
        Pmat, Qmat = _coeffs_D2_BYUP6([0.0, 0.0, 0.0, 0.0])
    elif dtype == DerivType.D1_Wm6:
        Pmat, Qmat = _coeffs_D1_Wm6()
    elif dtype == DerivType.D1_DSQ6A:
        Pmat, Qmat = _coeffs_D1_DSQ6A()
    elif dtype == DerivType.D2_DSQ6A:
        Pmat, Qmat = _coeffs_D2_DSQ6A()
    elif dtype == DerivType.D1_DSQ6B:
        Pmat, Qmat = _coeffs_D1_DSQ6B()
    elif dtype == DerivType.D2_DSQ6B:
        Pmat, Qmat = _coeffs_D2_DSQ6B()
    elif dtype == DerivType.D1_ME44:
        Pmat, Qmat = _coeffs_D1_ME44()
    elif dtype == DerivType.D2_ME44:
        Pmat, Qmat = _coeffs_D2_ME44()
    elif dtype == DerivType.D1_ME642:
        Pmat, Qmat = _coeffs_D1_ME642()
    elif dtype == DerivType.D2_ME642:
        Pmat, Qmat = _coeffs_D2_MEQ642()
    elif dtype == DerivType.D1_DSQ6B_LEFT:
        Pmat, Qmat = _coeffs_D1_DSQ6B_LEFT()
    elif dtype == DerivType.D1_DSQ6B_RIGHT:
        Pmat, Qmat = _coeffs_D1_DSQ6B_RIGHT()
    elif dtype == DerivType.D1_BYU_C6_Op10:
        Pmat, Qmat = _coeffs_D1_BYU_C6_Op10([0.0])
    elif dtype == DerivType.D1_BYU_P4_Op18:
        Pmat, Qmat = _coeffs_D1_BYU_P4_Op18([0.0])
    elif dtype == DerivType.D1_C6_4:
        Pmat, Qmat = _coeffs_D1_C6_4()
    elif dtype == DerivType.D1_C4_1:
        Pmat, Qmat = _coeffs_D1_C4_1()
    elif dtype == DerivType.D1_A4_3:
        Pmat, Qmat = _coeffs_D1_A4_3()
    elif dtype == DerivType.D2_2C4_1:
        Pmat, Qmat = _coeffs_D2_2C4_1()
    elif dtype == DerivType.D2_2C6_1:
        Pmat, Qmat = _coeffs_D2_2C6_1()
    elif dtype == DerivType.D2_2A_1:
        Pmat, Qmat = _coeffs_D2_2A_1()
    else:
        raise ValueError(f"Unsupported derivative type: {dtype}")

    explicit_derivs = [
        DerivType.D1_ME44,
        DerivType.D2_ME44,
        DerivType.D1_ME642,
        DerivType.D2_ME642,
    ]
    if dtype in explicit_derivs:
        P = np.ones(N)
        pbands = (0, 0)
    else:
        pparity = Pmat[3]
        pbands = Pmat[2]
        P, nlost = construct_banded_storage_numba(
            N, pbands[0], pbands[1], Pmat[0], Pmat[1], pparity, pbands[0], pbands[1]
        )
        if nlost > 0:
            raise ValueError(f"{dtype}: {nlost} P entries fall outside bands {pbands}.")

    # count the number of bands for the boundary terms in the Q matrix
    nqb = 0
    for iq in range(len(Qmat[1])):
        nqb = max(nqb, len(Qmat[1][iq]) - 1 - iq)

    qparity = Qmat[3]
    qbands = Qmat[2]
    # set the total number of bands for the Q matrix
    qtotbands = (max(nqb, qbands[0]), max(nqb, qbands[1]))

    Qb, nlost = construct_banded_storage_numba(
        N, qbands[0], qbands[1], Qmat[0], Qmat[1], qparity, qtotbands[0], qtotbands[1]
    )
    if nlost > 0:
        raise ValueError(
            f"{dtype}: {nlost} Q entries fall outside bands {qtotbands}."
        )

    return [P, pbands], [Qb, qtotbands]


def build_bh_derivative(N: int, dtype0: DerivType, dtypeBH: DerivType, mask):
    """
    Build a filter for boundary handling based on the specified type and parameters.
    Parameters:
        N (int): Size of the matrix.
        dtype0 (DerivType): Type of derivative for the interior region.
        dtypeBH (DerivType): Type of derivative for the boundary handling region.
        lambda (float): Mask function.  lambda = 1.0 corresponds to the background
                                        lambda = 0.0 corresponds to the BH region.
    Returns:
        tuple: Banded matrices for the interior and boundary handling regions.
    """
    P0mat, Q0mat = build_derivative(N, dtype0)
    p0bands = P0mat[1]
    q0bands = Q0mat[1]
    P0 = P0mat[0]
    Q0 = Q0mat[0]

    PBHmat, QBHmat = build_derivative(N, dtypeBH)
    pbhbands = PBHmat[1]
    qbhbands = QBHmat[1]
    PBH = PBHmat[0]
    QBH = QBHmat[0]

    # Blend row-wise in banded storage: the dense N x N version of this needed
    # six full matrices at once (4.8 GB at N = 10001).
    pbands = (max(p0bands[0], pbhbands[0]), max(p0bands[1], pbhbands[1]))
    qbands = (max(q0bands[0], qbhbands[0]), max(q0bands[1], qbhbands[1]))
    Pb = blend_banded_rows(
        pad_banded(P0, *p0bands, *pbands), pad_banded(PBH, *pbhbands, *pbands),
        mask, *pbands,
    )
    Qb = blend_banded_rows(
        pad_banded(Q0, *q0bands, *qbands), pad_banded(QBH, *qbhbands, *qbands),
        mask, *qbands,
    )

    return [Pb, pbands], [Qb, qbands]
