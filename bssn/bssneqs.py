import numpy as np
import sys
import os
import csv
from enum import Enum
from numba import njit

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nwave import *

DEBUG = False

class BSSNOuterBCType(Enum):
    """
    Enum for outer boundary condition
    """
    NONE = 0
    SOMMERFELD = 1
    SPONGE = 2

bssn_outer_bc_map = {
    "Sommerfeld" : BSSNOuterBCType.SOMMERFELD,
    "Sponge" : BSSNOuterBCType.SPONGE,
}

class GBSSNSystem:
    """
    Class to define the system of equations for the generalized BSSN equations.

    Parameters
    ----------
    eqs : int
        System of equations to use. 0 for Eulerian, 1 for Lagrangian.
    lapse : int
        Lapse system to use. 0 for no advection terms, 1 to include advection terms.
    shift : int
        Shift system to use. 0 for no advection terms, 1 to include advection terms.
    """

    def __init__(self, eqs, lapse, shift):
        if eqs not in [0, 1]:
            raise ValueError("Invalid equations system. Must be 0 or 1.")
        if lapse not in [0, 1]:
            raise ValueError("Invalid lapse system. Must be 0 or 1.")
        if shift not in [0, 1]:
            raise ValueError("Invalid shift system. Must be 0 or 1.")

        self.eqs = eqs
        self.lapse = lapse
        self.shift = shift


class BSSN(Equations):
    """
    The BSSN equations in 1D.

    The equations are from the paper "BSSN in Spherical Symmetry" by J. David Brown (2008).
    """

    def __init__(
        self,
        g,
        M,
        eta,
        extended_domain,
        apply_bc=None,
        gbssn_system=GBSSNSystem(1, 1, 1),
        outer_bc=BSSNOuterBCType.SOMMERFELD,
        have_d2=True,
    ):
        """
        Initialize the BSSN equations.
        Parameters
        ----------
        g : Grid1D
            Grid object.
        M : float
            Mass of the black hole.
        eta : float
            Damping parameter.
        apply_bc : "FUNCTION" or None
            If "FUNCTION", apply the boundary conditions using the function in each post-RK step.
        gbssn_system : GBSSNSystem
            System of equations to use.
        outer_bc : outer boundary condition type
        """

        NU = 9
        super().__init__(NU, g, apply_bc)

        self.gbssn_system = gbssn_system
        self.eta = eta
        self.M = M
        self.extended_domain = extended_domain
        self.have_d2 = have_d2
        self.r_horizon = 0.5 * M
        self.M_horizon = M

        self.U_CHI = 0
        self.U_GRR = 1
        self.U_GTT = 2
        self.U_ARR = 3
        self.U_K = 4
        self.U_GT = 5
        self.U_ALPHA = 6
        self.U_SHIFT = 7
        self.U_GB = 8

        nx = g.shp[0]
        self.Nc = 3
        self.C = []
        for i in range(self.Nc):
            self.C.append(np.zeros(nx))

        self.C_HAM = 0
        self.MOM = 1
        self.GAMCON = 2
        self.u_names = ["chi", "grr", "gtt", "Arr", "K", "Gt", "alpha", "beta", "gB"]
        self.c_names = ["ham", "mom", "gamcom"]
        self.outer_bc = outer_bc
        self.u_falloff = [
            1,
            1,
            -2,  # chi, grr, gtt
            2,
            2,
            1,  # Arr, K, Gt
            1,
            1,
            1,
        ]  # alpha, beta, gB
        self.u_inf = [
            1,
            1,
            1,  # chi, grr, gtt
            0,
            0,
            0,  # Arr, K, Gt
            1,
            0,
            0,
        ]  # alpha, beta, gB
        # Outer-boundary characteristic (wave) speeds per field, from Brown's
        # GBSSN characteristic analysis (papers/BrownSpherical/paper2.tex:370-404).
        # Proper speeds: physical/gravitational modes = 1; the 1+log gauge (lapse)
        # mode = sqrt(2/alpha); shift (Lagrangian Gamma-driver, case L^+_+) =
        # sqrt(1/(alpha^2 chi)). The radiative BC is only valid as r->inf, where
        # alpha->1, chi->1, g_rr->1, so all reduce to: v=1 for the geometric
        # fields and v=sqrt(2) for the lapse (sqrt(2/alpha)->sqrt2).
        self.u_wavespeed = np.ones(len(self.u_names))
        self.u_wavespeed[self.U_ALPHA] = np.sqrt(2.0)
        # u_reference is the reference solution at the outer boundary
        # for a sponge filter.
        r = g.xi[0]
        rmax = r[-1]
        self.u_reference = np.zeros(NU)
        self.u_reference[self.U_ALPHA] = 1.0 / (1.0 + self.M / (2 * abs(rmax))) ** 2
        self.u_reference[self.U_CHI] = 1.0 / (1.0 + self.M / (2 * abs(rmax))) ** (4)
        self.u_reference[self.U_GRR] = 1.0
        self.u_reference[self.U_GTT] = rmax**2
        self.u_reference[self.U_GT] = -2.0 / rmax

        self.u_parity = np.ones(NU)
        self.u_parity[self.U_GT] = -1.0
        self.u_parity[self.U_SHIFT] = -1.0
        self.u_parity[self.U_GB] = -1.0


    def get_r_horizon(self):
        return self.r_horizon

    def get_mass_horizon(self):
        return self.M_horizon

    def get_ah(self):
        return self.r_horizon, self.M_horizon

    def set_ah(self, rbh, mbh):
        self.r_horizon = rbh
        self.M_horizon = mbh

    def initialize(self, g: Grid, params):
        """
        Initial data for the BSSN equations.  This is a simple
        Schwarzschild solution in isotropic coordinates.
        """
        if params["initial_data"] == "Puncture":
            self.initial_data_puncture(g, params)
            self.reference_solution = []
        elif params["initial_data"] == "EddingtonFinkelstein":
            if self.gbssn_system.eqs == 0:
                raise ValueError(
                    "Eddington-Finkelstein initial data is not valid for Eulerian system."
                )
            self.initial_data_ef(g)
        else:
            raise ValueError(
                "Invalid initial data. Must be 'puncture' or 'EddingtonFinkelstein'."
            )

    def initial_data_puncture(self, g: Grid, params):
        """
        Initial data for the BSSN equations.  This is a simple
        Schwarzschild solution in isotropic coordinates.
        """
        r = g.xi[0]
        if params["collapsed_lapse"] == True:
            self.u[self.U_ALPHA][:] = 1.0 / (1.0 + self.M / (2 * abs(r))) ** 2
        else:
            self.u[self.U_ALPHA].fill(1.0)

        self.u[self.U_CHI][:] = 1.0 / (1.0 + self.M / (2 * abs(r))) ** (4)
        self.u[self.U_GRR][:] = 1.0
        self.u[self.U_GTT][:] = r**2
        self.u[self.U_ARR][:] = 0.0
        self.u[self.U_K][:] = 0.0
        self.u[self.U_GT][:] = -2.0 / r
        self.u[self.U_SHIFT][:] = 0.0
        self.u[self.U_GB][:] = 0.0

    def initial_data_ef(self, g: Grid):
        """
        Initial data for the BSSN equations.  This is a simple
        Schwarzschild solution in Eddington-Finkelstein coordinates.
        This can be used to verify the BSSN equations for LAGRANGIAN
        systems.
        """
        r = np.abs(g.xi[0])
        H = np.zeros_like(r)
        m = self.M

        H[:] = 2.0 * m / r
        self.u[self.U_ALPHA][:] = 1.0 / np.sqrt(1.0 + H)
        self.u[self.U_CHI][:] = 1.0 / (1.0 + H) ** (1.0 / 3.0)
        self.u[self.U_GRR][:] = (1.0 + H) ** (2.0 / 3.0)
        self.u[self.U_GTT][:] = r**2 / (1.0 + H) ** (1.0 / 3.0)
        self.u[self.U_ARR][:] = (
            -4.0
            * m
            * (1.0 + H) ** (1.0 / 6.0)
            * (3.0 * m + 2.0 * r)
            / (3.0 * r * r * (r + 2.0 * m))
        )
        self.u[self.U_K][:] = (
            2 * m * (r + 3.0 * m) / (r * r * (2.0 * m + r) * np.sqrt(1.0 + H))
        )
        self.u[self.U_GT][:] = (
            -2.0
            * (8 * m + 3 * r)
            * (1.0 + H) ** (1.0 / 3.0)
            / (3.0 * (2.0 * m + r) ** 2)
        )
        self.u[self.U_SHIFT][:] = H / (1 + H)
        self.u[self.U_GB][:] = 0.0

    def rhs(self, dtu, u, g: Grid):
        v = self.gbssn_system.eqs
        lambda_lapse = self.gbssn_system.lapse
        lambda_shift = self.gbssn_system.shift
        r = g.xi[0]

        chi = u[self.U_CHI]
        g_rr = u[self.U_GRR]
        g_tt = u[self.U_GTT]
        A_rr = u[self.U_ARR]
        K = u[self.U_K]
        Gamma_r = u[self.U_GT]
        alpha = u[self.U_ALPHA]
        beta_r = u[self.U_SHIFT]
        B_r = u[self.U_GB]

        chi_rhs = dtu[self.U_CHI]
        g_rr_rhs = dtu[self.U_GRR]
        g_tt_rhs = dtu[self.U_GTT]
        A_rr_rhs = dtu[self.U_ARR]
        K_rhs = dtu[self.U_K]
        Gamma_r_rhs = dtu[self.U_GT]
        alpha_rhs = dtu[self.U_ALPHA]
        beta_r_rhs = dtu[self.U_SHIFT]
        B_r_rhs = dtu[self.U_GB]

        if g.D1 is None or g.D2 is None:
            raise AttributeError(
                "Grid object 'g' must have non-None D1 and D2 attributes with 'grad' and 'grad2' methods."
            )

        filderivs = None
        if g.num_filters > 0:
            for fx in g.Filter:
                if fx.apply_filter == FilterApply.APPLY_DERIVS:
                    filderivs = fx

        if filderivs is not None:
            f_alpha = filderivs.filter(alpha)
            f_beta_r = filderivs.filter(beta_r)
            f_chi = filderivs.filter(chi)
            f_g_rr = filderivs.filter(g_rr)
            f_g_tt = filderivs.filter(g_tt)
            f_K = filderivs.filter(K)
            f_Gamma_r = filderivs.filter(Gamma_r)
        else:
            f_alpha = alpha
            f_beta_r = beta_r
            f_chi = chi
            f_g_rr = g_rr
            f_g_tt = g_tt
            f_K = K
            f_Gamma_r = Gamma_r

        d_alpha = g.D1.grad(f_alpha)
        d_beta_r = g.D1.grad(f_beta_r)
        d_chi = g.D1.grad(f_chi)
        d_g_rr = g.D1.grad(f_g_rr)
        d_g_tt = g.D1.grad(f_g_tt)
        d_K = g.D1.grad(f_K)
        d_Gamma_r = g.D1.grad(f_Gamma_r)

        USE_ADVECTION = True
        if USE_ADVECTION and g.D1.HAVE_ADVECTIVE_DERIV:
            # Advection terms
            dr = g.dx[0]
            ad_alpha = np.empty_like(alpha)
            ad_beta_r = np.empty_like(beta_r)
            ad_B_r = np.empty_like(B_r)
            ad_chi = np.empty_like(chi)
            ad_g_rr = np.empty_like(g_rr)
            ad_g_tt = np.empty_like(g_tt)
            ad_K = np.empty_like(K)
            ad_A_rr = np.empty_like(A_rr)
            ad_Gamma_r = np.empty_like(Gamma_r)

            g.D1.advec_grad(ad_B_r, B_r, beta_r, dr)
            g.D1.advec_grad(ad_alpha, alpha, beta_r, dr)
            g.D1.advec_grad(ad_beta_r, beta_r, beta_r, dr)
            g.D1.advec_grad(ad_g_rr, g_rr, beta_r, dr)
            g.D1.advec_grad(ad_g_tt, g_tt, beta_r, dr)
            g.D1.advec_grad(ad_A_rr, A_rr, beta_r, dr)
            g.D1.advec_grad(ad_K, K, beta_r, dr)
            g.D1.advec_grad(ad_Gamma_r, Gamma_r, beta_r, dr)
            g.D1.advec_grad(ad_chi, chi, beta_r, dr)
        else:
            ad_alpha = d_alpha
            ad_beta_r = d_beta_r
            ad_B_r = g.D1.grad(B_r)
            ad_chi = d_chi
            ad_g_rr = d_g_rr
            ad_g_tt = d_g_tt
            ad_K = d_K
            ad_A_rr = g.D1.grad(A_rr)
            ad_Gamma_r = d_Gamma_r

        if self.have_d2:
            # Use native second derivatives
            d2_chi = g.D2.grad(f_chi)
            d2_g_rr = g.D2.grad(f_g_rr)
            d2_g_tt = g.D2.grad(f_g_tt)
            d2_alpha = g.D2.grad(f_alpha)
            d2_beta_r = g.D2.grad(f_beta_r)
        else:
            # Use first derivatives to approximate second derivatives
            d2_chi = g.D1.grad(d_chi)
            d2_g_rr = g.D1.grad(d_g_rr)
            d2_g_tt = g.D1.grad(d_g_tt)
            d2_alpha = g.D1.grad(d_alpha)
            d2_beta_r = g.D1.grad(d_beta_r)

        BSSN.rhs_chi(
            chi_rhs,
            alpha,
            K,
            chi,
            g_rr,
            g_tt,
            beta_r,
            d_beta_r,
            ad_chi,
            d_g_rr,
            d_g_tt,
            v,
        )
        BSSN.rhs_g_rr(
            g_rr_rhs, alpha, A_rr, g_rr, g_tt, beta_r, ad_g_rr, d_g_tt, d_beta_r, v
        )
        BSSN.rhs_g_tt(
            g_tt_rhs, alpha, A_rr, g_rr, g_tt, beta_r, d_g_rr, ad_g_tt, d_beta_r, v
        )
        BSSN.rhs_A_rr(
            A_rr_rhs,
            alpha,
            A_rr,
            K,
            chi,
            g_rr,
            g_tt,
            beta_r,
            d_alpha,
            d2_alpha,
            d_beta_r,
            d_chi,
            d2_chi,
            d_g_rr,
            d2_g_rr,
            d_g_tt,
            d2_g_tt,
            ad_A_rr,
            d_Gamma_r,
            v,
        )
        BSSN.rhs_K(
            K_rhs,
            alpha,
            A_rr,
            K,
            chi,
            g_rr,
            g_tt,
            beta_r,
            ad_K,
            d_alpha,
            d_chi,
            d_g_rr,
            d_g_tt,
            d2_alpha,
        )
        BSSN.rhs_Gamma_r(
            Gamma_r_rhs,
            alpha,
            A_rr,
            chi,
            g_rr,
            g_tt,
            beta_r,
            d_K,
            d_alpha,
            d_chi,
            d_g_rr,
            d2_g_rr,
            d_g_tt,
            d2_g_tt,
            d_beta_r,
            d2_beta_r,
            ad_Gamma_r,
            v,
        )
        BSSN.rhs_gauge(
            alpha_rhs,
            beta_r_rhs,
            B_r_rhs,
            alpha,
            beta_r,
            B_r,
            K,
            ad_alpha,
            ad_beta_r,
            ad_B_r,
            d_Gamma_r,
            Gamma_r_rhs,
            self.eta,
            lambda_lapse,
            lambda_shift,
        )

        # Apply outer boundary conditions at r = rmax
        if (self.outer_bc == BSSNOuterBCType.SOMMERFELD):
            ng = g.nghost
            dxu = [
                d_chi,
                d_g_rr,
                d_g_tt,
                ad_A_rr,
                d_K,
                d_Gamma_r,
                d_alpha,
                d_beta_r,
                ad_B_r,
            ]
            for m in range(len(u)):
                BSSN.bc_sommerfeld(
                    dtu[m],
                    u[m],
                    dxu[m],
                    r,
                    self.u_falloff[m],
                    self.u_inf[m],
                    self.u_wavespeed[m],
                    ng,
                    self.extended_domain,
                )
        elif (self.outer_bc  == BSSNOuterBCType.SPONGE):
            width = 20.0
            rmax = r[-1]
            gamma_0 = 1.0
            sigma = width / 3.0
            #rstart = rmax - width
            #idx_right = np.searchsorted(r, rstart)

            # smooth cosine sponge
            # gamma = gamma_0 * np.sin(0.5*np.pi*xnorm)**2
            gamma = gamma_0*np.exp(-((r - rmax)/sigma)**2)

            imid = int(len(r)/2)
            for m in range(len(u)):
                dtu[m][imid:] -=  gamma[imid:]*(u[m][imid:] - self.u_reference[m])

            if self.extended_domain:
                rmin = r[0]
                gamma[0:imid] += gamma_0*np.exp(-((r[0:imid] - rmin)/sigma)**2)
                for m in range(len(u)):
                    dtu[m][:imid] -=  gamma[:imid]*(u[m][:imid] - self.u_parity[m]*self.u_reference[m])

        else:
            raise ValueError("BSSN outer boundary condition not initialized")


        DEBUG3 = False
        if DEBUG3:
            print("v = ", v)
            print("dt_alpha = ", l2norm(alpha_rhs[10:]))
            print("dt_beta_r = ", l2norm(beta_r_rhs[10:]))
            print("dt_chi = ", l2norm(chi_rhs[10:]))
            print("dt_g_rr = ", l2norm(g_rr_rhs[10:]))
            print("dt_g_tt = ", l2norm(g_tt_rhs[10:]))
            print("dt_A_rr = ", l2norm(A_rr_rhs[10:]))
            print("dt_K = ", l2norm(K_rhs[10:]))
            print("dt_Gamma_r = ", l2norm(Gamma_r_rhs[10:]))
            sys.exit(0)

        DEBUG_WRITE_RHS = False
        DEBUG_WRITE_DERIVS = False
        if DEBUG_WRITE_RHS:
            """write RHS to CSV file for debugging"""
            with open("bssn_rhs.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "Index",
                        "r",
                        "chi_rhs",
                        "g_rr_rhs",
                        "g_tt_rhs",
                        "A_rr_rhs",
                        "K_rhs",
                        "Gamma_r_rhs",
                        "alpha_rhs",
                        "beta_r_rhs",
                        "B_r_rhs",
                    ]
                )
                for i in range(g.shp[0]):
                    writer.writerow(
                        [
                            i,
                            r[i],
                            chi_rhs[i],
                            g_rr_rhs[i],
                            g_tt_rhs[i],
                            A_rr_rhs[i],
                            K_rhs[i],
                            Gamma_r_rhs[i],
                            alpha_rhs[i],
                            beta_r_rhs[i],
                            B_r_rhs[i],
                        ]
                    )

        if DEBUG_WRITE_DERIVS:
            """write derivatives to CSV file for debugging"""
            with open("bssn_derivs.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "Index",
                        "r",
                        "d_alpha",
                        "d_beta_r",
                        "d_chi",
                        "d_g_rr",
                        "d_g_tt",
                        "d_K",
                        "d_Gamma_r",
                    ]
                )
                for i in range(g.shp[0]):
                    writer.writerow(
                        [
                            i,
                            r[i],
                            d_alpha[i],
                            d_beta_r[i],
                            d_chi[i],
                            d_g_rr[i],
                            d_g_tt[i],
                            d_K[i],
                            d_Gamma_r[i],
                        ]
                    )

        if DEBUG_WRITE_DERIVS or DEBUG_WRITE_RHS:
            sys.exit(-1)

    @staticmethod
    @njit
    def rhs_A_rr(
        A_rr_rhs,
        alpha,
        A_rr,
        K,
        chi,
        g_rr,
        g_tt,
        beta_r,
        d_alpha,
        d2_alpha,
        d_beta_r,
        d_chi,
        d2_chi,
        d_g_rr,
        d2_g_rr,
        d_g_tt,
        d2_g_tt,
        ad_A_rr,
        d_Gamma_r,
        v,
    ):
        inv_g_rr = 1.0 / g_rr
        inv_g_tt = 1.0 / g_tt
        inv_g_rr2 = inv_g_rr * inv_g_rr
        inv_g_tt2 = inv_g_tt * inv_g_tt
        inv_6chi = 1.0 / (6.0 * chi)

        for i in range(A_rr_rhs.shape[0]):
            A_rr_rhs[i] = (
                -2.0 * alpha[i] * A_rr[i] ** 2 * inv_g_rr[i]
                + K[i] * alpha[i] * A_rr[i]
                - v * beta_r[i] * d_g_rr[i] * A_rr[i] * inv_g_rr[i] / 3.0
                - 2.0 * v * beta_r[i] * d_g_tt[i] * A_rr[i] * inv_g_tt[i] / 3.0
                - (2.0 / 3.0) * v * d_beta_r[i] * A_rr[i]
                + 2.0 * d_beta_r[i] * A_rr[i]
                + 2.0 * alpha[i] * chi[i] * d_g_rr[i] ** 2 * inv_g_rr2[i] / 3.0
                - alpha[i] * chi[i] * d_g_tt[i] ** 2 * inv_g_tt2[i] / 3.0
                - alpha[i] * d_chi[i] ** 2 * inv_6chi[i]
                - (2.0 / 3.0) * g_rr[i] * alpha[i] * chi[i] * inv_g_tt[i]
                + beta_r[i] * ad_A_rr[i]
                + (2.0 / 3.0) * g_rr[i] * alpha[i] * chi[i] * d_Gamma_r[i]
                - (alpha[i] * chi[i] * d_g_rr[i] * d_g_tt[i])
                * inv_g_rr[i]
                * inv_g_tt[i]
                / 2.0
                + chi[i] * d_g_rr[i] * d_alpha[i] * inv_g_rr[i] / 3.0
                + chi[i] * d_g_tt[i] * d_alpha[i] * inv_g_tt[i] / 3.0
                - (alpha[i] * d_g_rr[i] * d_chi[i]) * inv_g_rr[i] / 6.0
                - (alpha[i] * d_g_tt[i] * d_chi[i]) * inv_g_tt[i] / 6.0
                - (2.0 / 3.0) * d_chi[i] * d_alpha[i]
                - alpha[i] * chi[i] * d2_g_rr[i] * inv_g_rr[i] / 3.0
                + alpha[i] * chi[i] * d2_g_tt[i] * inv_g_tt[i] / 3.0
                - (2.0 / 3.0) * chi[i] * d2_alpha[i]
                + alpha[i] * d2_chi[i] / 3.0
            )

    @staticmethod
    @njit
    def rhs_Gamma_r(
        Gamma_r_rhs,
        alpha,
        A_rr,
        chi,
        g_rr,
        g_tt,
        beta_r,
        d_K,
        d_alpha,
        d_chi,
        d_g_rr,
        d2_g_rr,
        d_g_tt,
        d2_g_tt,
        d_beta_r,
        d2_beta_r,
        ad_Gamma_r,
        v,
    ):

        inv_g_rr = 1.0 / g_rr
        inv_g_tt = 1.0 / g_tt
        inv_g_rr2 = inv_g_rr * inv_g_rr
        inv_g_tt2 = inv_g_tt * inv_g_tt

        Gamma_r_rhs[:] = (
            -v * beta_r * d_g_tt**2 * inv_g_rr * inv_g_tt2
            + A_rr * alpha * d_g_tt * inv_g_rr2 * inv_g_tt
            - v * d_beta_r * d_g_tt * inv_g_rr * inv_g_tt / 3.0
            + d_beta_r * d_g_tt * inv_g_rr * inv_g_tt
            + beta_r * ad_Gamma_r
            + A_rr * alpha * d_g_rr * inv_g_rr2 * inv_g_rr
            - (4.0 / 3.0) * alpha * d_K * inv_g_rr
            - 2.0 * A_rr * d_alpha * inv_g_rr2
            + v * d_g_rr * d_beta_r * inv_g_rr2 / 2.0
            - d_g_rr * d_beta_r * inv_g_rr2 / 2.0
            - 3.0 * A_rr * alpha * d_chi * inv_g_rr2 / chi
            + v * beta_r * d2_g_rr * inv_g_rr2 / 6.0
            + v * beta_r * d2_g_tt * inv_g_rr * inv_g_tt / 3.0
            + v * d2_beta_r * inv_g_rr / 3.0
            + d2_beta_r * inv_g_rr
        )

    @staticmethod
    @njit
    def rhs_K(
        K_rhs,
        alpha,
        A_rr,
        K,
        chi,
        g_rr,
        g_tt,
        beta_r,
        ad_K,
        d_alpha,
        d_chi,
        d_g_rr,
        d_g_tt,
        d2_alpha,
    ):

        inv_g_rr = 1.0 / g_rr
        inv_g_tt = 1.0 / g_tt
        inv_g_rr2 = inv_g_rr * inv_g_rr

        K_rhs[:] = (
            3 * alpha * A_rr**2 * inv_g_rr2 / 2
            + K**2 * alpha / 3
            + beta_r * ad_K
            + chi * d_g_rr * d_alpha * inv_g_rr2 / 2
            - chi * d_g_tt * d_alpha * inv_g_rr * inv_g_tt
            + d_alpha * d_chi * inv_g_rr / 2
            - chi * d2_alpha * inv_g_rr
        )
        """
        print("chi       = ", l2norm(chi))
        print("alpha     = ", l2norm(alpha))
        print("d2_alpha  = ", l2norm(d2_alpha))
        print("K_rhs     = ", l2norm(K_rhs))
        """

    @staticmethod
    @njit
    def rhs_g_rr(
        grr_rhs, alpha, A_rr, g_rr, g_tt, beta_r, ad_g_rr, d_g_tt, d_beta_r, v
    ):

        inv_g_tt = 1.0 / g_tt
        grr_rhs[:] = (
            -2 * A_rr * alpha
            - (1 / 3) * v * beta_r * ad_g_rr
            + beta_r * ad_g_rr
            - 2 * g_rr * v * beta_r * d_g_tt * inv_g_tt / 3
            + 2 * g_rr * d_beta_r
            - 2 / 3 * g_rr * v * d_beta_r
        )

    @staticmethod
    @njit
    def rhs_g_tt(
        gtt_rhs, alpha, A_rr, g_rr, g_tt, beta_r, d_g_rr, ad_g_tt, d_beta_r, v
    ):

        inv_g_rr = 1.0 / g_rr
        gtt_rhs[:] = (
            A_rr * g_tt * alpha * inv_g_rr
            - g_tt * v * beta_r * d_g_rr * inv_g_rr / 3
            - 2 / 3 * v * beta_r * ad_g_tt
            + beta_r * ad_g_tt
            - 2 / 3 * g_tt * v * d_beta_r
        )

    @staticmethod
    @njit
    def rhs_chi(
        chi_rhs, alpha, K, chi, g_rr, g_tt, beta_r, d_beta_r, ad_chi, d_g_rr, d_g_tt, v
    ):

        chi_rhs[:] = (
            2 * K[:] * alpha[:] * chi[:] / 3
            - v * beta_r[:] * d_g_rr[:] * chi[:] / (3 * g_rr[:])
            - 2 * v * beta_r[:] * d_g_tt[:] * chi[:] / (3 * g_tt[:])
            - 2 / 3 * v * d_beta_r[:] * chi[:]
            + beta_r[:] * ad_chi[:]
        )
        """
        print("v       = ", v)
        print("chi     = ", l2norm(chi))
        print("alpha   = ", l2norm(alpha))
        print("d_g_tt  = ", l2norm(d_g_tt))
        print("chi_rhs = ", l2norm(chi_rhs))
        """

    @staticmethod
    @njit
    def rhs_gauge(
        alpha_rhs,
        beta_r_rhs,
        B_r_rhs,
        alpha,
        beta_r,
        B_r,
        K,
        ad_alpha,
        ad_beta_r,
        ad_B_r,
        d_Gamma_r,
        dt_Gamma_r,
        eta,
        lambda_lapse,
        lambda_shift,
    ):

        alpha_rhs[:] = lambda_lapse * beta_r * ad_alpha - 2 * alpha * K
        beta_r_rhs[:] = (3.0 / 4.0) * B_r + lambda_shift * beta_r * ad_beta_r
        B_r_rhs[:] = (
            dt_Gamma_r
            + lambda_shift * beta_r * ad_B_r
            - lambda_shift * beta_r * d_Gamma_r
            - eta * B_r
        )

    def apply_bcs(self, u, g: Grid):
        """
        Regularity conditions at the origin.  (This routine called from RK4.)
        """
        """       
        Apply regularity conditions at the origin.  Brown uses guard
        cells to apply the regularity conditions to some variables.
        These conditions are used to set some variables or their
        derivatives to zero at the origin.

        gtt = u[self.U_GTT]
        b = u[self.U_SHIFT]
        gtt[0] = (-315*gtt[1] + 210*gtt[2] - 126*gtt[3] + 45*gtt[4] - 7*gtt[5])/63
        b[0] = (-315*b[1] + 210*b[2] - 126*b[3] + 45*b[4] - 7*b[5])/63
        # print("Applying regularity conditions at the origin")

        # Apply regularity conditions at the origin
        if self.gbssn_system.eqs == 0:
            # Lagrangian
            self.u[self.U_GTT][0] = extrapolate_func(self.u[self.U_GTT], order=5)
            self.u[self.U_SHIFT][0] = extrapolate_func(self.u[self.U_SHIFT], order=5)
        elif self.gbssn_system.eqs == 1:
            # Eulerian
            self.u[self.U_SHIFT][0] = extrapolate_func(self.u[self.U_SHIFT], order=5)
        """
        # print("Applying regularity conditions at the origin")
        ng = g.nghost

        if self.extended_domain == False:
            set_inner_regularity(self.u[self.U_CHI], ng, 1)
            set_inner_regularity(self.u[self.U_GRR], ng, 1)
            set_inner_regularity(self.u[self.U_GTT], ng, 1)
            set_inner_regularity(self.u[self.U_ARR], ng, 1)
            set_inner_regularity(self.u[self.U_K], ng, 1)
            set_inner_regularity(self.u[self.U_GT], ng, -1)
            set_inner_regularity(self.u[self.U_ALPHA], ng, 1)
            set_inner_regularity(self.u[self.U_SHIFT], ng, -1)
            set_inner_regularity(self.u[self.U_GB], ng, -1)

    def cal_constraints(self, u, g: Grid):
        """
        Calculate the constraints for the BSSN equations.
        """
        Ham = self.C[self.C_HAM]
        Mom = self.C[self.MOM]
        Gamcon = self.C[self.GAMCON]
        nghost = g.get_nghost()

        chi = u[self.U_CHI]
        g_rr = u[self.U_GRR]
        g_tt = u[self.U_GTT]
        A_rr = u[self.U_ARR]
        K = u[self.U_K]
        Gamma_r = u[self.U_GT]

        if g.D1 is None or g.D2 is None:
            raise AttributeError(
                "Grid object 'g' must have non-None D1 and D2 attributes with 'grad' and 'grad2' methods."
            )

        d_chi = g.D1.grad(chi)
        d_g_rr = g.D1.grad(g_rr)
        d_g_tt = g.D1.grad(g_tt)
        d_A_rr = g.D1.grad(A_rr)
        d_K = g.D1.grad(K)

        d2_chi = g.D2.grad(chi)
        d2_g_tt = g.D2.grad(g_tt)

        r = g.xi[0]

        DEBUG_DERIVS = False
        if DEBUG_DERIVS:
            r = g.xi[0]
            dr = r[1] - r[0]
            ED1 = ExplicitFirst642_1D(dr)
            ED2 = ExplicitSecond642_1D(dr)
            ed_chi = ED1.grad(chi)
            ed_g_rr = ED1.grad(g_rr)
            ed_g_tt = ED1.grad(g_tt)
            ed_A_rr = ED1.grad(A_rr)
            ed_K = ED1.grad(K)
            ed2_chi = ED2.grad(chi)
            ed2_g_tt = ED2.grad(g_tt)
            print(f"...D1(chi):  {l2norm(d_chi - ed_chi):.2e}")
            print(f"...D1(g_rr): {l2norm(d_g_rr - ed_g_rr):.2e}")
            print(f"...D1(g_tt): {l2norm(d_g_tt - ed_g_tt):.2e}")
            print(f"...D1(A_rr): {l2norm(d_A_rr - ed_A_rr):.2e}")
            print(f"...D2(chi): {l2norm(d2_chi - ed2_chi):.2e}")
            print(f"...D2(g_tt): {l2norm(d2_g_tt - ed2_g_tt):.2e}")

        BSSN.get_constraints(
            Ham,
            Mom,
            Gamcon,
            chi,
            g_rr,
            g_tt,
            A_rr,
            K,
            Gamma_r,
            d_chi,
            d_g_rr,
            d_g_tt,
            d_A_rr,
            d_K,
            d2_chi,
            d2_g_tt,
            r,
            nghost,
        )

    @staticmethod
    def bc_sommerfeld(dtu, u, dxu, r, n_falloff, u_inf, v, ng, extended_domain):
        """
        Radiative (Sommerfeld) outer boundary condition with per-variable
        characteristic (wave) speed v:

            dtu = -v [ sign(r) dxu + n (u - u_inf)/abs(r) ]

        The radial derivative changes direction at the negative boundary, but
        the falloff term uses the positive radius at both boundaries.

        The per-variable falloff power n and asymptotic value u_inf encode each
        field's radial structure (e.g. gtt ~ r^2 via n<0). The supplied speed
        v is retained at both boundaries.
        """
        if n_falloff < 0:
            u_inf = r[-1] ** (abs(n_falloff))
        dtu[-ng:] = -v * (dxu[-ng:] + n_falloff * (u[-ng:] - u_inf) / np.abs(r[-ng:]))
        if extended_domain:
            dtu[:ng] = v * (dxu[:ng] - n_falloff * (u[:ng] - u_inf) / np.abs(r[:ng]))

    @staticmethod
    @njit
    def bc_sponge(dtu, u, gamma, u_ref, idx_right, idx_left):
        n = u.shape
        # right
        for i in range(idx_right, n):
            dtu[i] -= gamma[i] * (u[i] - u_ref)
        # left
        if idx_left > 0:
            for i in range(0, idx_left):
                dtu[i] -= gamma[i] * (u[i] - u_ref)
        
        
    @staticmethod
    @njit
    def get_constraints(
        Ham,
        Mom,
        Gamcon,
        chi,
        g_rr,
        g_tt,
        A_rr,
        K,
        Gamma_r,
        d_chi,
        d_g_rr,
        d_g_tt,
        d_A_rr,
        d_K,
        d2_chi,
        d2_g_tt,
        r,
        nghost,
    ):

        inv_g_rr = 1.0 / g_rr
        inv_g_tt = 1.0 / g_tt
        inv_g_rr2 = inv_g_rr * inv_g_rr

        Ham[:] = (
            -(3.0 / 2.0) * A_rr**2 * inv_g_rr2
            + (2.0 / 3.0) * K**2
            - (5.0 / 2.0) * d_chi**2 * inv_g_rr / chi
            + 2.0 * d2_chi * inv_g_rr
            + 2.0 * chi * inv_g_tt
            - 2.0 * chi * d2_g_tt * inv_g_rr * inv_g_tt
            + 2.0 * d_chi * d_g_tt * inv_g_rr * inv_g_tt
            + chi * d_g_rr * d_g_tt * inv_g_rr2 * inv_g_tt
            - d_chi * d_g_rr * inv_g_rr2
            + 0.5 * chi * d_g_tt**2 * inv_g_rr * inv_g_tt * inv_g_tt
        )

        Mom[:] = (
            d_A_rr * inv_g_rr
            - (2.0 / 3.0) * d_K
            - (3.0 / 2.0) * A_rr * d_chi * inv_g_rr / chi
            + (3.0 / 2.0) * A_rr * d_g_tt * inv_g_rr * inv_g_tt
            - A_rr * d_g_rr * inv_g_rr2
        )

        Gamcon[:] = -0.5 * d_g_rr * inv_g_rr2 + Gamma_r + d_g_tt * inv_g_rr * inv_g_tt

        for i in range(len(r)):
            if np.abs(r[i]) < 0.5:
                Ham[i] = 0.0
                Mom[i] = 0.0
                Gamcon[i] = 0.0

        Ham[:nghost] = 0.0
        Ham[-nghost:] = 0.0
        Mom[:nghost] = 0.0
        Mom[-nghost:] = 0.0
        Gamcon[:nghost] = 0.0
        Gamcon[-nghost:] = 0.0

    def find_horizon(self, g: Grid, r0):
        if r0 == None:
            r0 = 1.0

        r = g.xi[0]
        N = len(r)
        chi = self.u[self.U_CHI]
        g_rr = self.u[self.U_GRR]
        g_tt = self.u[self.U_GTT]
        A_rr = self.u[self.U_ARR]
        K = self.u[self.U_K]

        d_chi = g.D1.grad(chi)
        d_g_tt = g.D1.grad(g_tt)

        # Calculate the null-ray expansion.
        rTheta = np.zeros_like(chi)
        for i in range(N):
            A_tt = -A_rr[i] * g_tt[i] / (2.0 * g_rr[i])
            d_phi = -d_chi[i] / (4.0 * chi[i])
            e2p = 1.0 / np.sqrt(chi[i])
            Kbar_tt = A_tt + g_tt[i] * K[i] / 3.0
            rTheta[i] = np.abs(r[i]) * (
                (4.0 * g_tt[i] * d_phi + d_g_tt[i]) / (e2p * g_tt[i] * np.sqrt(g_rr[i]))
                - 2.0 * Kbar_tt / g_tt[i]
            )

        # Find location of the horizon--march from the outside.
        rin = 0.25 * r0
        rout = 4.0 * r0
        rh = find_ah_zero(rTheta, r, rin, rout)
        if rh != None:
            gtt_h = linear_interpolation(g_tt, r, rh)
            chi_h = linear_interpolation(chi, r, rh)
            area_h = 4.0 * np.pi * gtt_h / chi_h
            mass_h = np.sqrt(area_h / (16.0 * np.pi))
        else:
            mass_h = None

        return rh, mass_h, rTheta


@njit
def find_ah_zero(f: np.ndarray, x: np.ndarray, x1: float, x2: float):
    """
    Find zero of the function f, moving from the outer part of the grid.
    This is written for finding the apparent horizon from the null-ray expansion.

    Parameters:
        f = function to find zero
        x =  coordinate array
        x1 = lower bound of root
        x2 = upper bound of root

    Note:  The routine returns "None" if:
       (1) the bounds are not on the grid,
       (2) no root of f is found
    """

    xmin = x[0]
    xmax = x[-1]
    if not (xmin <= x1 <= xmax) or not (xmin <= x2 <= xmax):
        print(f"...find_ah_zero: x1={x1} and x2={x2} are out of range.")
        return None

    dx = x[1] - x[0]
    k1 = int((x1 - xmin) / dx)
    k2 = int((x2 - xmin) / dx)
    if abs(x1) > abs(x2):
        kout, kin = k1, k2
        xout, xin = x1, x2
    else:
        kout, kin = k2, k1
        xout, xin = x2, x1
    if xout > xin:
        for k in range(kout, kin, -1):
            if f[k] * f[k - 1] < 0:
                # Linear interpolation for zero crossing
                xzero = x[k] - f[k] * (x[k] - x[k - 1]) / (f[k] - f[k - 1])
                return xzero
    else:
        for k in range(kout, kin):
            if f[k] * f[k + 1] < 0:
                # Linear interpolation for zero crossing
                xzero = x[k] - f[k] * (x[k] - x[k + 1]) / (f[k] - f[k + 1])
                return xzero
    return None


def extrapolate_func(u, order=4):
    """
    To apply regularity at the origin. This function assumes that
    the grid is defined at points r = (i*h/2) for i = 0, 1, ..., N.

    The interpolating polynomial of order n is
        P_n(r) = a_0 + a_1*r + a_2*r^2 + ... + a_n*r^n

    The function at h/2 is set such that the extrapolated value at
    r = 0 is zero.
    """
    if order == 4:
        return (420 * u[1] - 378 * u[2] + 180 * u[3] - 35 * u[4]) / 315
    elif order == 5:
        return (1155 * u[1] - 1386 * u[2] + 990 * u[3] - 385 * u[4] + 63 * u[5]) / 693
    elif order == 6:
        return (
            6006 * u[1]
            - 9009 * u[2]
            + 8580 * u[3]
            - 5005 * u[4]
            + 1638 * u[5]
            - 231 * u[6]
        ) / 3003
    else:
        raise ValueError("Invalid order. Must be 4, 5 or 6.")


def extrapolate_deriv(u, order=4):
    """
    To apply regularity at the origin. This function assumes that
    the grid is defined at points r = (i*h/2) for i = 0, 1, ..., N.

    The interpolating polynomial of order n is
        P_n(r) = a_0 + a_1*r + a_2*r^2 + ... + a_n*r^n

    The function at h/2 is set such that the extrapolated value of
    dP_n/dr at r = 0 is zero.
    """
    if order == 4:
        return (229 * u[1] - 225 * u[2] + 111 * u[3] - 22 * u[4]) / 93
    elif order == 5:
        return (
            26765 * u[1] - 34890 * u[2] + 25770 * u[3] - 10205 * u[4] + 1689 * u[5]
        ) / 9129
    elif order == 6:
        return (
            36527 * u[1]
            - 59295 * u[2]
            + 58310 * u[3]
            - 34610 * u[4]
            + 11451 * u[5]
            - 1627 * u[6]
        ) / 10756
    else:
        raise ValueError("Invalid order. Must be 4, 5 or 6.")


def set_inner_regularity(u, ng, parity=1):
    """
    Apply inner boundary condition at the origin.
    """
    u[:ng] = parity * u[2 * ng - 1 : ng - 1 : -1]
