import numpy as np
import sys
import os
import csv
from enum import Enum
from numba import njit

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nwave import *
import gauges
import sommerfeld_tail as tail_bc

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
        lapse_gauge=None,
        sommerfeld_tail=True,
        enforce_det=True,
    ):
        """
        Initialize the BSSN equations.
        Parameters
        ----------
        g : Grid1D
            Grid object.
        M : float
            Mass of the black hole.
        eta : float or gauges.ShiftDamping
            Gamma-driver shift damping: a constant, or a damping object from
            gauges.SHIFT_DAMPING.
        apply_bc : "FUNCTION" or None
            If "FUNCTION", apply the boundary conditions using the function in each post-RK step.
        gbssn_system : GBSSNSystem
            System of equations to use.
        outer_bc : outer boundary condition type
        sommerfeld_tail : bool
            Add the default-on one-anchor R^-3 nonradiative residual correction.
            False preserves the ordinary Sommerfeld boundary condition.
        enforce_det : bool
            Rescale g_rr and g_tt by det^(-1/3) after every RK stage so that
            det(gtilde) = g_rr (g_tt / r^2)^2 = 1 (the flat value).  KO
            dissipation applied to g_rr and g_tt separately drives det away
            from 1 near the puncture; without this the first cells collapse
            (~78M at h = .05).  Default on; False restores the unprojected
            evolution.
        lapse_gauge : gauges.LapseGauge
            Lapse gauge from gauges.LAPSE_GAUGES; default 1+log with k = 2.
            A gauge with n_aux = 1 (the telegrapher) adds a 10th evolved field.
        """

        if not isinstance(sommerfeld_tail, (bool, np.bool_)):
            raise ValueError("sommerfeld_tail must be a boolean")
        self.sommerfeld_tail = sommerfeld_tail
        if not isinstance(enforce_det, (bool, np.bool_)):
            raise ValueError("enforce_det must be a boolean")
        self.enforce_det = bool(enforce_det)
        self._tail_geometry = (
            tail_bc.geometry(g.xi[0], g.nghost, extended_domain)
            if sommerfeld_tail and outer_bc == BSSNOuterBCType.SOMMERFELD else None
        )

        lapse_gauge = gauges.OnePlusLog() if lapse_gauge is None else lapse_gauge
        NU = 9 + lapse_gauge.n_aux
        super().__init__(NU, g, apply_bc)

        self.gbssn_system = gbssn_system
        self.lapse_gauge = lapse_gauge
        self.shift_damping = (
            eta if isinstance(eta, gauges.ShiftDamping) else gauges.ConstantEta(eta)
        )
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
        self.U_LAPSE_AUX = 9  # A = d_t alpha; only when lapse_gauge.n_aux = 1

        nx = g.shp[0]
        self.Nc = 3
        self.C = []
        for i in range(self.Nc):
            self.C.append(np.zeros(nx))

        self.C_HAM = 0
        self.MOM = 1
        self.GAMCON = 2
        self.u_names = ["chi", "grr", "gtt", "Arr", "K", "Gt", "alpha", "beta", "gB"]
        self.u_names += self.lapse_gauge.aux_names
        self.c_names = ["ham", "mom", "gamcom"]
        self.outer_bc = outer_bc
        # gtt's falloff/asymptote entries are unused: rhs() applies
        # bc_spherical_gtt to it instead.
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
        self.u_falloff += [2] * self.lapse_gauge.n_aux  # A = d_t alpha falls off like K
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
        self.u_inf += [0] * self.lapse_gauge.n_aux
        # Outer-boundary characteristic (wave) speeds per field, from Brown's
        # GBSSN characteristic analysis (papers/BrownSpherical/paper2.tex:370-404).
        # Proper speeds: physical/gravitational modes = 1; the Bona-Masso lapse
        # mode = alpha sqrt(f(alpha)) (sqrt(2/alpha) for 1+log); shift (Lagrangian
        # Gamma-driver, case L^+_+) = sqrt(1/(alpha^2 chi)). The radiative BC is
        # only valid as r->inf, where alpha->1, chi->1, g_rr->1, so all reduce to
        # v=1 for the geometric fields and v=sqrt(f(1)) for the lapse, which the
        # lapse gauge supplies (sqrt(2) for 1+log with k=2).
        self.u_wavespeed = np.ones(len(self.u_names))
        self.u_wavespeed[self.U_ALPHA] = self.lapse_gauge.speed_inf()
        if self.lapse_gauge.n_aux:
            self.u_wavespeed[self.U_LAPSE_AUX] = self.lapse_gauge.speed_inf()
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
            self.initial_data_ef(g, params.get("ef_stuff_radius", 1.8))
        else:
            raise ValueError(
                "Invalid initial data. Must be 'puncture' or 'EddingtonFinkelstein'."
            )
        if self.enforce_det:
            self.project_initial_det(g)
        if self.lapse_gauge.n_aux:
            self.u[self.U_LAPSE_AUX][:] = self.lapse_gauge.initial_aux(
                self.u[self.U_ALPHA], self.u[self.U_K]
            )

    # Initial data whose det(gtilde) differs from 1 by more than this are
    # re-split onto det = 1 by project_initial_det; below it the per-stage
    # projection absorbs the round-off.
    INITIAL_DET_TOL = 1.0e-12

    def project_initial_det(self, g: Grid):
        """Re-split initial data onto det(gtilde) = 1, keeping the physical data.

        Called by initialize() when enforce_det is on.  Points that already
        satisfy det = 1 to INITIAL_DET_TOL are left bitwise unchanged; that
        is every point of the puncture and Eddington-Finkelstein data on
        main.  At the other points, with f = det^(-1/3), the conformal
        split is changed but the physical metric and extrinsic curvature
        are not:

            gtilde -> f gtilde,  chi -> f chi,  Atilde_rr -> f Atilde_rr,
            Gt     -> (Gt - (ln f)' / (2 g_rr)) / f.

        The Gt update is the exact change of Gt^r = g_rr'/(2 g_rr^2)
        - g_tt'/(g_rr g_tt) under g -> f g; (ln f)' uses the grid's D1.
        The per-stage projection (enforce_unit_det) instead rescales only
        gtilde, removing the discrete constraint drift as SENR/NRPy do.
        """
        r = g.xi[0]
        g_rr = self.u[self.U_GRR]
        g_tt = self.u[self.U_GTT]
        det = g_rr * (g_tt / (r * r)) ** 2
        if not (np.all(np.isfinite(det)) and np.all(det > 0.0)):
            raise ValueError("initial data: non-positive or non-finite det(gtilde)")
        off = np.abs(det - 1.0) > self.INITIAL_DET_TOL
        if not np.any(off):
            return
        if g.D1 is None:
            raise ValueError(
                "project_initial_det needs the grid's D1 operator to update Gt"
            )
        # Only points off det = 1 change, so exact data outside a blended
        # region stay bitwise exact (the D1 stencil would otherwise spread
        # truncation error from the region's edge into them).
        ln_f = np.where(off, -np.log(det) / 3.0, 0.0)
        f = np.exp(ln_f)
        Gt = self.u[self.U_GT]
        Gt[off] = ((Gt - 0.5 * g.D1.grad(ln_f) / g_rr) / f)[off]
        for k in (self.U_GRR, self.U_GTT, self.U_CHI, self.U_ARR):
            self.u[k][off] *= f[off]
        if not self.extended_domain:
            ng = g.nghost
            for k, parity in ((self.U_CHI, 1), (self.U_GRR, 1), (self.U_GTT, 1),
                              (self.U_ARR, 1), (self.U_GT, -1)):
                set_inner_regularity(self.u[k], ng, parity)

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

    def initial_data_ef(self, g: Grid, stuff_radius=1.8):
        """
        Schwarzschild in Eddington-Finkelstein (Kerr-Schild) coordinates,
        for the Lagrangian system.

        Unlike puncture data, this slice contains the physical singularity
        at r = 0 (K ~ r^-3/2), which no grid resolves.  Following the
        turducken approach (Brown et al. 2009, arXiv:0707.3101), the data
        are blended smoothly into the same-mass isotropic puncture with a
        collapsed lapse over stuff_radius/2 < r < stuff_radius (in units of
        M, inside the horizon at r = 2M).  Gt is computed from the blended
        conformal metric, so only the Hamiltonian and momentum constraints
        are violated, and only inside the horizon.  Gt and beta are odd in
        x, as for an extended domain.
        """
        if not 0.0 < stuff_radius < 2.0:
            raise ValueError("ef_stuff_radius must lie inside the horizon, 0 < R < 2")
        x = g.xi[0]
        r = np.abs(x)
        m = self.M

        def ks(r):
            H = 2.0 * m / r
            return dict(
                chi=(1.0 + H) ** (-1.0 / 3.0),
                grr=(1.0 + H) ** (2.0 / 3.0),
                q=(1.0 + H) ** (-1.0 / 3.0),  # gtt / r^2
                Arr=-4.0 * m * (1.0 + H) ** (1.0 / 6.0) * (3.0 * m + 2.0 * r)
                / (3.0 * r * r * (r + 2.0 * m)),
                K=2.0 * m * (r + 3.0 * m) / (r * r * (2.0 * m + r) * np.sqrt(1.0 + H)),
                alpha=1.0 / np.sqrt(1.0 + H),
                beta=H / (1.0 + H),
            )

        # C-infinity step w: 0 for r <= r0, 1 for r >= r1
        r1 = stuff_radius * m
        r0 = 0.5 * r1
        z = np.clip((r - r0) / (r1 - r0), 0.0, 1.0)

        def bump(z):
            zs = np.maximum(z, 1.0e-3)
            return np.where(z > 1.0e-3, np.exp(-1.0 / zs), 0.0), 1.0 / zs**2

        a, da = bump(z)
        b, db = bump(1.0 - z)
        w = a / (a + b)
        dw = a * b * (da + db) / (a + b) ** 2 / (r1 - r0)

        # Kerr-Schild outside, puncture inside (flat conformal metric)
        rs = np.maximum(r, r0)
        f = ks(rs)
        chi_p = (1.0 + m / (2.0 * r)) ** -4
        f0 = dict(chi=chi_p, grr=1.0, q=1.0, Arr=0.0, K=0.0,
                  alpha=np.sqrt(chi_p), beta=0.0)
        u = {k: w * f[k] + (1.0 - w) * f0[k] for k in f}

        H, dH = 2.0 * m / rs, -2.0 * m / rs**2
        dgrr = w * (2.0 / 3.0) * (1.0 + H) ** (-1.0 / 3.0) * dH + dw * (f["grr"] - 1.0)
        dq = w * (-1.0 / 3.0) * (1.0 + H) ** (-4.0 / 3.0) * dH + dw * (f["q"] - 1.0)

        self.u[self.U_CHI][:] = u["chi"]
        self.u[self.U_GRR][:] = u["grr"]
        self.u[self.U_GTT][:] = r**2 * u["q"]
        self.u[self.U_ARR][:] = u["Arr"]
        self.u[self.U_K][:] = u["K"]
        self.u[self.U_ALPHA][:] = u["alpha"]
        self.u[self.U_SHIFT][:] = np.sign(x) * u["beta"]
        # Gt = grr'/(2 grr^2) - gtt'/(grr gtt) with gtt = r^2 q
        self.u[self.U_GT][:] = np.sign(x) * (
            dgrr / (2.0 * u["grr"] ** 2) - (2.0 / r + dq / u["q"]) / u["grr"]
        )
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
        # Covariant Laplacian of the lapse, D^2 alpha = gamma^{ij} D_i D_j alpha.
        # Computed once so rhs_K and any lapse gauge needing it share one copy.
        lap_alpha = BSSN.laplacian_alpha(
            chi, g_rr, g_tt, d_alpha, d_chi, d_g_rr, d_g_tt, d2_alpha
        )

        BSSN.rhs_K(
            K_rhs,
            alpha,
            A_rr,
            K,
            g_rr,
            beta_r,
            ad_K,
            lap_alpha,
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
            0.0 if self.lapse_gauge.n_aux else self.lapse_gauge.alpha2_f(alpha),
            self.shift_damping(r, chi, g_rr, d_chi, self.M),
            lambda_lapse,
            lambda_shift,
        )
        if self.lapse_gauge.eta_lapse != 0.0:
            alpha_rhs += self.lapse_gauge.eta_lapse * lap_alpha
        if self.lapse_gauge.n_aux:
            # Second-order lapse: d_t alpha = A + advection replaces the
            # Bona-Masso form, and A has its own evolution equation.
            lapse_aux = u[self.U_LAPSE_AUX]
            d_lapse_aux = g.D1.grad(lapse_aux)
            dtu[self.U_LAPSE_AUX][:] = (
                self.lapse_gauge.rhs_aux(alpha, lapse_aux, K, lap_alpha, chi)
                + lambda_lapse * beta_r * d_lapse_aux
            )
            alpha_rhs[:] = lapse_aux + lambda_lapse * beta_r * ad_alpha

        # Apply outer boundary conditions at r = rmax
        if (self.outer_bc == BSSNOuterBCType.SOMMERFELD):
            ng = g.nghost
            # Radiation needs ordinary (centred) radial derivatives, not the
            # beta-upwinded ones used for advection: with beta = 0 the upwind
            # choice picks the same one-sided stencil at both ends and breaks
            # the fields' parity.  Without upwinding or derivative filtering,
            # ad_A_rr and ad_B_r already are the centred derivatives.
            if filderivs is None and not (USE_ADVECTION and g.D1.HAVE_ADVECTIVE_DERIV):
                d_A_rr_bc, d_B_r_bc = ad_A_rr, ad_B_r
            else:
                f_A_rr_bc = filderivs.filter(A_rr) if filderivs is not None else A_rr
                f_B_r_bc = filderivs.filter(B_r) if filderivs is not None else B_r
                d_A_rr_bc, d_B_r_bc = g.D1.grad(f_A_rr_bc), g.D1.grad(f_B_r_bc)
            dxu = [
                d_chi,
                d_g_rr,
                d_g_tt,
                d_A_rr_bc,
                d_K,
                d_Gamma_r,
                d_alpha,
                d_beta_r,
                d_B_r_bc,
            ]
            if self.lapse_gauge.n_aux:
                dxu.append(d_lapse_aux)
            for m in range(len(u)):
                if m == self.U_GTT:
                    # The spherical coordinate background is r**2, not a
                    # constant asymptote. Radiate its dimensionless deviation.
                    q = u[m] / r**2 - 1.0
                    f_q = filderivs.filter(q) if filderivs is not None else q
                    dq = g.D1.grad(f_q)
                    if self.sommerfeld_tail:
                        tail_bc.apply(
                            dtu[m], q, dq, r, 1.0, 0.0, self.u_wavespeed[m],
                            *self._tail_geometry, spherical_gtt=True,
                        )
                        continue
                    BSSN.bc_spherical_gtt(
                        dtu[m], q, dq, r,
                        self.u_wavespeed[m], ng, self.extended_domain,
                    )
                    continue
                if self.sommerfeld_tail:
                    tail_bc.apply(
                        dtu[m], u[m], dxu[m], r, self.u_falloff[m],
                        self.u_inf[m], self.u_wavespeed[m], *self._tail_geometry,
                    )
                    continue
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
    def laplacian_alpha(chi, g_rr, g_tt, d_alpha, d_chi, d_g_rr, d_g_tt, d2_alpha):
        """
        Covariant Laplacian of the lapse in 1D spherical symmetry,

            D^2 alpha = gamma^{ij} D_i D_j alpha,

        with gamma_{ij} = chi^{-1} diag(g_rr, g_tt, g_tt sin^2 theta).
        This is minus the four alpha-derivative terms of dt K.
        """
        inv_g_rr = 1.0 / g_rr
        inv_g_tt = 1.0 / g_tt
        inv_g_rr2 = inv_g_rr * inv_g_rr

        return (
            chi * d2_alpha * inv_g_rr
            - chi * d_g_rr * d_alpha * inv_g_rr2 / 2
            + chi * d_g_tt * d_alpha * inv_g_rr * inv_g_tt
            - d_alpha * d_chi * inv_g_rr / 2
        )

    @staticmethod
    @njit
    def rhs_K(
        K_rhs,
        alpha,
        A_rr,
        K,
        g_rr,
        beta_r,
        ad_K,
        lap_alpha,
    ):

        inv_g_rr = 1.0 / g_rr
        inv_g_rr2 = inv_g_rr * inv_g_rr

        K_rhs[:] = (
            3 * alpha * A_rr**2 * inv_g_rr2 / 2
            + K**2 * alpha / 3
            + beta_r * ad_K
            - lap_alpha
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
        alpha2_f,
        eta,
        lambda_lapse,
        lambda_shift,
    ):
        # alpha2_f = alpha^2 f(alpha), the Bona-Masso coefficient of K
        # eta      = shift damping, a scalar or an array over r
        alpha_rhs[:] = lambda_lapse * beta_r * ad_alpha - alpha2_f * K
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
            set_inner_regularity(u[self.U_CHI], ng, 1)
            set_inner_regularity(u[self.U_GRR], ng, 1)
            set_inner_regularity(u[self.U_GTT], ng, 1)
            set_inner_regularity(u[self.U_ARR], ng, 1)
            set_inner_regularity(u[self.U_K], ng, 1)
            set_inner_regularity(u[self.U_GT], ng, -1)
            set_inner_regularity(u[self.U_ALPHA], ng, 1)
            set_inner_regularity(u[self.U_SHIFT], ng, -1)
            set_inner_regularity(u[self.U_GB], ng, -1)

        if self.lapse_gauge.n_aux and not self.extended_domain:
            set_inner_regularity(u[self.U_LAPSE_AUX], ng, 1)  # A is even, like alpha

        if self.enforce_det:
            enforce_unit_det(u[self.U_GRR], u[self.U_GTT], g.xi[0])
            if not self.extended_domain:
                # The ghosts' r differs from their mirrors' by round-off, so
                # the rescale alone breaks parity at 1e-15; refill them.
                set_inner_regularity(u[self.U_GRR], ng, 1)
                set_inner_regularity(u[self.U_GTT], ng, 1)

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
    def bc_spherical_gtt(dtu, q, dq, r, v, ng, extended_domain):
        """Sommerfeld condition for gtt, applied to its deviation from flat space.

        Writing gtt = r**2 (1 + q), q falls off like 1/R (R = |r|), so the
        outgoing-wave condition is q_t = -v (q_R + q/R) and d_t gtt = r**2 q_t;
        sign(r) points the derivative outward on the negative-r side.  Only
        the outer ng cells (both ends on the extended domain) are changed.

        dq should be the grid derivative of q itself: differentiating gtt and
        subtracting 2 gtt/r would cancel two large r**2 terms.  Under derivative
        filtering dq comes from the filtered q while q stays unfiltered, as for
        the other fields.
        """
        regions = [slice(-ng, None)]
        if extended_domain:
            regions.append(slice(0, ng))
        for region in regions:
            R = np.abs(r[region])
            if np.any(R == 0):
                raise ValueError("Outer radial boundary cannot contain r=0")
            dtu[region] = -v * r[region]**2 * (
                np.sign(r[region]) * dq[region] + q[region] / R
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
        field's radial structure (n<0 takes u_inf = r[-1]**|n|). The supplied
        speed v is retained at both boundaries.  gtt uses bc_spherical_gtt
        instead, since its r**2 background is not a constant asymptote.
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

    def find_horizon(self, g: Grid, r0, interpolation="cubic", *, recover=False):
        """Return radius, mass and expansion. Record (degree, fallback) for a hit.

        Exact nodes use degree 1 with no fallback. A miss clears the record.
        With recover=True, retry outside the tracking window on a miss.
        """
        if interpolation not in ("linear", "cubic"):
            raise ValueError('horizon_interpolation must be "linear" or "cubic"')
        self.horizon_interpolation_used = None
        if r0 is None or not np.isfinite(r0) or r0 <= 0.0:
            r0 = 1.0

        r = g.xi[0]
        chi = self.u[self.U_CHI]
        g_rr = self.u[self.U_GRR]
        g_tt = self.u[self.U_GTT]
        A_rr = self.u[self.U_ARR]
        K = self.u[self.U_K]

        d_chi = g.D1.grad(chi)
        d_g_tt = g.D1.grad(g_tt)

        # Finite negative metric samples admit a diagnostic continuation.
        rTheta = np.full_like(chi, np.nan)
        usable = (np.isfinite(chi) & (chi != 0.0) & np.isfinite(g_rr)
                  & (g_rr != 0.0) & np.isfinite(g_tt) & (g_tt != 0.0))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for i in np.flatnonzero(usable):
                A_tt = -A_rr[i] * g_tt[i] / (2.0 * g_rr[i])
                d_phi = -d_chi[i] / (4.0 * chi[i])
                e2p = 1.0 / np.sqrt(abs(chi[i]))
                Kbar_tt = A_tt + g_tt[i] * K[i] / 3.0
                rTheta[i] = np.abs(r[i]) * (
                    (4.0 * g_tt[i] * d_phi + d_g_tt[i]) / (e2p * g_tt[i] * np.sqrt(abs(g_rr[i])))
                    - 2.0 * Kbar_tt / g_tt[i]
                )

        first = max(g.nghost, np.searchsorted(r, 0.0, side="right"))
        stop = len(r) - g.nghost
        if first < stop:
            bounds = (max(0.25 * r0, r[first]), min(4.0 * r0, r[stop - 1]))
            searches = [bounds]
            active = (r[first], r[stop - 1])
            if recover and bounds != active:
                searches.append(active)
            for attempt, (rin, rout) in enumerate(searches):
                if rin > rout:
                    continue
                result = _search_horizon(r, rTheta, g_rr, g_tt, chi, rin, rout,
                                         first, stop, interpolation)
                if result is not None:
                    rh, mass, self.horizon_interpolation_used = result
                    if attempt:
                        print("Horizon warning: recovered outside tracking window.")
                    return rh, mass, rTheta
        return None, None, rTheta


def _search_horizon(r, theta, radial, angular, chi, rin, rout, first, stop, interpolation):
    candidate_stop = stop
    cubic = interpolation == "cubic"
    while first < candidate_stop:
        rh, left = _find_ah_bracket(theta, r, rin, rout, first, candidate_stop, cubic)
        if rh is None:
            return None
        if left == -1:
            node = np.searchsorted(r, rh)
            samples = slice(node, node + 1)
            gtt_h, chi_h = angular[node], chi[node]
            candidate_stop = node
        else:
            samples = slice(left, left + 2)
            with np.errstate(over="ignore", invalid="ignore"):
                gtt_h = linear_interpolation(angular, r, rh)
                chi_h = linear_interpolation(chi, r, rh)
            candidate_stop = left + 1

        linear_in_bounds = rin <= rh <= rout
        mass, negative_area = _horizon_mass(gtt_h, chi_h) if linear_in_bounds else (None, False)
        degree = 1
        refine = cubic and left != -1
        if refine and first < left and left + 2 < stop:
            stencil = slice(left - 1, left + 3)
            samples = stencil
            result = _refine_horizon_cubic(
                r[stencil], theta[stencil], angular[stencil], chi[stencil], rin, rout
            )
            if result is not None:
                cubic_r, cubic_angular, cubic_chi = result
                cubic_mass, cubic_negative_area = _horizon_mass(cubic_angular, cubic_chi)
                clean = (
                    (radial[stencil] > 0.0).all() and (angular[stencil] > 0.0).all()
                    and (chi[stencil] > 0.0).all() and cubic_angular > 0.0 and cubic_chi > 0.0
                )
                if cubic_mass is not None and (clean or mass is None):
                    if clean:
                        return cubic_r, cubic_mass, (3, False)
                    rh, gtt_h, chi_h = result
                    mass, negative_area = cubic_mass, cubic_negative_area
                    degree = 3

        if degree == 1 and not linear_in_bounds:
            continue
        warnings = []
        if (chi[samples] <= 0.0).any() or chi_h <= 0.0:
            warnings.append("nonpositive chi")
        if ((radial[samples] <= 0.0).any() or (angular[samples] <= 0.0).any()
                or gtt_h <= 0.0):
            warnings.append("nonpositive conformal metric")
        fallback = refine and degree == 1
        if fallback:
            warnings.append("linear fallback")
        if degree == 3:
            warnings.append("linear area undefined" if linear_in_bounds else "linear root out of bounds")
        if negative_area:
            warnings.append("absolute area")
        if mass is None:
            warnings.append("undefined area at crossing")
        if warnings:
            print(f"Horizon warning at r={rh:g}: {', '.join(warnings)}.")
        if mass is not None:
            return rh, mass, (degree, fallback)
    return None


def _horizon_mass(angular, chi):
    if not (np.isfinite(angular) and np.isfinite(chi)) or chi == 0.0:
        return None, False
    area = 4.0 * np.pi * float(angular) / float(chi)
    if not np.isfinite(area):
        area = (float(angular) / float(chi)) * (4.0 * np.pi)
    if not np.isfinite(area):
        return None, False
    return np.sqrt(abs(area) / (16.0 * np.pi)), area < 0.0


def _cubic_coefficients(nodes, values):
    """Power coefficients in cell units, based at the left bracket node."""
    a, b = nodes[0], nodes[3]
    outer_left, left, right, outer_right = values
    slope = right - left
    left_slope = (left - outer_left) / -a
    right_slope = (outer_right - right) / (b - 1.0)
    second = (slope - left_slope) / (1.0 - a)
    third = ((right_slope - slope) / b - second) / (b - a)
    return left, slope - second + a * third, second - (1.0 + a) * third, third


def _evaluate_cubic(coeff, z):
    c0, c1, c2, c3 = coeff
    return c0 + z * (c1 + z * (c2 + z * c3))


def _refine_horizon_cubic(r, theta, angular, chi, rin, rout):
    """Refine the selected cell and interpolate the angular and conformal fields."""
    if not (np.isfinite(theta).all() and np.isfinite(angular).all()
            and np.isfinite(chi).all()):
        return None
    dx = r[2] - r[1]
    nodes = (r - r[1]) / dx
    with np.errstate(over="ignore", invalid="ignore"):
        coeff = _cubic_coefficients(nodes, theta)
        _, c1, c2, c3 = coeff
        # The derivative's three Bernstein coefficients must agree in sign.
        # This sufficient condition certifies monotonicity throughout [0, 1].
        direction = 1.0 if theta[2] > 0.0 else -1.0
        derivative = direction * np.array([c1, c1 + c2, c1 + 2.0 * c2 + 3.0 * c3])
        end = _evaluate_cubic(coeff, 1.0)
        if not (np.isfinite(derivative).all() and (derivative > 0.0).all()
                and np.isfinite(end) and direction * end > 0.0):
            return None

        lo, hi = 0.0, 1.0
        for _ in range(48):
            z = (lo + hi) / 2.0
            value = _evaluate_cubic(coeff, z)
            if not np.isfinite(value):
                return None
            if value == 0.0:
                lo = hi = z
                break
            if direction * value < 0.0:
                lo = z
            else:
                hi = z
        z = (lo + hi) / 2.0
        root = r[1] + dx * z
        if not (r[1] <= root <= r[2] and rin <= root <= rout):
            return None
        return (root, _evaluate_cubic(_cubic_coefficients(nodes, angular), z),
                _evaluate_cubic(_cubic_coefficients(nodes, chi), z))


@njit
def _find_ah_bracket(f, x, x1, x2, first, stop, include_clipped=False):
    """Return the outermost eligible linear root and cell index (-1 for a node)."""
    lo, hi = min(x1, x2), max(x1, x2)
    if (len(x) == 0 or not np.isfinite(x1) or not np.isfinite(x2)
            or lo < x[0] or hi > x[-1]):
        print(f"...find_ah_zero: x1={x1} and x2={x2} are out of range.")
        return None, -1

    # Include both cells cut by the bounds, with both nodes inside the allowed
    # grid. Traversal follows decreasing radius magnitude on a signed search.
    node_lo = max(first, np.searchsorted(x, lo, side="right") - 1)
    node_hi = min(stop - 1, np.searchsorted(x, hi))
    outer_bound = x1 if abs(x1) > abs(x2) else x2
    step = 1 if outer_bound == lo else -1
    start, end = (node_lo, node_hi) if step == 1 else (node_hi, node_lo)
    for k in range(start, end + step, step):
        fk = f[k]
        if fk == 0.0 and lo <= x[k] <= hi:
            return x[k], -1
        if k == end:
            break
        j = k + step
        fj = f[j]
        if (np.isfinite(fk) and np.isfinite(fj)
                and ((fk < 0.0 < fj) or (fj < 0.0 < fk))):
            # Keep legacy arithmetic for ordinary scales.
            numerator = fk * (x[k] - x[j])
            denominator = fk - fj
            if numerator != 0.0 and np.isfinite(numerator) and np.isfinite(denominator):
                root = x[k] - numerator / denominator
            else:
                scale = max(abs(fk), abs(fj))
                fraction = (fk / scale) / (fk / scale - fj / scale)
                root = x[k] - fraction * (x[k] - x[j])
            if include_clipped or lo <= root <= hi:
                return root, min(k, j)
    return None, -1


@njit
def find_ah_zero(f: np.ndarray, x: np.ndarray, x1: float, x2: float):
    """Find a nodal zero or linear crossing inside the requested bounds."""
    root, _ = _find_ah_bracket(f, x, x1, x2, 0, len(x))
    return root


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


def enforce_unit_det(g_rr, g_tt, r):
    """Rescale g_rr, g_tt in place by det^(-1/3) so g_rr (g_tt/r^2)^2 = 1.

    The isotropic rescaling is the closest point on det = 1 in the
    log-metric sense and keeps the ratio g_rr r^2 / g_tt.  A non-positive
    or non-finite det is an evolution failure: raise rather than mask it.
    """
    det = g_rr * (g_tt / (r * r)) ** 2
    if not (np.all(np.isfinite(det)) and np.all(det > 0.0)):
        raise FloatingPointError("enforce_det: non-positive or non-finite det(gtilde)")
    fact = 1.0 / np.cbrt(det)
    g_rr *= fact
    g_tt *= fact


def set_inner_regularity(u, ng, parity=1):
    """
    Apply inner boundary condition at the origin.
    """
    u[:ng] = parity * u[2 * ng - 1 : ng - 1 : -1]
