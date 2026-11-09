"""Unmasked Hamiltonians for reference-metric CAHD (full angular metric b).

H_g is geometric. H_B_ref is the attachment's NRPy reference-metric BSSN
trace: H_g + chi * Dhat_i C^i. Native pynwave's trace-free Ricci evolution
uses a raw-coordinate connection convention; we do NOT modify that evolution.
"""
import numpy as np
from gamma import gamma_constraint


def hamiltonians(chi, a, b, Arr, K, Gamma, dchi, da, db,
                 d2chi, d2a, d2b, dGamma, r):
    """Return fresh (H_g, H_B_ref), without masks, floors, or ghost zeroing.

    C' is expanded analytically using the supplied first/second derivative
    operators, not D1 applied a second time to a discretized connection.
    The spherical reference divergence is C' + 2C/r, NOT just C'.
    """
    ia, ib = 1.0 / a, 1.0 / b
    Hg = (-1.5 * Arr**2 * ia**2 + (2.0 / 3.0) * K**2
          - 2.5 * dchi**2 * ia / chi + 2.0 * d2chi * ia
          + 2.0 * chi * ib - 2.0 * chi * d2b * ia * ib
          + 2.0 * dchi * db * ia * ib + chi * da * db * ia**2 * ib
          - dchi * da * ia**2 + 0.5 * chi * db**2 * ia * ib**2)
    C = gamma_constraint(a, b, Gamma, da, db)
    dGamma_metric = (0.5 * d2a * ia**2 - da**2 * ia**3
                     - d2b * ia * ib + da * db * ia**2 * ib
                     + db**2 * ia * ib**2)
    HB = Hg + chi * (dGamma - dGamma_metric + 2.0 * C / r)
    return Hg, HB


def chi_adjustment(chi, HB, c_H, h, cfl):
    """phi_t -= c_H*(h/CFL)*H_B implies chi_t += 4*chi*c_H*(h/CFL)*H_B."""
    return 4.0 * c_H * (h / cfl) * chi * HB


def diagnostic_fields(u, g, have_d2=True):
    """Raw-state (unfiltered) H_g and H_B_ref using the configured D2 policy."""
    chi, a, b, Arr, K, Gamma = u[:6]
    dchi, da, db, dGamma = (g.D1.grad(x) for x in (chi, a, b, Gamma))
    d2chi, d2a, d2b = (g.D2.grad(x) for x in (chi, a, b)) if have_d2 else (
        g.D1.grad(x) for x in (dchi, da, db))
    return hamiltonians(chi, a, b, Arr, K, Gamma, dchi, da, db,
                        d2chi, d2a, d2b, dGamma, g.xi[0])


def diagnostic_summary(u, g, c_H, cfl, shell_inner=1., shell_outer=10., have_d2=True):
    """Fixed-shell coordinate-dr norms, no apparent-horizon mask."""
    if not (np.isfinite(shell_inner) and np.isfinite(shell_outer)
            and 0 <= shell_inner < shell_outer):
        raise ValueError('Hamiltonian diagnostic shell must satisfy 0 <= inner < outer')
    r = g.xi[0]
    physical = np.ones(len(r), dtype=bool)
    if g.nghost:
        physical[:g.nghost] = False
        physical[-g.nghost:] = False
    shell = physical & (np.abs(r) >= shell_inner) & (np.abs(r) <= shell_outer)
    if not np.any(shell):
        raise ValueError('Hamiltonian diagnostic shell contains no physical cells')
    Hg, HB = diagnostic_fields(u, g, have_d2)
    result = {'c_H': c_H, 'cfl': cfl, 'dr': g.dx[0],
              'kappa_H': c_H * g.dx[0] / cfl,
              'shell_inner': shell_inner, 'shell_outer': shell_outer,
              'shell_points': int(shell.sum())}
    for name, values in [('H_g', Hg), ('H_B_ref', HB), ('H_B_ref_minus_H_g', HB-Hg)]:
        result[name+'_rms_dr_shell'] = float(np.sqrt(np.mean(values[shell]**2)))
        result[name+'_l2_dr_shell'] = float(np.sqrt(g.dx[0]*np.sum(values[shell]**2)))
        result[name+'_linf_shell'] = float(np.max(np.abs(values[shell])))
    # dt=CFL*h in this uniform-grid solver: 8*kappa*chi/a*dt/h**2.
    # Multiply by the dimensionless Laplacian symbol for a mode's RK argument.
    diffusion_step = 8.0 * c_H * u[0] / u[1]
    result['H_diffusion_step_min_physical'] = float(np.min(diffusion_step[physical]))
    result['H_diffusion_step_max_physical'] = float(np.max(diffusion_step[physical]))
    result['finite_hamiltonians_physical'] = bool(np.isfinite(Hg[physical]).all() and np.isfinite(HB[physical]).all())
    return result
