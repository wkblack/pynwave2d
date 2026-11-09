"""Unmasked radial Gamma constraint and optional CCD adjustment diagnostics.

Here b = g_theta_theta includes r**2. Norms use coordinate dr (not volume).
No positivity repair, puncture exclusion, or horizon mask is applied here.
"""
import numpy as np


def gamma_constraint(a, b, Gamma, da, db):
    """Signed C^r = evolved Gamma^r - contracted metric connection."""
    return Gamma - da / (2 * a * a) + db / (a * b)


def shift_divergence(a, b, beta, da, db, dbeta):
    """Metric-derived divergence, retained for convention comparisons only."""
    return dbeta + beta * (da / (2 * a) + db / b)


def reference_shift_divergence(beta, dbeta, r):
    """NRPy fixed-reference-determinant divergence: beta' + 2 beta/r.

    The puncture solver uses a cell-centered grid excluding r=0. Do not
    silently invent a regular-origin limit for puncture data or mask cells.
    dbeta follows the caller's derivative policy; beta is the raw stage field.
    """
    if np.any(np.asarray(r) == 0):
        raise ValueError("Reference CCD divergence requires a grid excluding r=0")
    return dbeta + 2.0 * beta / r


def diagnostic_fields(u, g):
    """Fresh raw-state diagnostics; D1 of unfiltered state, no ghost zeroing.

    Unlike RHS derivative filtering these diagnose the actual evolved fields.
    Negative metric components remain visible (magnitude then NaN), not clipped.
    """
    a, b, Gamma, beta = u[1], u[2], u[5], u[7]
    da, db, dbeta = (g.D1.grad(x) for x in (a, b, beta))
    C = gamma_constraint(a, b, Gamma, da, db)
    div = reference_shift_divergence(beta, dbeta, g.xi[0])
    with np.errstate(invalid='ignore'):
        magnitude = np.sqrt(a) * np.abs(C)
    return C, magnitude, div


def diagnostic_summary(u, g, c_G, dt, shell_inner=1., shell_outer=10.):
    """Fixed |r| shell RMS/L2 in dr; health/rate extrema on all physical cells.

    Both mirrored halves contribute, with no normalization beyond stated RMS.
    Excludes only declared ghost cells; shell never follows the apparent horizon.
    """
    if not (np.isfinite(shell_inner) and np.isfinite(shell_outer)
            and 0 <= shell_inner < shell_outer):
        raise ValueError('Gamma diagnostic shell must satisfy 0 <= inner < outer')
    r = g.xi[0]
    physical = np.ones(len(r), dtype=bool)
    if g.nghost:
        physical[:g.nghost] = False
        physical[-g.nghost:] = False
    shell = physical & (np.abs(r) >= shell_inner) & (np.abs(r) <= shell_outer)
    if not np.any(shell):
        raise ValueError('Gamma diagnostic shell contains no physical cells')
    C, mag, div = diagnostic_fields(u, g)
    rate_dt = c_G * div * dt
    result = {'c_G': c_G, 'shell_inner': shell_inner, 'shell_outer': shell_outer,
              'shell_points': int(shell.sum()), 'dr': g.dx[0]}
    for name, values in [('C_signed', C), ('C_magnitude', mag)]:
        result[name + '_mean_dr_shell'] = float(np.mean(values[shell]))
        result[name + '_rms_dr_shell'] = float(np.sqrt(np.mean(values[shell]**2)))
        result[name + '_l2_dr_shell'] = float(np.sqrt(g.dx[0]*np.sum(values[shell]**2)))
        result[name + '_linf_shell'] = float(np.max(np.abs(values[shell])))
    for name, values in [('chi', u[0]), ('g_rr', u[1]), ('g_tt', u[2]),
                         ('divbeta', div), ('cG_divbeta_dt', rate_dt)]:
        result[name + '_min_physical'] = float(np.min(values[physical]))
        result[name + '_max_physical'] = float(np.max(values[physical]))
    result['chi_nonpositive_count_physical'] = int(np.sum(u[0][physical] <= 0))
    result['finite_state'] = bool(np.isfinite(np.asarray(u)).all())
    result['finite_gamma_diagnostics_physical'] = bool(all(np.isfinite(v[physical]).all() for v in (C, mag, div)))
    result['gamma_antidamping_fraction_physical'] = float(np.mean(rate_dt[physical] < 0))
    return result
