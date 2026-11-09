import numpy as np
import sys
import os

# from pdb import set_trace as bp

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nwave import *
import bssneqs as bssn


class ConfigError(Exception):
    """Raised for invalid parameter-file configuration."""
    pass


def write_curve(filename, time, x, eqs):
    with open(filename, "w") as f:
        f.write(f"# TIME {time}\n")
        for m in range(len(eqs.u_names)):
            f.write(f"# {eqs.u_names[m]}\n")
            for xi, di in zip(x, eqs.u[m]):
                f.write(f"{xi:.8e} {di:.8e}\n")
        for m in range(len(eqs.c_names)):
            f.write(f"# {eqs.c_names[m]}\n")
            for xi, di in zip(x, eqs.C[m]):
                f.write(f"{xi:.8e} {di:.8e}\n")


def write_curve_masked(filename, time, x, eqs, mask):
    umask = np.empty_like(x)
    with open(filename, "w") as f:
        f.write(f"# TIME {time}\n")
        for m in range(len(eqs.u_names)):
            f.write(f"# {eqs.u_names[m]}\n")
            umask[:] = eqs.u[m][:] * mask[:]
            for xi, di in zip(x, umask):
                f.write(f"{xi:.8e} {di:.8e}\n")
        for m in range(len(eqs.c_names)):
            f.write(f"# {eqs.c_names[m]}\n")
            umask[:] = eqs.C[m][:] * mask[:]
            for xi, di in zip(x, umask):
                f.write(f"{xi:.8e} {di:.8e}\n")


def write_curve_functions(filename, namelist, time, x, ulist):
    """
    Write a list of functions (ulist) to a curve file.
    The function names are in the list (namelist).
    """
    with open(filename, "w") as f:
        f.write(f"# TIME {time}\n")
        for uname, u in zip(namelist, ulist):
            f.write(f"# {uname}\n")
            for xpt, upt in zip(x, u):
                f.write(f"{xpt:.8e} {upt:.8e}\n")


def verify_params(params):
    if params["Nx"] < 10:
        raise ValueError("Nx must be at least 10")
    if params["NGhost"] < 2:
        raise ValueError("NGhost must be at least 2")
    if params["NGhost"] > 4:
        raise ValueError("NGhost must be at most 4")
    if params["cfl"] <= 0.0:
        raise ValueError("cfl must be positive")
    if params["cfl"] >= 1.0:
        raise ValueError("cfl must be less than 1.0")
    if params["Nt"] <= 0:
        raise ValueError("Nt must be positive")
    if params["output_interval"] <= 0:
        raise ValueError("output_interval must be positive")
    if params["print_interval"] <= 0:
        raise ValueError("print_interval must be positive")
    if params["output_interval"] < params["print_interval"]:
        raise ValueError("output_interval must be greater than print_interval")
    if params["output_dir"] == "":
        raise ValueError("output_dir must be a non-empty string")
    if params["Mass"] <= 0.0:
        raise ValueError("Mass must be positive")
    if params["eta"] < 1.0:
        raise ValueError("eta must be greater than or equal to 1.0")


def get_filter_type(filter_str):
    try:
        return filter_type_map[filter_str]
    except KeyError:
        raise ValueError(f"Unknown filter type string: '{filter_str}'")


def get_filter_apply(filter_str):
    try:
        return filter_apply_map[filter_str]
    except KeyError:
        raise ValueError(f"Unknown filter apply string: '{filter_str}'")


def get_d1_type(d_str):
    try:
        return d1_type_map[d_str]
    except KeyError:
        raise ValueError(f"Unknown D1 type string: '{d_str}'")


def get_d2_type(d_str):
    try:
        return d2_type_map[d_str]
    except KeyError:
        raise ValueError(f"Unknown D2 type string: '{d_str}'")


def get_cfd_solve(d_str):
    try:
        return cfd_solve_map[d_str]
    except KeyError:
        raise ValueError(f"Unknown Deriv Solve type string: '{d_str}'")

def get_bssn_outer_bc(d_str):
    try:
        return bssn.bssn_outer_bc_map[d_str]
    except KeyError:
        raise ValueError(f"Unknown BSSN outer bc type string: '{d_str}'")

def init_derivative_operators(r, params):
    d1_param = params["D1"]
    # Check if d1 is a string or a list of strings
    if isinstance(d1_param, str):
        d1_list = [d1_param]
    elif isinstance(d1_param, list) and all(isinstance(item, str) for item in d1_param):
        d1_list = d1_param
    else:
        raise TypeError("D1 must be a string or a list of strings")

    d2_param = params["D2"]
    # Check if d2 is a string or a list of strings
    if isinstance(d2_param, str):
        d2_list = [d2_param]
    elif isinstance(d2_param, list) and all(isinstance(item, str) for item in d2_param):
        d2_list = d2_param
    else:
        raise TypeError("D2 must be a string or a list of strings")

    method_str = params.get("DerivSolveMethod", "LUSOLVE")
    method = get_cfd_solve(method_str)
    mask_bh = params.get("BHMask", False)

    if mask_bh and len(d1_list) != 2:
        raise TypeError("D1 must be a list of two operators for BHMask")

    print(f"init_derivative_operators>>  d1_param = {d1_param}")
    print(f"init_derivative_operators>>  d2_param = {d2_param}")
    print(f"init_derivative_operators>>  d1_list = {d1_list}")
    print(f"init_derivative_operators>>  mask_bh  = {mask_bh}")
    dr = r[1] - r[0]

    if mask_bh:
        rbh = params.get("BHMaskPos", 0.0)
        rbh_width = params.get("BHMaskWidth", 0.1)
        d1_0_type = get_d1_type(d1_list[0])
        d1_bh_type = get_d1_type(d1_list[1])
        if d1_0_type and d1_bh_type in CompactFirstDerivatives:
            print(
                f"init_derivative_operators>> Setting BH Masked derivative for D1. Background type: {d1_0_type}, BH type: {d1_bh_type}"
            )
            print(
                f"init_derivative_operators>> BH Mask. position: {rbh}, width: {rbh_width}"
            )
            D1 = NCompactDerivative.bh_deriv(
                r, d1_0_type, d1_bh_type, method, rbh, rbh_width
            )
        else:
            raise TypeError(
                "init_derivative_operators>> Invalid FIRST derivative types for masking"
            )

        d2_0_type = get_d2_type(d2_list[0])
        d2_bh_type = get_d2_type(d2_list[1])
        if d2_0_type and d2_bh_type in CompactSecondDerivatives:
            print(
                f"init_derivative_operators>> Setting BH Masked derivative for D2. Background type: {d2_0_type}, BH type: {d2_bh_type}"
            )
            D2 = NCompactDerivative.bh_deriv(
                r, d2_0_type, d2_bh_type, method, rbh, rbh_width
            )
        else:
            raise TypeError(
                "init_derivative_operators>> Invalid SECOND derivative types for masking"
            )

    elif len(d1_list) >= 1:
        d1type = get_d1_type(d1_list[0])
        print(f"init_derivative_operators>> Setting D1 type: {d1type}")
        if d1type == DerivType.D1_E44:
            D1 = ExplicitFirst44_1D(dr)
        elif d1type == DerivType.D1_E642:
            D1 = ExplicitFirst642_1D(dr)
        elif d1type in CompactFirstDerivatives:
            D1 = NCompactDerivative.deriv(r, d1type, method)
        else:
            raise NotImplementedError("D1 Type = {d1type}")

        d2type = get_d2_type(d2_list[0])
        print(f"init_derivative_operators>>  Setting D2 type: {d2type}")
        if d2type == DerivType.D2_E44:
            D2 = ExplicitSecond44_1D(dr)
        elif d2type == DerivType.D2_E642:
            D2 = ExplicitSecond642_1D(dr)
        elif d2type in CompactSecondDerivatives:
            D2 = NCompactDerivative.deriv(r, d2type, method)
        else:
            raise NotImplementedError("D2 Type = {d2type}")
    else:
        raise RuntimeError(
            "init_derivative_operators>> Failed to initialize derivative operators: D1 and D2 are unbound."
        )

    return D1, D2

def _require(dct: dict[str, Any], key: str, kind: str = "entry"):
    if key not in dct:
        raise ConfigError(f"Missing required field '{key}' in {kind}.")
    return dct[key]

def parse_filter_entry(entry: dict[str, Any], r) -> Filter1D:
    if not isinstance(entry, dict):
        raise ConfigError(f"Each filter entry must be a table/dict, got {type(entry).__name__}.")

    dr = r[1] - r[0]
    ftype_str = _require(entry, "type", "filter type")
    ftype = get_filter_type(ftype_str)
    fapply_str = _require(entry, "apply", "filter apply")
    fapply = get_filter_apply(fapply_str)
    fsolve_str = entry.get("solve_method", "None")
    fsolve = get_cfd_solve(fsolve_str)
    fbounds = entry.get("filter_boundary", False)
    ffreq = int(entry.get("frequency", 1))
    if ffreq < 0:
        raise ValueError(f"Filter: 'frequency' must be 0 or positive if provided.")

    if ftype == FilterType.KREISS_OLIGER_O6:
        sigma = entry.get("sigma", 0.1)
        bssn_filter = KreissOligerFilterO6_1D(dr, sigma, filter_boundary=fbounds)
        return bssn_filter
    elif ftype == FilterType.KREISS_OLIGER_O8:
        sigma = entry.get("sigma", 0.1)
        bssn_filter = KreissOligerFilterO8_1D(dr, sigma, filter_boundary=fbounds)
        return bssn_filter
    elif ftype in CompactFilterTypes:
        alpha = entry.get("alpha", 0.0)
        beta = entry.get("beta", 0.0)
        mask_bh = False
        if mask_bh:
            #mask_pos = params.get("BHMaskPos", 0.0)
            #mask_width = params.get("BHMaskWidth", 0.1)
            mask_pos = 0.0
            mask_width = 0.1
            print(
                f"init_filter>> Creating BH masked filter: Background type: {ftype}, apply: {fapply}, method: {fmethod}"
            )
            print(
                f"init_derivative_operators>> BH Mask. position: {mask_pos}, width: {mask_width}"
            )
            bssn_filter = NCompactFilter.init_bh_filter(
                r,
                ftype,
                fapply,
                fsolve,
                ffreq,
                mask_pos,
                mask_width,
                fbounds,
                alpha,
                beta,
            )
            return bssn_filter
        else:
            print(
                f"init_filter>> Creating filter: type: {ftype}, apply: {fapply}, method: {fsolve}"
            )
            bssn_filter = NCompactFilter.init_filter(
                r, ftype, fapply, fsolve, ffreq, fbounds, alpha, beta
            )
            return bssn_filter
    elif ftype == FilterType.NONE:
        pass
    else:
        raise NotImplementedError(
            "init_filter>>  Filter = { KO6, KO8, JTT6, JTP6, JTT8, JTP8, KP4 }"
        )


def init_filter(r, params):
    parsed: list[Filter1D] = []
    if "filters" not in params:
        # Pure [[filters]] design: scalar Filter= keys are no longer supported.
        # Fail loudly rather than silently run unfiltered — the silent-none trap
        # invalidated an entire operator sweep. Migrate to a [[filters]] block.
        legacy = params.get("Filter", None)
        if isinstance(legacy, str) and legacy != "None":
            raise ConfigError(
                "Scalar 'Filter=' keys are no longer read. Use a [[filters]] "
                "array block instead, e.g.:\n"
                '[[filters]]\ntype = "KO8"\napply = "Rhs"\nsigma = 0.4\n'
                "filter_boundary = false"
            )
        parsed.append(FilterNone_1D())
        return parsed

    farray = params["filters"]
    if not isinstance(farray, list):
        raise ConfigError("'filters' must be an array of tables ([[filters]]).")

    for i, entry in enumerate(farray):
        spec = parse_filter_entry(entry, r)
        parsed.append(spec)

    return parsed

    


def Xinit_filter(r, params):
    f_param = params.get("Filter", "None")

    if isinstance(f_param, str):
        fparam_list = [f_param]
    elif isinstance(f_param, list) and all(isinstance(item, str) for item in f_param):
        fparam_list = f_param
    else:
        raise TypeError("init_filter>>  Filter must be a string or a list of strings")

    filters = []
    mask_bh = params.get("BHMask", False)

    for fstr in fparam_list:
        ftype = get_filter_type(fstr)
        if ftype == FilterType.KREISS_OLIGER_O6:
            sigma = params.get("FilterKOsigma", 0.1)
            fbounds = params.get("FilterBoundary", False)
            dr = r[1] - r[0]
            bssn_filter = KreissOligerFilterO6_1D(dr, sigma, filter_boundary=fbounds)
            filters.append(bssn_filter)
        elif ftype == FilterType.KREISS_OLIGER_O8:
            sigma = params.get("FilterKOsigma", 0.1)
            fbounds = params.get("FilterBoundary", False)
            dr = r[1] - r[0]
            bssn_filter = KreissOligerFilterO8_1D(dr, sigma, filter_boundary=fbounds)
            filters.append(bssn_filter)
        elif ftype in CompactFilterTypes:
            fapply = get_filter_apply(params.get("FilterApply", FilterApply.APPLY_VARS))
            fmethod = CFDSolve.SCIPY
            fbounds = params.get("FilterBoundary", False)
            ffreq = params.get("FilterFrequency", 1)
            alpha = params.get("FilterAlpha", 0.0)
            beta = params.get("FilterBeta", 0.0)
            if mask_bh:
                mask_pos = params.get("BHMaskPos", 0.0)
                mask_width = params.get("BHMaskWidth", 0.1)
                print(
                    f"init_filter>> Creating BH masked filter: Background type: {ftype}, apply: {fapply}, method: {fmethod}"
                )
                print(
                    f"init_derivative_operators>> BH Mask. position: {mask_pos}, width: {mask_width}"
                )
                bssn_filter = NCompactFilter.init_bh_filter(
                    r,
                    ftype,
                    fapply,
                    fmethod,
                    ffreq,
                    mask_pos,
                    mask_width,
                    fbounds,
                    alpha,
                    beta,
                )
                filters.append(bssn_filter)
            else:
                print(
                    f"init_filter>> Creating filter: type: {ftype}, apply: {fapply}, method: {fmethod}"
                )
                bssn_filter = NCompactFilter.init_filter(
                    r, ftype, fapply, fmethod, ffreq, fbounds, alpha, beta
                )
                filters.append(bssn_filter)
        elif ftype == FilterType.NONE:
            pass
        else:
            raise NotImplementedError(
                "init_filter>>  Filter = { KO6, KO8, JTT6, JTP6, JTT8, JTP8, KP4 }"
            )
    return filters

@njit
def set_bhmask(mask, r, rbh, rout):
    """
    Set the BH mask for the grid.
    The mask is set to 1 for points outside the BH radius and 0 for points inside.
    """
    for i in range(len(r)):
        if np.abs(r[i]) < rbh or np.abs(r[i]) > rout:
            mask[i] = 0
        else:
            mask[i] = 1
