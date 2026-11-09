import numpy as np
import sys
import os
import csv

# from pdb import set_trace as bp

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bssneqs as bssn
import bssnutils as butil
import tomllib
from nwave import *


def main(parfile):
    # Read parameters
    with open(parfile, "rb") as f:
        params = tomllib.load(f)

    # Verify parameters
    butil.verify_params(params)

    # Set up grid
    level = params.get("level", 0)
    nx0 = params["Nx"]
    if nx0 % 2 != 0:
        nx0 -= 1
        print(f"Adjusted Nx to {nx0} (must be even)")

    if "Xmin" in params and params["Xmin"] < 0.0:
        extended_domain = True
        cellgrid = True
        params["Nx"] = (1 << level) * nx0 + 1
    else:
        extended_domain = False
        cellgrid = True
        params["Nx"] = (1 << level) * nx0 + 1

    nghost = params["NGhost"]

    g = Grid1D(params, cell_centered=cellgrid)

    r = g.xi[0]
    dr = g.dx[0]
    Nr = len(r)

    D1, D2 = butil.init_derivative_operators(r, params)
    g.set_D1(D1)
    g.set_D2(D2)
    print(f"D1 type: {g.D1.get_type()}")
    print(f"D2 type: {g.D2.get_type()}")

    F1 = butil.init_filter(r, params)
    g.set_filter(F1)

    if g.num_filters > 0:
        for f in g.Filter:
            print(f"f type = {type(f)}")
            print(f"Filter type: {f.get_filter_type()}")
            print(f"Filter apply: {f.get_apply_filter()}")

    # GBSSN system: (sys, lapse advection, shift advection)
    #    sys = 0 (Eulerian), 1 (Lagrangian)
    bssn_outer_bc_str = params.get("bssn_outer_bc", "Sommerfeld")
    bssn_outer_bc = butil.get_bssn_outer_bc(bssn_outer_bc_str)
    #bssn_outer_bc_sponge_width = params.get("bssn_outer_bc_sponge_width", 20.0)

    sys = bssn.GBSSNSystem(1, 1, 1)
    eqs = bssn.BSSN(
        g,
        params["Mass"],
        params["eta"],
        extended_domain,
        BCType.FUNCTION,
        sys,
        outer_bc=bssn_outer_bc,
        have_d2=True,
    )
    eqs.initialize(g, params)

    output_dir = params["output_dir"] + "_" + str(level)
    output_interval = params["output_interval"] * (1 << level)
    print_interval = params["print_interval"] * (1 << level)

    os.makedirs(output_dir, exist_ok=True)

    Nt = params["Nt"] * (1 << level)
    time = 0.0
    dt = params["cfl"] * dr
    rk4 = RK4(eqs, g)

    eqs.cal_constraints(eqs.u, g)
    rbh_guess = 0.7
    rbh, mbh, rTheta = eqs.find_horizon(g, rbh_guess)
    horizon_found = rbh is not None and np.isfinite(rbh) and rbh > 0.0
    rbh, mbh, rbh_guess, mask_radius = update_horizon_tracking(
        rbh, mbh, rbh_guess, 0.0
    )
    eqs.set_ah(rbh, mbh)
    bhpts = int(rbh * len(r) / (r[-1] - r[0]))
    print(f"Horizon(s) found at {rbh:03f} with masses {mbh:03f} and bhpts={bhpts:d}")
    rmask = params.get("bhmask_outer_radius", 75.0)
    bhmask = np.ones(Nr, dtype=int)
    butil.set_bhmask(bhmask, r, mask_radius, rmask)

    step = 0
    fname = f"{output_dir}/bssn_{step:04d}.curve"
    butil.write_curve(fname, 0.0, g.xi[0], eqs)

    #fname = f"{output_dir}/rtheta_{step:04d}.curve"
    #butil.write_curve_functions(fname, ["rTheta"], time, g.xi[0], [rTheta])

    # Create a CSV file for constraint norms
    conname = f"{output_dir}/bssn_constraints.dat"
    confile = open(conname, mode="w", newline="")
    writer = csv.writer(confile)
    writer.writerow(["time", "Ham", "Mom", "radius_bh", "mass_bh", "horizon_found", "mask_inner_radius"])
    writer.writerow(
        [
            time,
            l2norm_mask(eqs.C[0], dr, bhmask),
            l2norm_mask(eqs.C[1], dr, bhmask),
            rbh,
            mbh,
            horizon_found,
            mask_radius,
        ]
    )

    filvar = None
    filter_frequency = -1
    if g.num_filters > 0:
        for fx in g.Filter:
            if fx.get_apply_filter == FilterApply.APPLY_VARS:
                filvar = fx
                filter_frequency = fx.get_frequency()
                break

    for i in range(1, Nt + 1):
        rk4.step(eqs, g, dt)
        if filter_frequency > 0 and i % filter_frequency == 0:
            # Apply filter to the variables
            print("Applying filter to variables")
            for j in range(eqs.Nu):
                eqs.u[j][:] = filvar.filter(eqs.u[j])

        time += dt
        if i % print_interval == 0 or i % output_interval == 0:
            eqs.cal_constraints(eqs.u, g)
            rbh, mbh, rTheta = eqs.find_horizon(g, rbh_guess)
            horizon_found = rbh is not None and np.isfinite(rbh) and rbh > 0.0
            rbh, mbh, rbh_guess, mask_radius = update_horizon_tracking(
                rbh, mbh, rbh_guess, mask_radius
            )
            eqs.set_ah(rbh, mbh)
            bhpts = int(rbh * len(r) / (r[-1] - r[0]))
            print(
                f"Horizon(s) found at {rbh:.03f} with masses {mbh:03f} and pts (radius)={bhpts:d}."
            )
            butil.set_bhmask(bhmask, r, mask_radius, rmask)
        if i % print_interval == 0:
            hamnorm = l2norm_mask(eqs.C[0], dr, bhmask)
            momnorm = l2norm_mask(eqs.C[1], dr, bhmask)
            print(
                f"Step {i:d}, t={time:.2e}, |chi|={l2norm_mask(eqs.u[0], dr, bhmask):.2e}, |grr|={l2norm_mask(eqs.u[1], dr, bhmask):.2e}, |gtt|={l2norm_mask(eqs.u[2], dr, bhmask):.2e}, |Arr|={l2norm_mask(eqs.u[3], dr, bhmask):.2e}, |K|={l2norm_mask(eqs.u[4], dr, bhmask):.2e}, |Gt|={l2norm_mask(eqs.u[5], dr, bhmask):.2e}, |alpha|={l2norm_mask(eqs.u[6], dr, bhmask):.2e}, |beta|={l2norm_mask(eqs.u[7], dr, bhmask):.2e}, |gB|={l2norm_mask(eqs.u[8], dr, bhmask):.2e}, |Ham|={hamnorm:.2e}, |Mom|={momnorm:.2e}"
            )
            rbh, mbh = eqs.get_ah()
            writer.writerow([time, hamnorm, momnorm, rbh, mbh, horizon_found, mask_radius])
        if i % output_interval == 0:
            fname = f"{output_dir}/bssn_{i:04d}.curve"
            butil.write_curve(fname, time, g.xi[0], eqs)
            #fname = f"{output_dir}/rtheta_{i:04d}.curve"
            #butil.write_curve_functions(fname, ["rTheta"], time, g.xi[0], [rTheta])

        if np.isnan(eqs.u[1]).any():
            print("Solution has a NaN.  Bye.")
            break

    confile.close()


def update_horizon_tracking(radius, mass, previous_guess, previous_mask_radius):
    """Keep a valid search guess and explicitly retain the last valid mask."""
    if radius is not None and np.isfinite(radius) and radius > 0.0:
        return radius, mass, radius, radius
    print("Horizon not found; retaining search guess and last valid mask radius.")
    return 0.0, 0.0, previous_guess, previous_mask_radius


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:  python solver.py <parfile>")
        sys.exit(1)

    parfile = sys.argv[1]
    main(parfile)
