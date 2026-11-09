from .eqs import Equations
from .grid import Grid
from .types import *
from .filters import *
from .utils import l2norm, write_matrix_to_file
import numpy as np


class RK4:
    def __init__(self, e: Equations, g: Grid):
        self.k1 = []
        self.k2 = []
        self.k3 = []
        self.k4 = []
        self.us = []

        self.do_rhs_filter = False
        if g.num_filters > 0:
            for fx in g.Filter:
                if fx.get_apply_filter() == FilterApply.RHS:
                    self.do_rhs_filter = True

        for i in range(e.Nu):
            self.k1.append(np.zeros(tuple(e.shp)))
            self.k2.append(np.zeros(tuple(e.shp)))
            self.k3.append(np.zeros(tuple(e.shp)))
            self.k4.append(np.zeros(tuple(e.shp)))
            self.us.append(np.zeros(tuple(e.shp)))

    def step(self, e: Equations, g: Grid, dt):
        LTRACE = False

        nu = len(e.u)
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4
        us = self.us
        u0 = e.u

        assert len(k4) == e.Nu, "RK: wrong number of work arrays"
        # print(f"k3 shape = {k3[0].shape}")

        rhsfilters = []
        if self.do_rhs_filter:
            for fx in g.Filter:
                if fx.get_apply_filter() == FilterApply.RHS:
                    rhsfilters.append(fx)

        # Stage 1
        if LTRACE:
            print(
                f"@@@ RK1A |Ex|={l2norm(u0[0])}, |Ey|={l2norm(u0[1])}, |Hz|={l2norm(u0[2])}"
            )
            write_matrix_to_file("Hz_begin", u0[2])
        e.rhs(k1, u0, g)
        if self.do_rhs_filter:
            for i in range(nu):
                # print(f"calling filter {i}. sigma = {g.Filter.get_sigma()}")
                for rhsfilter in rhsfilters:
                    wrk = rhsfilter.filter(u0[i])
                    k1[i] += wrk
        for i in range(nu):
            us[i][:] = u0[i][:] + 0.5 * dt * k1[i][:]

        if LTRACE:
            print(
                f"@@@ RK1B |Ex|={l2norm(us[0])}, |Ey|={l2norm(us[1])}, |Hz|={l2norm(us[2])}"
            )
            write_matrix_to_file("Ex_B", us[0])
            write_matrix_to_file("Ey_B", us[1])

        if e.apply_bc == BCType.FUNCTION:
            e.apply_bcs(us, g)
        if LTRACE:
            print(
                f"@@@ RK1C |Ex|={l2norm(us[0])}, |Ey|={l2norm(us[1])}, |Hz|={l2norm(us[2])}"
            )
            write_matrix_to_file("Ex_C", us[0])
            write_matrix_to_file("Ey_C", us[1])

        # Stage 2
        e.rhs(k2, us, g)
        if self.do_rhs_filter:
            for i in range(nu):
                for rhsfilter in rhsfilters:
                    wrk = rhsfilter.filter(us[i])
                    k2[i] += wrk
        for i in range(nu):
            us[i][:] = u0[i][:] + 0.5 * dt * k2[i][:]
        if e.apply_bc == BCType.FUNCTION:
            e.apply_bcs(us, g)

        # Stage 3
        e.rhs(k3, us, g)
        if self.do_rhs_filter:
            for i in range(nu):
                for rhsfilter in rhsfilters:
                    wrk = rhsfilter.filter(us[i])
                    k3[i] += wrk
        for i in range(nu):
            us[i][:] = u0[i][:] + dt * k3[i][:]
        if e.apply_bc == BCType.FUNCTION:
            e.apply_bcs(us, g)

        # Stage 4
        e.rhs(self.k4, us, g)
        if self.do_rhs_filter:
            for i in range(nu):
                for rhsfilter in rhsfilters:
                    wrk = rhsfilter.filter(us[i])
                    k4[i] += wrk
        for i in range(nu):
            u0[i][:] += dt / 6 * (k1[i][:] + 2 * k2[i][:] + 2 * k3[i][:] + k4[i][:])
        if e.apply_bc == BCType.FUNCTION:
            e.apply_bcs(u0, g)
