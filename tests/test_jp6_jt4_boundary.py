"""JP6 closure consistency, independent moments, refinement and parity checks."""
from pathlib import Path
import math
import sys
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nwave.cd import NCompactDerivative, _coeffs_D1_JP6_JT4, _coeffs_D2_JP6_JT4
from nwave.types import DerivType, CFDSolve


def grid(n):
    h = 2.0 / n
    return np.arange(n + 8) * h - 1 - 2.5 * h


def operator(x, order, method=CFDSolve.LUSOLVE, masked=False):
    dtype = getattr(DerivType, f'D{order}_JP6_JT4')
    if masked:
        inner = getattr(DerivType, f'D{order}_ME642')
        return NCompactDerivative.bh_deriv(x, dtype, inner, method, 0.0, 0.05)
    return NCompactDerivative.deriv(x, dtype, method)


class JP6JT4BoundaryTests(unittest.TestCase):
    def test_boundary_and_interior_polynomial_moments(self):
        # Both P and Q rows must satisfy the derivative equation.
        for order, coeffs in [(1, _coeffs_D1_JP6_JT4), (2, _coeffs_D2_JP6_JT4)]:
            P, Q = coeffs()
            rows = [(P[1][i], Q[1][i], 4, False) for i in (0, 1)]
            rows.append((P[0], Q[0], 6, True))
            for row, (p, q, accuracy, centered) in enumerate(rows):
                xp = np.arange(len(p)) - (len(p)//2 if centered else 0)
                xq = np.arange(len(q)) - (len(q)//2 if centered else 0)
                for degree in range(order + accuracy):
                    expected = sum(w * math.factorial(degree) / math.factorial(degree-order)
                                   * j**(degree-order) for j, w in zip(xp, p)) if degree >= order else 0.
                    actual = sum(w * j**degree for j, w in zip(xq, q))
                    with self.subTest(order=order, row=row, degree=degree):
                        self.assertAlmostEqual(actual, expected, delta=2e-9)

    def test_all_row_polynomials_across_grids_and_backends(self):
        for n in (80, 160, 320):
            x = grid(n)
            for order in (1, 2):
                for method in (CFDSolve.LUSOLVE, CFDSolve.SCIPY):
                    for masked in (False, True):
                        d = operator(x, order, method, masked)
                        for degree in range(order + 1):
                            expected = np.full_like(x, math.factorial(order)) if degree == order else np.zeros_like(x)
                            error = d.grad(x**degree) - expected
                            with self.subTest(n=n, order=order, method=method, masked=masked, degree=degree):
                                np.testing.assert_allclose(error[3:-3], 0., rtol=0., atol=3e-9 if order == 2 else 3e-11)
                                # Outer allocated rows are evolved by the
                                # Sommerfeld closure, not disposable ghosts.
                                np.testing.assert_allclose(error, 0., rtol=0.,
                                                           atol=5e-8 if order == 2 else 5e-10)

    def test_fourth_order_boundary_and_sixth_order_interior_refinement(self):
        for order in (1, 2):
            errors = []
            for n in (40, 80, 160):
                x = grid(n)
                exact = 11*np.cos(11*x+.2) if order == 1 else -121*np.sin(11*x+.2)
                e = abs(operator(x, order).grad(np.sin(11*x+.2))-exact)
                errors.append([e.max(), max(e[3:11].max(), e[-11:-3].max()), e[abs(x)<.5].max()])
            rates = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
            self.assertTrue(np.all(rates[:, :2] > 3.5), (order, rates))
            self.assertTrue(np.all(rates[:, 2] > 5.5), (order, rates))

    def test_reflection_symmetry(self):
        x = np.linspace(-1., 1., 129)
        f = np.exp(.3*x) + np.sin(4*x+.2)
        for order in (1, 2):
            d = operator(x, order)
            np.testing.assert_allclose(d.grad(f[::-1]), (-1)**order*d.grad(f)[::-1],
                                       rtol=2e-10, atol=2e-10)

    def test_dirichlet_wave_has_no_growing_modes(self):
        # Q-only moment repair passed polynomial tests but failed this gate.
        for n in (32, 64):
            x = np.linspace(0., 1., n)
            eye = np.eye(n)
            DD = np.column_stack([operator(x, 2, CFDSolve.SCIPY).grad(e) for e in eye])[1:-1, 1:-1]
            ev = np.linalg.eigvals(DD)
            self.assertLessEqual(abs(ev.imag).max(), 1e-7)
            self.assertLess(ev.real.max(), 0.)
            m = n-2
            wave = np.block([[np.zeros((m,m)), np.eye(m)], [DD, np.zeros((m,m))]])
            eig = np.linalg.eigvals(wave)
            self.assertLessEqual(eig.real.max(), 1e-7)
            z = .1*(x[1]-x[0])*eig
            self.assertLessEqual(abs(1+z+z*z/2+z**3/6+z**4/24).max(), 1+1e-10)

    def test_sommerfeld_radial_wave_proxy(self):
        # Frozen inner three cells, radiative outer three cells, as in the
        # reduced-wave audit. This is not a general boundary-stability proof.
        n = 64
        h = 10/(n-7)
        r = 10+(np.arange(n)-3)*h
        eye = np.eye(n)
        D, DD = [np.column_stack([operator(r, order, CFDSolve.SCIPY).grad(e)
                                  for e in eye]) for order in (1, 2)]
        A = np.block([[np.zeros((n,n)), eye],
                      [DD+2/r[:,None]*D, np.zeros((n,n))]])
        S = -D-np.diag(1/r)
        for tail in (False, True):
            actual = A.copy()
            for field in (0, 1):
                anchor = n-4
                residual = A[field*n+anchor].copy()
                residual[field*n:(field+1)*n] -= S[anchor]
                for i in range(n-3, n):
                    actual[field*n+i] = 0.
                    actual[field*n+i, field*n:(field+1)*n] = S[i]
                    if tail:
                        actual[field*n+i] += (r[anchor]/r[i])**3 * residual
            keep = np.r_[np.arange(3,n), n+np.arange(3,n)]
            eig = np.linalg.eigvals(actual[np.ix_(keep,keep)])
            self.assertLessEqual(eig.real.max(), 1e-10)
            z = .1*h*eig
            self.assertLessEqual(abs(1+z+z*z/2+z**3/6+z**4/24).max(), 1+1e-10)



if __name__ == '__main__':
    unittest.main()
