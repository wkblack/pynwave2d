"""Opt-in true-sixth boundary pair: exact fixtures and actual factory paths."""
from fractions import Fraction
from pathlib import Path
import json
import math
import sys
import unittest
from unittest.mock import patch
import numpy as np
from scipy.linalg import solve_banded
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"bssn"))
from nwave import cd
from nwave.types import CFDSolve, DerivType, d1_type_map, d2_type_map
from nwave.jp6_repaired_data import BOUNDARIES

TYPES = (DerivType.D1_JP6_REPAIRED, DerivType.D2_JP6_REPAIRED)


class RepairedTests(unittest.TestCase):
    def make(self, order, n=80):
        return cd.NCompactDerivative.deriv(np.linspace(-1., 1., n), TYPES[order-1], CFDSolve.LUSOLVE)

    def test_exact_moments_and_literal_data(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/jp6_repaired_exact.json').read_text())
        for order in (1, 2):
            raw = fixture[f'D{order}']
            ps = [[Fraction(x) for x in row] for row in raw['P_left']]
            qs = [[Fraction(x) for x in row] for row in raw['Q_left']]
            for p, q in zip(ps, qs):
                for degree in range(order + 6):
                    rhs = sum(v * (math.factorial(degree)//math.factorial(degree-order)) * j**(degree-order)
                              for j, v in enumerate(p)) if degree >= order else Fraction(0)
                    self.assertEqual(sum(v*j**degree for j,v in enumerate(q)), rhs)
            pp = [np.trim_zeros(np.array([float(v) for v in row]), 'b') for row in ps]
            qq = [np.trim_zeros(np.array([float(v) for v in row]), 'b') for row in qs]
            if order == 2:
                original, _ = cd._coeffs_D2_JTP6()
                norm = np.linalg.norm(original[0])
                for p,q in zip(pp,qq):
                    scale = norm/np.linalg.norm(p); p *= scale; q *= scale
            for key, rows in [('P',pp),('Q',qq)]:
                for actual, expected in zip(BOUNDARIES[order][key], rows):
                    np.testing.assert_array_equal(actual, expected)

    def test_all_allocated_polynomials_and_reflection(self):
        for order in (1,2):
            for n in (56,80,160):
                x = np.linspace(-1.,1.,n); op = self.make(order,n)
                for degree in range(order+6):
                    exact = np.zeros(n) if degree < order else math.factorial(degree)/math.factorial(degree-order)*x**(degree-order)
                    np.testing.assert_allclose(op.grad(x**degree), exact, rtol=0, atol=4e-8)
                f = np.sin(3*x) + .1*np.cos(7*x)
                np.testing.assert_allclose(op.grad(f[::-1]), (-1)**order*op.grad(f)[::-1], rtol=1e-8,atol=2e-8)

    def test_banded_reference_and_residual(self):
        rng = np.random.default_rng(20261031)
        for order in (1,2):
            for n in (64,128,800,1600):
                op = self.make(order,n)
                for f in (rng.normal(size=n), np.sin(np.linspace(-2,2,n)), np.ones(n)+1e-10*rng.normal(size=n)):
                    g = op.grad(f)
                    # Independent banded Q multiplication, not the application kernel.
                    rhs = np.zeros(n); ql,qu = op.Qbands
                    for j in range(n):
                        for i in range(max(0,j-qu),min(n,j+ql+1)):
                            rhs[i] += op.Qb[qu+i-j,j]*f[j]*op.denom
                    reference = solve_banded(op.Pbands,op.P,rhs)
                    scale = max(1.,np.max(np.abs(reference)))
                    self.assertLess(np.max(np.abs(g-reference))/scale, 3e-7)
                    residual = -rhs.copy(); pl,pu = op.Pbands
                    for j in range(n):
                        for i in range(max(0,j-pu),min(n,j+pl+1)):
                            residual[i] += op.P[pu+i-j,j]*g[j]
                    self.assertLess(np.max(np.abs(residual))/max(1.,np.max(np.abs(rhs))), 2e-10)
                self.assertIsNone(op._Q)
                self.assertIsNone(op.lu_factorization)  # no wasted wide LU
                self.assertEqual(op.get_method(), CFDSolve.LUSOLVE)
                self.assertIn('pentadiagonal-lu',op.solver_kind)

    def test_uniform_refinement(self):
        for order in (1,2):
            errors=[]
            # The 56-point closure needs resolved wavelengths before its
            # asymptotic rate appears; coarse wide-stencil errors oscillate.
            for n in (160,320,640):
                x=np.linspace(-1,1,n); op=self.make(order,n)
                exact=7*np.cos(7*x) if order==1 else -49*np.sin(7*x)
                errors.append(np.max(np.abs(op.grad(np.sin(7*x))-exact)))
            self.assertGreater(np.log2(errors[-2]/errors[-1]),5.0)

    def test_selection_and_rejections(self):
        self.assertEqual(d1_type_map['JP6_REPAIRED'], TYPES[0])
        self.assertEqual(d2_type_map['JP6_REPAIRED'], TYPES[1])
        self.assertEqual(d1_type_map['JP6'], DerivType.D1_JP6)
        self.assertEqual([t.value for t in TYPES],[42,43])
        for typ in TYPES:
            with self.assertRaisesRegex(ValueError,'56 allocated'):
                cd.NCompactDerivative.deriv(np.arange(55.),typ,CFDSolve.LUSOLVE)
            with self.assertRaisesRegex(ValueError,'56 allocated'):
                cd.build_derivative(55,typ)
            for method in CFDSolve:
                if method in (CFDSolve.LUSOLVE, CFDSolve.SCIPY): continue
                with self.assertRaisesRegex(ValueError,'only structured LUSOLVE'):
                    cd.NCompactDerivative.deriv(np.arange(56.),typ,method)
            for pair in ((typ,DerivType.D1_JP6),(DerivType.D1_JP6,typ)):
                with self.assertRaisesRegex(ValueError,'BH masks'):
                    cd.NCompactDerivative.bh_deriv(np.arange(80.),*pair,CFDSolve.LUSOLVE,1.,.1)
                with self.assertRaisesRegex(ValueError,'BH masks'):
                    cd.build_bh_derivative(80,*pair,np.ones(80))
            with self.assertRaisesRegex(ValueError,'uniform'):
                cd.NCompactDerivative.deriv(np.arange(80.)**2,typ,CFDSolve.LUSOLVE)
        op=self.make(1)
        for f in (np.ones(80,dtype=np.float32),np.ones((80,1)),np.full(80,np.nan)):
            with self.assertRaises(ValueError):op.grad(f)

    def test_scipy_reference_backend(self):
        rng = np.random.default_rng(20261031)
        for order in (1, 2):
            for n in (56, 64, 128, 800, 1600):
                x = np.linspace(-1., 1., n)
                fast = self.make(order, n)
                reference = cd.NCompactDerivative.deriv(x, TYPES[order-1], CFDSolve.SCIPY)
                self.assertEqual(reference.get_method(), CFDSolve.SCIPY)
                self.assertEqual(reference.solver_kind, 'scipy-wide-banded-reference')
                self.assertFalse(hasattr(reference, 'structured_solver'))
                self.assertIsNone(reference.lu_factorization)
                np.testing.assert_array_equal(fast.P, reference.P)
                np.testing.assert_array_equal(fast.Qb, reference.Qb)
                for f in (np.sin(7*x), rng.normal(size=n), np.ones(n), 1+1e-10*rng.normal(size=n)):
                    got, expected = fast.grad(f), reference.grad(f)
                    scale = max(1., np.max(np.abs(expected)))
                    self.assertLess(np.max(np.abs(got-expected))/scale, 3e-7)
                self.assertIsNone(reference._Q)
        import bssnutils
        operators = bssnutils.init_derivative_operators(np.linspace(-1.,1.,80),
            {"D1":"JP6_REPAIRED", "D2":"JP6_REPAIRED", "DerivSolveMethod":"SCIPY", "BHMask":False})
        self.assertTrue(all(op.method == CFDSolve.SCIPY for op in operators))

    def test_actual_bssn_configuration_dispatch(self):
        import bssnutils
        x = np.linspace(-1., 1., 80)
        params = {"D1": "JP6_REPAIRED", "D2": "JP6_REPAIRED",
                  "DerivSolveMethod": "LUSOLVE", "BHMask": False}
        operators = bssnutils.init_derivative_operators(x, params)
        for order, op in enumerate(operators, 1):
            self.assertEqual(op.dtype, TYPES[order-1])
            np.testing.assert_array_equal(op.grad(np.sin(x)), self.make(order).grad(np.sin(x)))
        params.update(BHMask=True, D1=["JP6", "JP6_REPAIRED"], D2=["JP6", "JP6_REPAIRED"])
        with self.assertRaisesRegex(ValueError, 'BH masks'):
            bssnutils.init_derivative_operators(x, params)

    def test_fast_path_never_materializes_full_matrices(self):
        with patch.object(cd, 'banded_to_full', side_effect=AssertionError('dense NxN')):
            for order in (1,2):
                op=self.make(order,1600); self.assertIsNone(op._Q)
                self.assertTrue(np.isfinite(op.grad(np.ones(1600))).all())


if __name__ == '__main__':
    unittest.main()
