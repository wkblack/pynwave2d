"""Regressions for building compact-derivative matrices in banded storage.

Run: python -m unittest discover -s tests -p test_banded_construction.py -v

The reference is the previous dense pipeline: build each N x N matrix with
construct_banded_matrix_numba, blend BH operators row by row as dense
matrices, then convert with full_to_banded.  The banded construction must
reproduce it bit for bit without allocating any N x N array.
"""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nwave.cd as cd
from nwave import CFDSolve, DerivType, NCompactDerivative
from nwave.utils import (banded_to_full, construct_banded_matrix_numba,
                         construct_banded_storage_numba, full_to_banded)

# Every type build_derivative accepts, except D1_BYUP6 and D2_BYUP6, which
# fail upstream too (numba cannot reflect their nested-list boundary
# coefficients) and cannot be selected from a parameter file.
ALL_TYPES = [
    'D1_JP6_JT4', 'D2_JP6_JT4', 'D1_JT4', 'D2_JT4', 'D1_JT6', 'D2_JT6', 'D1_JP6', 'D2_JP6', 'D1_KP4', 'D1_Wm6', 'D1_DSQ6A', 'D2_DSQ6A', 'D1_DSQ6B', 'D2_DSQ6B',
    'D1_ME44', 'D2_ME44', 'D1_ME642', 'D2_ME642', 'D1_DSQ6B_LEFT',
    'D1_DSQ6B_RIGHT', 'D1_BYU_C6_Op10', 'D1_BYU_P4_Op18', 'D1_C6_4',
    'D1_C4_1', 'D1_A4_3', 'D2_2C4_1', 'D2_2C6_1', 'D2_2A_1',
]
# (interior, BH) pairs as used with BHMask.
BH_PAIRS = [('D1_JP6_JT4', 'D1_ME642'), ('D2_JP6_JT4', 'D2_ME642'), ('D1_JP6', 'D1_ME642'), ('D2_JP6', 'D2_ME642'), ('D1_JT6', 'D1_ME44'), ('D2_JT6', 'D2_ME642')]


def dense_storage(N, kl, ku, coeffs, bcoeffs, parity, klt, kut):
    """The previous path: dense matrix, then full_to_banded."""
    A = construct_banded_matrix_numba(N, kl, ku, coeffs, bcoeffs, parity)
    AB = full_to_banded(A, klt, kut)
    nlost = np.count_nonzero(A != banded_to_full(AB, klt, kut, N, N))
    return AB, nlost


def dense_build_derivative(N, dtype):
    with patch.object(cd, 'construct_banded_storage_numba', dense_storage):
        return cd.build_derivative(N, dtype)


def to_full(AB, bands, N):
    if bands == (0, 0):
        return np.diag(np.asarray(AB, dtype=np.float64).reshape(N))
    return banded_to_full(AB, bands[0], bands[1], N, N)


def dense_build_bh_derivative(N, dtype0, dtypeBH, mask):
    """The previous dense row blend of build_bh_derivative."""
    (P0, p0b), (Q0, q0b) = dense_build_derivative(N, dtype0)
    (PB, pbb), (QB, qbb) = dense_build_derivative(N, dtypeBH)
    m = mask[:, None]
    P = m * to_full(P0, p0b, N) + (1 - m) * to_full(PB, pbb, N)
    Q = m * to_full(Q0, q0b, N) + (1 - m) * to_full(QB, qbb, N)
    pbands = (max(p0b[0], pbb[0]), max(p0b[1], pbb[1]))
    qbands = (max(q0b[0], qbb[0]), max(q0b[1], qbb[1]))
    return (full_to_banded(P, *pbands), pbands), (full_to_banded(Q, *qbands), qbands)


class BandedConstruction(unittest.TestCase):
    def test_every_operator_matches_dense_bitwise(self):
        for N in (41, 97):
            for name in ALL_TYPES:
                with self.subTest(N=N, dtype=name):
                    (P, pb), (Q, qb) = cd.build_derivative(N, DerivType[name])
                    (Pd, pbd), (Qd, qbd) = dense_build_derivative(N, DerivType[name])
                    self.assertEqual((pb, qb), (pbd, qbd))
                    np.testing.assert_array_equal(P, Pd)
                    np.testing.assert_array_equal(Q, Qd)

    def test_bh_blend_matches_dense_bitwise(self):
        N = 97
        rng = np.random.default_rng(5)
        mask = np.r_[np.zeros(20), rng.uniform(size=N - 40), np.ones(20)]
        for d0, dbh in BH_PAIRS:
            with self.subTest(pair=(d0, dbh)):
                (P, pb), (Q, qb) = cd.build_bh_derivative(N, DerivType[d0], DerivType[dbh], mask)
                (Pd, pbd), (Qd, qbd) = dense_build_bh_derivative(
                    N, DerivType[d0], DerivType[dbh], mask)
                self.assertEqual((pb, qb), (pbd, qbd))
                np.testing.assert_array_equal(P, Pd)
                np.testing.assert_array_equal(Q, Qd)

    def test_entries_outside_bands_are_counted(self):
        (Pmat, Qmat) = cd._coeffs_D1_JTP6()
        kl, ku = Qmat[2]
        args = (61, kl, ku, Qmat[0], Qmat[1], Qmat[3])
        nqb = max(len(b) - 1 - i for i, b in enumerate(Qmat[1]))
        _, nlost = construct_banded_storage_numba(*args, max(nqb, kl), max(nqb, ku))
        self.assertEqual(nlost, 0)
        # Too-narrow bands: the count must match the dense reference exactly,
        # so lost entries at both boundaries are counted.
        _, nlost = construct_banded_storage_numba(*args, kl, ku)
        _, nlost_dense = dense_storage(*args, kl, ku)
        self.assertGreater(nlost_dense, 0)
        self.assertEqual(nlost, nlost_dense)

    def test_full_Q_is_built_lazily_from_banded(self):
        r = np.linspace(-10., 10., 61)
        op = NCompactDerivative.deriv(r, DerivType.D1_JP6, CFDSolve.LUSOLVE)
        self.assertIsNone(op._Q)
        Q = op.Q
        np.testing.assert_array_equal(Q, banded_to_full(op.Qb, *op.Qbands, 61, 61))
        self.assertIs(op.Q, Q)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'reads /proc/self/status')
    def test_bh_operator_build_allocates_no_dense_matrix(self):
        # Peak RSS during the build, measured from the current RSS just before
        # it, so an earlier higher peak cannot hide an allocation.  One dense
        # 2001 x 2001 matrix is 32 MB; the old dense blend held several.
        code = """import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from nwave import NCompactDerivative, DerivType, CFDSolve
def status_kb(key):
    for line in open('/proc/self/status'):
        if line.startswith(key):
            return int(line.split()[1])
def build(N):
    NCompactDerivative.bh_deriv(np.linspace(-100., 100., N), DerivType.D1_JP6,
                                DerivType.D1_ME642, CFDSolve.LUSOLVE, 0.0, 1.0)
build(101)
with open('/proc/self/clear_refs', 'w') as f:
    f.write('5')  # reset VmHWM to the current RSS
before = status_kb('VmRSS:')
build(2001)
print((status_kb('VmHWM:') - before) / 1024)
"""
        p = subprocess.run([sys.executable, '-c', code, str(ROOT)],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertLess(float(p.stdout.split()[-1]), 16.0)


if __name__ == '__main__':
    unittest.main()
