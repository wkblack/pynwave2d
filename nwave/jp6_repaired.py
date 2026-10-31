"""Cached structured LU application for the frozen JP6_REPAIRED pair.

No symbolic generation, dense N-by-N setup, coefficient truncation or fastmath.
The public derivative object retains candidate P/Q and truthful LUSOLVE metadata.
"""
import numpy as np
from scipy.linalg import solve_banded, solve, lu_factor, lu_solve
from numba import njit
from .bandedLUSolve import lu_banded, lu_solve_banded

@njit(cache=True)
def apply_fused(g, f, W, dp, dq, columns, denom):
    rank = W.shape[1]
    correction = np.empty(rank, dtype=np.float64)
    for a in range(rank):
        qsum = 0.0
        psum = 0.0
        for j in range(len(columns)):
            column = columns[j]
            qsum += dq[a, j] * f[column]
            psum += dp[a, j] * g[column]
        correction[a] = denom * qsum - psum
    for i in range(len(g)):
        change = 0.0
        for a in range(rank):
            change += W[i, a] * correction[a]
        g[i] += change
    return g

def _changed_rows(a, a_bands, b, b_bands, n):
    al, au = a_bands
    bl, bu = b_bands
    changed = np.zeros(n, dtype=bool)
    for offset in range(-max(au, bu), max(al, bl) + 1):
        lo = max(0, -offset)
        hi = min(n, n - offset)
        if lo >= hi:
            continue
        av = a[au + offset, lo:hi] if -au <= offset <= al else 0.0
        bv = b[bu + offset, lo:hi] if -bu <= offset <= bl else 0.0
        indices = np.flatnonzero(av != bv)
        changed[lo + indices + offset] = True
    return changed


def _row(a, bands, i, n):
    lower, upper = bands
    columns = np.arange(max(0, i - lower), min(n, i + upper + 1))
    result = np.zeros(n)
    result[columns] = a[upper + i - columns, columns]
    return result


class BoundaryWoodbury:
    def __init__(self, original, candidate):
        if original.N != candidate.N or original.denom != candidate.denom:
            raise ValueError('Boundary update requires matching grids and derivative order')
        self.original = original
        n = original.N
        changed = _changed_rows(original.P, original.Pbands,
                                candidate.P, candidate.Pbands, n)
        changed |= _changed_rows(original.Qb, original.Qbands,
                                 candidate.Qb, candidate.Qbands, n)
        rows = np.flatnonzero(changed)
        self.rows = rows
        self.rank = len(rows)
        if not self.rank:
            self.columns = np.empty(0, dtype=np.int64)
            self.dp = np.empty((0, 0))
            self.dq = np.empty((0, 0))
            self.W = np.empty((n, 0))
            self.schur_condition = 1.0
            return
        dp = np.empty((self.rank, n))
        dq = np.empty((self.rank, n))
        for a, i in enumerate(rows):
            dp[a] = (_row(candidate.P, candidate.Pbands, i, n)
                     - _row(original.P, original.Pbands, i, n))
            dq[a] = (_row(candidate.Qb, candidate.Qbands, i, n)
                     - _row(original.Qb, original.Qbands, i, n))
        self.columns = np.flatnonzero(np.any(dp != 0, axis=0)
                                      | np.any(dq != 0, axis=0))
        self.dp = dp[:, self.columns].copy()
        self.dq = dq[:, self.columns].copy()
        unit_rows = np.zeros((n, self.rank))
        unit_rows[rows, np.arange(self.rank)] = 1.0
        v = solve_banded(original.Pbands, original.P, unit_rows)
        schur = np.eye(self.rank) + dp @ v
        self.W = solve(schur.T, v.T).T.copy()
        self.schur_condition = np.linalg.cond(schur)

    def grad(self, f):
        g = self.original.grad(f)
        return apply_fused(g, f, self.W, self.dp, self.dq, self.columns,
                           self.original.denom)

def row_from_banded(ab,bands,row):
    lo,up=bands;n=ab.shape[1];out=np.zeros(n)
    for j in range(max(0,row-lo),min(n,row+up+1)):
        out[j]=ab[up+row-j,j]
    return out


def compact_rows(matrix):
    nz=[np.flatnonzero(row) for row in matrix]
    width=max(map(len,nz),default=0)
    indices=np.zeros((len(nz),width),dtype=np.int64)
    values=np.zeros((len(nz),width));counts=np.array(list(map(len,nz)),dtype=np.int64)
    for k,ids in enumerate(nz):indices[k,:len(ids)]=ids;values[k,:len(ids)]=matrix[k,ids]
    return indices,values,counts


@njit(cache=True)
def _rhs(f,denom,interior_coeff,boundaries,indices,values,counts):
    n=len(f);rhs=np.empty(n)
    for row in range(2,n-2):
        value=0.
        for k in range(3):value+=interior_coeff[row-2,k]*f[row-1+k]*denom
        rhs[row]=value
    for k in range(4):
        value=0.
        for j in range(counts[k]):value+=values[k,j]*f[indices[k,j]]*denom
        rhs[boundaries[k]]=value
    return rhs


@njit(cache=True)
def _boundary_product(vector,indices,values,counts):
    out=np.zeros(4)
    for k in range(4):
        for j in range(counts[k]):out[k]+=values[k,j]*vector[indices[k,j]]
    return out


@njit(cache=True)
def _assemble(z,Y,gb):
    n=len(z)+4;out=np.empty(n)
    out[0]=gb[0];out[1]=gb[1];out[n-2]=gb[2];out[n-1]=gb[3]
    for i in range(len(z)):
        value=z[i]
        for j in range(4):value-=Y[i,j]*gb[j]
        out[i+2]=value
    return out


@njit(cache=True)
def _finish_fused(z,rhs_boundary,Sinv,Y,indices,values,counts):
    residual=rhs_boundary.copy()
    for a in range(4):
        value=0.
        for j in range(counts[a]):value+=values[a,j]*z[indices[a,j]]
        residual[a]-=value
    gb=np.zeros(4)
    for a in range(4):
        for b in range(4):gb[a]+=Sinv[a,b]*residual[b]
    n=len(z)+4;out=np.empty(n)
    out[0]=gb[0];out[1]=gb[1];out[n-2]=gb[2];out[n-1]=gb[3]
    for i in range(len(z)):
        value=z[i]
        for a in range(4):value-=Y[i,a]*gb[a]
        out[i+2]=value
    return out


class BorderedDerivative:
    """Cache Pii LU, Y=Pii^-1 Pib, and the four-by-four Schur LU.

    Factorization uses generic band LU, not SPD-specific Cholesky. Masked
    operators are deliberately rejected: this implementation does not qualify their
    interior sparsity, coupling, or invertibility assumptions.
    """
    def __init__(self,operator):
        if operator.mask is not None:raise NotImplementedError('Masked operators not validated')
        self.N=operator.N;self.denom=float(operator.denom)
        if self.N<8:raise ValueError('At least eight allocated points required')
        self.boundaries=np.array([0,1,self.N-2,self.N-1],dtype=np.int64)
        ni=self.N-4;Pbb=np.zeros((4,4));Pbi=np.zeros((4,ni));Pib=np.zeros((ni,4));Pii=np.zeros((5,ni));Qb=np.zeros((4,self.N));Qi=np.zeros((ni,3))
        for k,row in enumerate(self.boundaries):
            p=row_from_banded(operator.P,operator.Pbands,int(row));Pbb[k]=p[self.boundaries];Pbi[k]=p[2:-2]
            Qb[k]=row_from_banded(operator.Qb,operator.Qbands,int(row))
        for row in range(2,self.N-2):
            pl,pu=operator.Pbands;ql,qu=operator.Qbands
            jp=np.arange(max(0,row-pl),min(self.N,row+pu+1))
            vp=operator.P[pu+row-jp,jp]
            nz=vp!=0;ji=jp[nz];vi=vp[nz]
            for j,v in zip(ji,vi):
                if 2<=j<self.N-2:
                    if abs(j-row)>2:raise ValueError('P interior is not pentadiagonal')
                    Pii[2+row-j,j-2]=v
                else:
                    k=0 if j==0 else 1 if j==1 else 2 if j==self.N-2 else 3
                    Pib[row-2,k]=v
            jq=np.arange(max(0,row-ql),min(self.N,row+qu+1))
            vq=operator.Qb[qu+row-jq,jq]
            if np.any(abs(jq[vq!=0]-row)>1):raise ValueError('Q interior is not three point')
            for k in range(3):
                j=row-1+k
                Qi[row-2,k]=operator.Qb[qu+row-j,j]
        self.Pii=Pii.copy();self.Pib=Pib;self.Pbi=Pbi;self.Pbb=Pbb;self.Qboundary=Qb;self.Qinterior=Qi
        self.interior_lu=lu_banded((2,2),Pii.copy(),overwrite_ab=True,check_finite=True)
        self.Y=lu_solve_banded(self.interior_lu,Pib.copy(),overwrite_b=False,check_finite=True)
        self.S=Pbb-Pbi@self.Y
        self.schur_condition=float(np.linalg.cond(self.S))
        if not np.isfinite(self.schur_condition):raise ValueError('Singular Schur complement')
        self.schur_lu=lu_factor(self.S,check_finite=True)
        self.schur_inverse=lu_solve(self.schur_lu,np.eye(4),check_finite=True)
        self.q_compact=compact_rows(Qb);self.pbi_compact=compact_rows(Pbi)

    def rhs(self,f):
        f=np.asarray(f)
        if f.shape!=(self.N,) or f.dtype!=np.dtype('float64'):raise ValueError('Expected one real float64 vector of exact length')
        if not np.isfinite(f).all():raise ValueError('Nonfinite input')
        return _rhs(f,self.denom,self.Qinterior,self.boundaries,*self.q_compact)

    def grad(self,f):
        return self.solve_rhs(self.rhs(f))

    def solve_rhs(self,rhs):
        rhs=np.asarray(rhs,dtype=float)
        if rhs.shape!=(self.N,) or not np.isfinite(rhs).all():raise ValueError("Invalid RHS")
        z=lu_solve_banded(self.interior_lu,rhs[2:-2].copy(),overwrite_b=True,check_finite=True)
        return _finish_fused(z,rhs[self.boundaries],self.schur_inverse,self.Y,*self.pbi_compact)
