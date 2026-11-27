"""Memory-bounded Woodbury solve for the author's Localized Lasso IRLS updates.

The author's D is block diagonal by feature. Factor each sample-sized block,
then solve (I + A D^-1 A.T) v = y by CG without materializing D^-1 A.T.
IRLS iterations, epsilon constants, objective and initialization are unchanged.
Each linear solve checks its residual; no convergence failure is accepted.
"""
import torch
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, splu
from scipy.spatial.distance import cdist


def fit_localized(x, y, graph, num_iter=10, lam_net=1., lam_exc=.01):
    n,d=x.shape
    blocks=[sparse.eye(n,format='csc') for _ in range(d)]
    diagnostics=[]
    for iteration in range(num_iter):
        factors=[splu(block) for block in blocks]
        def matvec(v):
            result=v.copy()
            for j,factor in enumerate(factors):
                result+=x[:,j]*factor.solve(x[:,j]*v)
            return result
        operator=LinearOperator((n,n),matvec=matvec,dtype=np.float64)
        v,info=cg(operator,y,rtol=1e-13,atol=1e-15,maxiter=5000)
        residual=np.linalg.norm(matvec(v)-y)/max(np.linalg.norm(y),1e-30)
        if info or residual>1e-11:
            raise RuntimeError(f'Woodbury solve failed: {info=}, {residual=}')
        w=np.column_stack([factor.solve(x[:,j]*v) for j,factor in enumerate(factors)])
        distances=cdist(w,w)
        weights=.5/(distances+1e-9)*graph
        laplacian=np.diag(weights.sum(0))+np.diag(weights.sum(1))-2*weights
        laplacian=(laplacian+laplacian.T)*.5+.001*np.eye(n)
        network=sparse.csc_matrix(laplacian)
        exclusive=np.abs(w).sum(1)[:,None]/(np.abs(w)+1e-9)
        blocks=[lam_net*network+lam_exc*sparse.diags(exclusive[:,j],format='csc') for j in range(d)]
        objective=float(np.square(y-(x*w).sum(1)).sum()+lam_net*(distances*graph).sum()+lam_exc*np.square(np.abs(w).sum(1)).sum())
        diagnostics.append(dict(iteration=iteration,objective=objective,relative_residual=residual))
        print('IRLS',iteration,objective,residual,flush=True)
    return w.T,diagnostics
