import unittest
import torch
import numpy as np
from scipy.spatial.distance import cdist

from experiments.localized_lasso_solver import fit_localized


class LocalizedSolverTests(unittest.TestCase):
    def test_block_woodbury_matches_independent_dense_normal_equations(self):
        rng=np.random.default_rng(42)
        n,d=12,4
        x=rng.normal(size=(n,d))
        y=rng.normal(size=n)
        graph=(np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])==1).astype(float)
        a=np.zeros((n,n*d))
        for i in range(n):
            a[i,np.arange(d)*n+i]=x[i]
        penalty=np.eye(n*d)
        for _ in range(4):
            vector=np.linalg.solve(a.T@a+penalty,a.T@y)
            w=vector.reshape((n,d),order='F')
            distances=cdist(w,w)
            edge_weights=.5/(distances+1e-9)*graph
            laplacian=np.diag(edge_weights.sum(0))+np.diag(edge_weights.sum(1))-2*edge_weights
            laplacian=(laplacian+laplacian.T)*.5+.001*np.eye(n)
            exclusive=(np.abs(w).sum(1)[:,None]/(np.abs(w)+1e-9)).flatten(order='F')
            penalty=np.kron(np.eye(d),laplacian)+.01*np.diag(exclusive)
        actual,diagnostics=fit_localized(x,y,graph,num_iter=4)
        np.testing.assert_allclose(actual,w.T,rtol=1e-8,atol=1e-10)
        self.assertTrue(all(row['relative_residual']<1e-11 for row in diagnostics))


if __name__=='__main__':
    unittest.main()
