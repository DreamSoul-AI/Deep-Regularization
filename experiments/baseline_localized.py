"""Localized Lasso ranking adapter using verified block Woodbury linear solves."""
import json
from localized_lasso_solver import fit_localized
import baseline_supplement as suite
import numpy as np
from sklearn.neighbors import kneighbors_graph


def score_localized(method,x,y,k,assets,output,case):
    if method!='localized_lasso':
        raise ValueError(method)
    graph=kneighbors_graph(x,5,mode='connectivity',include_self=False)
    graph=((graph+graph.T)*.5).toarray()
    weights,diagnostics=fit_localized(x,y.astype(float),graph)
    np.save(output/(case+'-localized-weights.npy'),weights)
    (output/(case+'-linear-solves.json')).write_text(json.dumps(diagnostics,indent=2))
    return np.linalg.norm(weights,axis=1)


if __name__=='__main__':
    suite.score_model=score_localized
    suite.main()
