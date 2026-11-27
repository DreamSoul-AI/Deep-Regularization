"""Report-defined shared linear/square gate; no changes to existing methods."""
import ast
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


@dataclass(frozen=True)
class EffectiveGateConfig:
    epochs: int = 120
    batch_size: int = 64
    hidden_size: int = 32
    penalty: float = .003172
    rho: float = .18857
    learning_rate: float = .002084


def load_retained_prox():
    """Reuse retained numerical kernel functions verbatim; verify source bytes."""
    directory=Path(__file__).resolve().parents[1]/'docs/research/20260920-xor-ring-thread/sources'
    sources=[('2f836a8e083b6831da0a.py','2f836a8e083b6831da0a8db7a80681e2e2f7c0b1f19d0d3f08569a1503bf428b',
              {'deterministic_topk','_ratio_norm_value_numpy','_ratio_prox_objective_numpy','_ratio_prox_kkt_residual_numpy'}),
             ('6ed77fc604592abeabf3.py','6ed77fc604592abeabf3ab3087ff937a04911c8f87a023279bd63c1c3615b0a6',
              {'canonical_signed_ratio_prox'})]
    namespace=dict(torch=torch,np=np,Any=Any)
    for name,expected,wanted in sources:
        raw=(directory/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:
            raise RuntimeError('Retained solver source changed: '+name)
        nodes=[n for n in ast.parse(raw.decode('utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
        if {n.name for n in nodes}!=wanted:
            raise RuntimeError('Retained solver definitions are incomplete')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),name,'exec',dont_inherit=True),namespace)
    return namespace['canonical_signed_ratio_prox'],namespace['deterministic_topk']


class EffectiveGateMLP(nn.Module):
    """g_j=s_j*sqrt(||U_j||²+||V_j||²) controls both raw-feature channels.

    Ordinary raw U initialization, V=0, s=1. No normalized forward or new
    parameterization: h=(x*s)U.T + ((x²-square_mean)*s)V.T+b.
    """
    def __init__(self, features, square_mean, hidden_size=32):
        super().__init__()
        self.gate=nn.Parameter(torch.ones(features))
        self.linear=nn.Linear(features,hidden_size)
        self.square_weight=nn.Parameter(torch.zeros(hidden_size,features))
        self.layers=nn.Sequential(nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,1))
        self.register_buffer('square_mean',torch.as_tensor(square_mean,dtype=torch.float32))

    def effective_gate(self):
        return self.gate*torch.cat([self.linear.weight,self.square_weight],dim=0).norm(dim=0)

    def forward(self,x):
        h=self.linear(x*self.gate)
        h=h+nn.functional.linear((x.square()-self.square_mean)*self.gate,self.square_weight)
        return self.layers(h)


def fit_effective_gate(inputs,targets,*,seed=0,config=None):
    config=config or EffectiveGateConfig()
    torch.manual_seed(seed)
    raw=np.asarray(inputs)
    mean,std=raw.mean(0),raw.std(0)+1e-8
    x=torch.tensor((raw-mean)/std,dtype=torch.float32)
    y=torch.tensor(targets,dtype=torch.float32)
    model=EffectiveGateMLP(x.shape[1],x.square().mean(0),config.hidden_size)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate,weight_decay=0.)
    loader=DataLoader(TensorDataset(x,y),batch_size=config.batch_size,shuffle=True,num_workers=0)
    prox,_=load_retained_prox()
    z=model.effective_gate().detach().clone()
    dual=torch.zeros_like(z)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    trace=[]
    for epoch in range(config.epochs):
        total=0.
        model.train()
        for xx,yy in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(model(xx).flatten(),yy)
            objective=loss+.5*config.rho*(model.effective_gate()-z+dual).square().sum()
            if not torch.isfinite(objective):
                raise FloatingPointError('Nonfinite effective-gate objective')
            objective.backward()
            optimizer.step()
            total+=float(loss.detach())*len(xx)
        with torch.no_grad():
            g=model.effective_gate()
            previous=z.clone()
            z,info=prox(g+dual,config.penalty/config.rho)
            dual=dual+g-z
            trace.append(dict(epoch=epoch,task_loss=total/len(x),primal=float((g-z).norm()),
                dual_residual=float(config.rho*(z-previous).norm()),prox=info))
        if epoch%20==19:
            print('epoch',epoch+1,trace[-1]['task_loss'],trace[-1]['primal'],flush=True)
    model.eval()
    scores=model.effective_gate().detach().abs()
    state=dict(model=model.state_dict(),initial=initial,z=z,dual=dual,mean=torch.tensor(mean),std=torch.tensor(std),
               config=asdict(config),seed=seed,trace=trace,optimizer=optimizer.state_dict(),
               scope='report reconstruction; fixed trial8 rounded parameters; no exact October source recovered')
    return scores,state
