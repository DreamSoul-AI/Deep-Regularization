"""Numerical-feature, scalar-output LassoFlexNet paper reimplementation.

Source: Lui et al., ICML 2026, sections 4--5 and appendix C. This is not the
unreleased author code. ``paper_literal`` preserves printed Algorithm 2;
``sequential_feasible`` implements the separate constrained problem in the text.
The two maps differ: the printed map can violate the stated hierarchy.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from sklearn.tree import DecisionTreeClassifier

IMPLEMENTATION_REVISION='figure8-feature-bottleneck-channel-residual-v2'


@dataclass(frozen=True)
class FlexConfig:
    bins: int = 8
    pfe_width: int = 8
    pfe_layers: int = 2
    embedding: int = 8
    mixer_hidden: int = 64
    mixer_blocks: int = 2
    dropout: float = .1
    tau: float = .1
    hierarchy: float = 10.


class PiecewiseLinearEncoding(nn.Module):
    """Univariate supervised tree knots; fit on training rows/labels only."""
    def __init__(self, left, right, valid):
        super().__init__()
        self.register_buffer('left', torch.as_tensor(left, dtype=torch.float32))
        self.register_buffer('right', torch.as_tensor(right, dtype=torch.float32))
        self.register_buffer('valid', torch.as_tensor(valid, dtype=torch.bool))

    @classmethod
    def fit(cls, x, y, bins=8, seed=1):
        if bins < 2 or x.ndim != 2 or len(x) != len(y):
            raise ValueError('Require a training matrix, aligned labels and bins >= 2')
        d = x.shape[1]
        left, right = np.zeros((d,bins)), np.ones((d,bins))
        valid = np.zeros((d,bins), dtype=bool)
        for j in range(d):
            tree = DecisionTreeClassifier(max_leaf_nodes=bins, random_state=seed)
            tree.fit(x[:,j:j+1], y)
            thresholds = tree.tree_.threshold[tree.tree_.feature >= 0]
            edges = np.unique(np.r_[x[:,j].min(), thresholds, x[:,j].max()])
            if len(edges) == 1:
                edges = np.r_[edges, edges+1.]
            count = len(edges)-1
            left[j,:count], right[j,:count] = edges[:-1], edges[1:]
            valid[j,:count] = True
        return cls(left, right, valid)

    def forward(self, x):
        encoded = ((x.unsqueeze(-1)-self.left)/(self.right-self.left).clamp_min(1e-12)).clamp(0,1)
        return encoded*self.valid


class FeatureLinear(nn.Module):
    def __init__(self, features, incoming, outgoing):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(features,incoming,outgoing))
        self.bias = nn.Parameter(torch.zeros(features,outgoing))
        nn.init.uniform_(self.weight, -incoming**-.5, incoming**-.5)

    def forward(self, x):
        return torch.einsum('bdi,dio->bdo', x, self.weight)+self.bias


class FeatureResidual(nn.Module):
    def __init__(self, features, width, dropout):
        super().__init__()
        self.linear = FeatureLinear(features,width,width)
        self.scale = nn.Parameter(torch.ones(features,width))
        self.bias = nn.Parameter(torch.zeros(features,width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = F.layer_norm(F.relu(self.linear(x)), (x.shape[-1],))
        return x+self.dropout(z*self.scale+self.bias)


class MixerBlock(nn.Module):
    def __init__(self, features, embedding, hidden, dropout, first=False):
        super().__init__()
        self.first = first
        self.token_norm = nn.Identity() if first else nn.LayerNorm(embedding)
        # Fig. 8 keeps normalization/residual in embedding mixing. Only the
        # initial feature-mixing sublayer loses its input-dependent bypass.
        self.channel_norm = nn.LayerNorm(embedding)
        self.token = nn.Sequential(nn.Linear(features,hidden),nn.GELU(),nn.Dropout(dropout),
                                   nn.Linear(hidden,features),nn.Dropout(dropout))
        self.channel = nn.Sequential(nn.Linear(embedding,hidden),nn.GELU(),nn.Dropout(dropout),
                                     nn.Linear(hidden,embedding),nn.Dropout(dropout))

    def forward(self, x):
        z = self.token(self.token_norm(x).transpose(1,2)).transpose(1,2)
        x = z if self.first else x+z
        z = self.channel(self.channel_norm(x))
        return x+z


class LassoFlexNet(nn.Module):
    def __init__(self, encoder, config=FlexConfig()):
        super().__init__()
        self.encoder, self.config = encoder, config
        d, bins = encoder.left.shape
        self.pfe = nn.Sequential(FeatureLinear(d,bins,config.pfe_width),
            *[FeatureResidual(d,config.pfe_width,config.dropout) for _ in range(config.pfe_layers)],
            FeatureLinear(d,config.pfe_width,config.embedding))
        self.standardize = nn.BatchNorm1d(d*config.embedding,affine=False)
        self.beta = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([MixerBlock(d,config.embedding,config.mixer_hidden,
                                      config.dropout,first=i==0) for i in range(config.mixer_blocks)])
        self.head = nn.Linear(config.embedding,1)
        self.intercept = nn.Parameter(torch.zeros(()))

    @property
    def constrained_weight(self):
        # The token linear map is shared across embedding channels: zeroing this
        # entire feature column blocks every coordinate of that feature's PFE.
        return self.blocks[0].token[0].weight

    def forward(self, x):
        z = self.pfe(self.encoder(x))
        z = self.standardize(z.flatten(1)).reshape_as(z)
        skip = z.mean(-1)@self.beta+self.intercept
        for block in self.blocks:
            z = block(z)
        return skip+self.config.tau*self.head(z.mean(1)).squeeze(-1)

    def hierarchy_violation(self):
        return (self.constrained_weight.abs().amax(0)-self.config.hierarchy*self.beta.abs()).clamp_min(0).amax()


def soft_threshold(value, threshold):
    return value.sign()*(value.abs()-threshold).clamp_min(0)


def sequential_prox(beta_ema, weight_ema, beta_threshold, weight_threshold, hierarchy):
    """Exact coordinatewise prox for the separate, bounded W problem in §5."""
    beta = soft_threshold(beta_ema,beta_threshold)
    weight = soft_threshold(weight_ema,weight_threshold)
    cap = hierarchy*beta.abs()
    return beta,weight.sign()*torch.minimum(weight.abs(),cap.unsqueeze(0))


def printed_algorithm2(beta, weight, beta_ema, weight_ema, threshold, hierarchy):
    """Literal scalar-output Algorithm 2, with a threshold per feature group.

    It intentionally does NOT append an unprinted repair clamp. Use diagnostics
    to expose hierarchy violations instead of calling this map always feasible.
    """
    shrunk = beta.sign()*soft_threshold(beta_ema,threshold).abs()
    ordered = weight_ema.abs().T.sort(dim=1,descending=True).values
    d,k = ordered.shape
    prefix = torch.cat([ordered.new_zeros(d,1),ordered.cumsum(1)],dim=1)
    m = torch.arange(k+1,device=ordered.device,dtype=ordered.dtype)
    candidate = hierarchy/(1+m*hierarchy**2)*(shrunk.abs()[:,None]+hierarchy*prefix-threshold[:,None]).clamp_min(0)
    upper = torch.cat([ordered.new_full((d,1),float('inf')),ordered],dim=1)
    lower = torch.cat([ordered,ordered.new_zeros(d,1)],dim=1)
    valid = (lower<=candidate)&(candidate<=upper)
    if not bool(valid.any(1).all()):
        raise RuntimeError('Printed Algorithm 2 has no valid sorted interval')
    selected = valid.to(torch.int64).argmax(1)
    cap = candidate[torch.arange(d,device=ordered.device),selected]
    return shrunk,weight.sign()*torch.minimum(weight_ema.abs(),cap[None,:])


class SequentialEMA:
    """EMA centers are updated after AdamW, before prox, only for β and W1.

    EMA decay/order and the W L1 coefficient are explicit local choices because
    the paper does not provide a complete executable configuration.
    """
    def __init__(self, model, decay=.99, mode='sequential_feasible'):
        if mode not in ('sequential_feasible','paper_literal'):
            raise ValueError(mode)
        self.model,self.decay,self.mode = model,decay,mode
        self.beta = model.beta.detach().clone()
        self.weight = model.constrained_weight.detach().clone()

    @staticmethod
    def adaptive_step(parameter, optimizer):
        group = next(g for g in optimizer.param_groups if any(p is parameter for p in g['params']))
        state = optimizer.state[parameter]
        corrected = state['exp_avg_sq']/(1-group['betas'][1]**float(state['step']))
        return group['lr']/(corrected.sqrt()+group['eps'])

    @torch.no_grad()
    def step(self, optimizer, penalty):
        beta,weight = self.model.beta,self.model.constrained_weight
        self.beta.lerp_(beta,1-self.decay)
        self.weight.lerp_(weight,1-self.decay)
        threshold = penalty*self.adaptive_step(beta,optimizer)
        if self.mode == 'paper_literal':
            b,w = printed_algorithm2(beta,weight,self.beta,self.weight,threshold,self.model.config.hierarchy)
        else:
            b,w = sequential_prox(self.beta,self.weight,threshold,
                                  penalty*self.adaptive_step(weight,optimizer),self.model.config.hierarchy)
        beta.copy_(b)
        weight.copy_(w)

    def state_dict(self):
        return dict(beta=self.beta.clone(),weight=self.weight.clone())

    def load_state_dict(self, state):
        self.beta.copy_(state['beta'])
        self.weight.copy_(state['weight'])
