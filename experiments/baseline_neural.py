"""Frozen synthetic-ranking transfer of Deep Lasso benchmark neural selectors.

MLP/FT model definitions are loaded from the pinned upstream source. Training is
a local adapter: no Optuna search, reg_weight=.1, AGL gamma=2 and epsilon=1e-8.
These are comparison settings, not claimed paper-optimal hyperparameters.
"""
import copy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split
import baseline_supplement as suite

DEVICE = os.environ.get('RANKING_NEURAL_DEVICE', 'cpu')
if DEVICE == 'cuda':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def score_neural(method, x, y, k, assets, output, case):
    torch.manual_seed(1)
    np.random.seed(1)
    model_dir = assets/'upstream/deep_tabular/models'
    if method == 'attention_maps':
        cls = suite.load_module(model_dir/'ft_transformer_attention_map.py','upstream_ft').FTTransformerAttentionMap
        net = cls(128,1,None,192,True,3,8,1.3333333333,.2,.1,0.,'reglu',True,'kaiming',None,None)
    else:
        cls = suite.load_module(model_dir/'mlp.py','upstream_mlp').MLP
        net = cls(128,1,None,100,[100,100],.1)
    net = net.to(DEVICE)
    fit, valid = train_test_split(np.arange(len(x)),test_size=.2,random_state=1,stratify=y)
    tx, ty = torch.tensor(x,dtype=torch.float32,device=DEVICE), torch.tensor(y,dtype=torch.float32,device=DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    history = []

    def train(stage, adaptive=None):
        opt = torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=2e-4)
        best, best_state, stale = float('inf'),copy.deepcopy(net.state_dict()),0
        for epoch in range(200):
            net.train()
            lr = .001*min(1.,(epoch+1)/5)*(.1 if epoch>=40 else 1)*(.1 if epoch>=80 else 1)
            for group in opt.param_groups:
                group['lr'] = lr
            for ids in torch.tensor(fit)[torch.randperm(len(fit))].split(512):
                xx = tx[ids].clone().requires_grad_(method=='deep_lasso')
                loss = criterion(net(xx,None),ty[ids])
                if method == 'deep_lasso':
                    grad = torch.autograd.grad(loss,xx,create_graph=True)[0]
                    reg = (grad.square().sum(0)+1e-8).sqrt().sum()
                elif method in ('first_layer_lasso','adaptive_group_lasso'):
                    norms = (net.layers[0].weight.square().sum(0)+1e-8).sqrt()
                    reg = norms.sum() if adaptive is None else (norms*adaptive).sum()
                else:
                    reg = None
                objective = loss if reg is None else .9*loss+.1*reg
                opt.zero_grad()
                objective.backward()
                opt.step()
            if (epoch+1)%20 == 0:
                net.eval()
                with torch.no_grad():
                    val = criterion(net(tx[valid],None),ty[valid]).item()
                history.append(dict(stage=stage,epoch=epoch+1,validation_loss=val))
                print('progress',case,method,stage,epoch+1,val,flush=True)
                if val < best:
                    best,best_state,stale=val,copy.deepcopy(net.state_dict()),0
                else:
                    stale += 20
                if stale >= 20:
                    break
        net.load_state_dict(best_state)

    train('initial')
    if method == 'adaptive_group_lasso':
        weights = net.layers[0].weight.detach().square().sum(0).clamp_min(1e-8).reciprocal()
        train('adaptive',weights)
    torch.save(dict(state_dict=net.state_dict(),history=history),output/(case+'-'+method+'.pt'))
    net.eval()
    calibration = torch.tensor(np.linspace(0,len(x)-1,128).astype(int))
    if method == 'deep_lasso':
        xx = tx[calibration].clone().requires_grad_()
        grad = torch.autograd.grad(criterion(net(xx,None),ty[calibration]),xx)[0]
        return grad.detach().square().sum(0).sqrt().cpu().numpy()
    if method in ('first_layer_lasso','adaptive_group_lasso'):
        return net.layers[0].weight.detach().square().sum(0).sqrt().cpu().numpy()
    maps=[]
    def hook(module,args,result):
        maps.append(result[1]['attention_probs'].detach().sum(0))
    handles=[block['attention'].register_forward_hook(hook) for block in net.layers]
    with torch.no_grad():
        net(tx[calibration],None)
    for handle in handles:
        handle.remove()
    return (sum(maps)/(128*3*8))[0,1:].cpu().numpy()


if __name__ == '__main__':
    suite.score_model=score_neural
    suite.main()
