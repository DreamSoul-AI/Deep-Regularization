"""Train the explicitly named numerical/binary LassoFlexNet reimplementation.

Primary importance: absolute skip coefficients at the best internal-validation
checkpoint on the positive-lambda path. No support labels enter model selection.
Author-unreported defaults are frozen here and recorded in every run manifest.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.lassoflexnet import FlexConfig,LassoFlexNet,PiecewiseLinearEncoding,SequentialEMA,IMPLEMENTATION_REVISION
from baseline_suite import metrics


def fit_flex(x,y,*,config,device='cpu',seed=1,pretrain_epochs=200,path_epochs=100,
             patience=20,batch_size=512,path_size=12,lambda_start=1e-7,
             lambda_multiplier=1e4,path_power=.5,lr=.001,weight_decay=1e-5,
             ema_decay=.99,mode='sequential_feasible'):
    if not set(np.unique(y)).issubset({0,1}):
        raise ValueError('This numerical reimplementation supports binary labels only')
    if min(pretrain_epochs,path_epochs,patience,path_size,batch_size)<1 or batch_size<2:
        raise ValueError('Positive training/path lengths and batch size >=2 are required')
    np.random.seed(seed)
    torch.manual_seed(seed)
    fit,valid=train_test_split(np.arange(len(x)),test_size=.2,random_state=seed,stratify=y)
    encoder=PiecewiseLinearEncoding.fit(x[fit],y[fit],bins=config.bins,seed=seed)
    model=LassoFlexNet(encoder,config).to(device)
    tx=torch.as_tensor(x,dtype=torch.float32,device=device)
    ty=torch.as_tensor(y,dtype=torch.float32,device=device)
    criterion=torch.nn.BCEWithLogitsLoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    history=[]
    prox=None

    def snapshot():
        return dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),
                    ema=copy.deepcopy(prox.state_dict()) if prox else None)

    def restore(state):
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if prox and state['ema'] is not None:
            prox.load_state_dict(state['ema'])

    def stage(penalty,epochs):
        best,best_state,stale=float('inf'),None,0
        max_violation=0.
        for epoch in range(epochs):
            model.train()
            for ids in torch.tensor(fit)[torch.randperm(len(fit))].split(batch_size):
                if len(ids)<2:
                    continue
                optimizer.zero_grad()
                loss=criterion(model(tx[ids]),ty[ids])
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Non-finite LassoFlexNet loss')
                loss.backward()
                optimizer.step()
                if penalty>0:
                    prox.step(optimizer,penalty)
                    violation=float(model.hierarchy_violation().detach())
                    max_violation=max(max_violation,violation)
                    if mode=='sequential_feasible' and violation>1e-6:
                        raise RuntimeError(f'Hierarchy violated after sequential prox: {violation}')
            model.eval()
            with torch.no_grad():
                val=float(criterion(model(tx[valid]),ty[valid]))
            if val<best-1e-8:
                best,best_state,stale=val,snapshot(),0
            else:
                stale+=1
            if stale>=patience:
                break
        restore(best_state)
        history.append(dict(penalty=float(penalty),epochs=epoch+1,validation_bce=best,
                            active=int(torch.count_nonzero(model.beta)),
                            max_update_hierarchy_violation=max_violation,
                            selected_hierarchy_violation=float(model.hierarchy_violation().detach())))
        print('stage',history[-1],flush=True)
        return best,best_state

    stage(0.,pretrain_epochs)
    warm_scores=model.beta.detach().abs().cpu().numpy().copy()
    prox=SequentialEMA(model,decay=ema_decay,mode=mode)
    end=lambda_start*lambda_multiplier
    penalties=np.linspace(lambda_start**(1/path_power),end**(1/path_power),path_size)**path_power
    best_loss,best_path,best_lambda=float('inf'),None,None
    for penalty in penalties:
        val,state=stage(float(penalty),path_epochs)
        if val<best_loss:
            best_loss,best_path,best_lambda=val,state,float(penalty)
        if torch.count_nonzero(model.beta)==0:
            break
    restore(best_path)
    scores=model.beta.detach().abs().cpu().numpy()
    checkpoint=dict(model=model.state_dict(),config=asdict(config),history=history,architecture_revision=IMPLEMENTATION_REVISION,
                    selected_lambda=best_lambda,selected_validation_bce=best_loss,
                    warm_scores=warm_scores.tolist(),mode=mode,
                    selected_hierarchy_violation=float(model.hierarchy_violation().detach()),
                    fit_indices=fit.tolist(),validation_indices=valid.tolist())
    return scores,checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--limit',type=int,default=24)
    p.add_argument('--pretrain-epochs',type=int,default=200)
    p.add_argument('--path-epochs',type=int,default=100)
    p.add_argument('--path-size',type=int,default=12)
    p.add_argument('--patience',type=int,default=20)
    p.add_argument('--mode',choices=['sequential_feasible','paper_literal'],default='sequential_feasible')
    p.add_argument('--tasks',nargs='+',choices=['xor','ring','ring+xor','ring+xor+sum'],
                   default=['xor','ring','ring+xor','ring+xor+sum'])
    args=p.parse_args()
    torch.set_num_threads(1)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')
    args.output.mkdir(parents=True,exist_ok=True)
    result=args.output/'results.jsonl'
    done=set()
    if result.exists():
        done={(r['task'],r['fold']) for r in map(json.loads,result.read_text(encoding='utf-8').splitlines())}
    config=FlexConfig()
    manifest=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  model=asdict(config),architecture_revision=IMPLEMENTATION_REVISION,
                  paper='https://proceedings.mlr.press/v306/lui26a.html',
                  label='local numerical/binary reimplementation; not author code',
                  seed=1,lr=.001,weight_decay=1e-5,batch_size=512,ema_decay=.99,
                  lambda_start=1e-7,lambda_multiplier=1e4,path_power=.5,
                  source_hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                                 ['src/lassoflexnet.py','experiments/ranking_lassoflexnet.py']})
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8'))!=manifest:
            raise ValueError('Existing run has different sources/configuration; use a new output directory')
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    cases=[(t,f) for t in args.tasks for f in range(6)]
    for task,fold in cases[args.start:args.start+args.limit]:
        if (task,fold) in done:
            continue
        path=args.cases/f'{task}-fold{fold}.npz'
        data=np.load(path)
        raw=data['x'][:,data['permutation']][data['train']]
        x=(raw-raw.mean(0))/(raw.std(0)+1e-8)
        start=time.perf_counter()
        score,checkpoint=fit_flex(x,data['y'][data['train']],config=config,device=args.device,
            pretrain_epochs=args.pretrain_epochs,path_epochs=args.path_epochs,path_size=args.path_size,
            patience=args.patience,mode=args.mode)
        torch.save(checkpoint,args.output/f'{task}-{fold}.pt')
        record=dict(task=task,fold=fold,method='lassoflexnet_reimpl_'+args.mode,status='ok',
                    dimensions=x.shape[1],seconds=time.perf_counter()-start,
                    data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scores=score.tolist(),
                    metrics=metrics(score,data['support'],expected_dim=x.shape[1]),
                    selected_lambda=checkpoint['selected_lambda'],
                    hierarchy_violation=checkpoint['selected_hierarchy_violation'])
        with result.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        print('case',task,fold,record['metrics'],flush=True)


if __name__=='__main__':
    main()
