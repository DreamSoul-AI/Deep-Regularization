"""Fixed public wrapper defaults at 512 dimensions, including rho_init=200."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from legacy_ranking_reference import load_reference,COMMIT
from baseline_suite import metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=True)
    namespace,source_hash=load_reference()
    manifest=dict(source_commit=COMMIT,source_sha256=source_hash,dimensions=512,
        training='Unmodified fixed Git-object model and _train_input_group function',
        wrapper_defaults=dict(lr=.005,C=.05,epochs=500,warmup_epochs=120,batch_size=64,
            rho_init=200.,hidden_size=32,hidden_layers=2,activation='mish',dropout=.04308691548552568,
            feat_drop=.6,weight_decay=.001),seed=0,device='cpu',
        source_hashes={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in
            ['experiments/ranking_public_defaults.py','experiments/legacy_ranking_reference.py']})
    path=args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise ValueError('Existing public-default run differs')
    path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    result=args.output/'results.jsonl'
    done={(r['task'],r['fold']) for r in map(json.loads,result.read_text().splitlines())} if result.exists() else set()
    for task in ['xor','ring','ring+xor','ring+xor+sum']:
        for fold in range(6):
            if (task,fold) in done:
                continue
            case=args.cases/f'{task}-fold{fold}.npz'
            data=np.load(case)
            x=data['x'][:,data['permutation']][data['train']]
            assert x.shape[1]==512
            mean,std=x.mean(0),x.std(0)+1e-8
            start=time.perf_counter()
            torch.manual_seed(0)
            model=namespace['GatedFeatureSelectionMLP'](512,2)
            namespace['_train_input_group'](model,(x-mean)/std,data['y'][data['train']],2,
                lr=.005,C=.05,epochs=500,warmup_epochs=120,batch_size=64,rho_init=200.,
                device='cpu',use_ratio_norm=True,use_admm=True,n_features=512)
            model.eval()
            score=model.gate.detach().abs()
            checkpoint=args.output/f'{task}-fold{fold}.pt'
            torch.save(dict(model=model.state_dict(),mean=torch.tensor(mean),std=torch.tensor(std)),checkpoint)
            row=dict(task=task,fold=fold,method='public_512_defaults_rho200',status='ok',scores=score.tolist(),
                metrics=metrics(score,data['support'],expected_dim=512),data_sha256=hashlib.sha256(case.read_bytes()).hexdigest(),
                seconds=time.perf_counter()-start,checkpoint=checkpoint.name)
            with result.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n')
            print(task,fold,row['metrics'],flush=True)


if __name__=='__main__':
    main()
