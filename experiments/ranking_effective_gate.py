"""512-D ranking reference using the October report; existing methods unchanged."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.effective_gate import EffectiveGateConfig,fit_effective_gate,load_retained_prox
from baseline_suite import metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--limit',type=int,default=12)
    p.add_argument('--tasks',nargs='+',choices=['xor','ring','ring+xor','ring+xor+sum'],default=['xor','ring'])
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=True)
    config=EffectiveGateConfig()
    manifest=dict(config=asdict(config),dimensions=512,tasks=args.tasks,seed=0,device='cpu',
        changes='Only new reference; existing LassoNet and raw-gate Taylor-Fisher methods unchanged',
        source_hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ['src/effective_gate.py','experiments/ranking_effective_gate.py']},
        square='Shared parent gate; square of train-standardized input, train mean centered only',
        preprocessing='Original train fold only; same saved 512-D cases as existing comparisons',
        budget='120 epochs, no warmup/WD/dropout, constant C/rho/lr, no search',
        source_report='https://my.feishu.cn/docx/AXzzdP73Go5VhUxIdw7coClanEb',
        source_math='https://dreamsoul.feishu.cn/docx/Bxa7dPa32oF4hvxwk1Pcnm3Pnvf',
        torch=torch.__version__,numpy=np.__version__)
    path=args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('Existing sources/configuration differ; use a fresh output')
    path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    result=args.output/'results.jsonl'
    done={(r['task'],r['fold']) for r in map(json.loads,result.read_text().splitlines())} if result.exists() else set()
    _,topk=load_retained_prox()
    cases=[(task,fold) for task in args.tasks for fold in range(6)]
    for task,fold in cases[args.start:args.start+args.limit]:
        if (task,fold) in done:
            continue
        file=args.cases/f'{task}-fold{fold}.npz'
        data=np.load(file)
        assert data['x'].shape[1]==512
        x=data['x'][:,data['permutation']][data['train']]
        start=time.perf_counter()
        score,state=fit_effective_gate(x,data['y'][data['train']],config=config)
        selected,selection=topk(score,len(data['support']))
        checkpoint=args.output/f'{task}-fold{fold}.pt'
        torch.save(state,checkpoint)
        row=dict(task=task,fold=fold,method='effective_gate_oct9_reference',status='ok',scores=score.tolist(),
            metrics=metrics(score,data['support'],expected_dim=512),selection=selection,selected=selected.tolist(),
            dimensions=512,data_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            checkpoint=checkpoint.name,seconds=time.perf_counter()-start)
        with result.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        print('case',task,fold,row['metrics'],flush=True)


if __name__=='__main__':
    main()
