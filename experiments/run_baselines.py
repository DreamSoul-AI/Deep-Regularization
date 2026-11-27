"""Run/resume the complete fixed-setting ranking baseline comparison.

Create pinned sources with prepare_baseline_sources.py first. Dependencies are
installed explicitly by the operator; this command never installs system tools.
Each method writes its own directory, so workers can later be partitioned safely.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

LEGACY=['random','rf','mi','relief','attributions','stg','canceloutsigmoid','canceloutsoftmax',
        'deeppink','treeshap','xgboost','fsnet','lassonet','mrmr','mrmr_fullrank']
SUPPLEMENT=['univariate','lasso','e2efs','cae_keras']
NEURAL=['first_layer_lasso','adaptive_group_lasso','deep_lasso','attention_maps']
METHODS=LEGACY+SUPPLEMENT+NEURAL+['localized_lasso','lassoflexnet_reimpl','lassoflexnet']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    p.add_argument('--cae-python',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--tasks',nargs='+',choices=['xor','ring','ring+xor','ring+xor+sum'],
                   default=['xor','ring','ring+xor','ring+xor+sum'])
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    args.assets=args.assets.resolve()
    args.cases=args.cases.resolve()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    status={}
    for method in args.methods:
        if method=='lassoflexnet':
            status[method]=dict(status='blocked',reason='Official repository only has README; code coming soon, verified 2026-11-28')
        elif method=='cae_keras' and not args.cae_python:
            status[method]=dict(status='blocked',reason='Provide --cae-python for the separate concrete-autoencoder/tf-keras environment')
        else:
            directory=args.output/method
            directory.mkdir(parents=True,exist_ok=True)
            executable=str(args.cae_python.resolve()) if method=='cae_keras' else sys.executable
            env={**os.environ,'RANKING_NEURAL_DEVICE':args.device if method=='attention_maps' else 'cpu',
                 'LOKY_MAX_CPU_COUNT':'1','PYTHONIOENCODING':'utf-8'}
            if method in LEGACY:
                script='baseline_suite.py'
                source_args=['--stage',str(args.assets/'legacy')]
            elif method=='lassoflexnet_reimpl':
                script='ranking_lassoflexnet.py'
                source_args=[]
            else:
                script=('baseline_neural.py' if method in NEURAL else
                        'baseline_localized.py' if method=='localized_lasso' else 'baseline_supplement.py')
                source_args=['--assets',str(args.assets)]
            command=[executable,'-u',str(root/'experiments'/script),*source_args,
                     '--cases',str(args.cases),'--output',str(directory),'--tasks',*args.tasks]
            if method!='lassoflexnet_reimpl':
                command+=['--method',method]
            else:
                command+=['--device',args.device]
            if method=='lassonet':
                command+=['--device',args.device]
            with (directory/'worker.log').open('a',encoding='utf-8') as log:
                returncode=subprocess.call(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            result=directory/('results.jsonl' if method=='lassoflexnet_reimpl' else method+'.jsonl')
            rows=list(map(json.loads,result.read_text(encoding='utf-8').splitlines())) if result.exists() else []
            successful={(r['task'],r['fold'],r['method']) for r in rows if r['status']=='ok'}
            expected=6*len(args.tasks)*(10 if method=='attributions' else 1)
            status[method]=dict(status='complete' if len(successful)==expected else 'incomplete',
                completed=len(successful),expected=expected,returncode=returncode)
        (args.output/'run_status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
        print(method,status[method],flush=True)
    if any(r['status']!='complete' for r in status.values()):
        sys.exit(1)


if __name__=='__main__':
    main()
