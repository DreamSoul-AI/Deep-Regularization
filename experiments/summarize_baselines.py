"""Recompute and summarize paired ranking results without counting failed retries."""
import argparse
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import sys

import torch
import numpy as np
from baseline_suite import metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ours',type=Path)
    args=p.parse_args()
    records={}
    failures=defaultdict(list)
    streams=[list(map(json.loads,file.read_text().splitlines())) for file in sorted(args.results.rglob('*.jsonl'))]
    if args.ours:
        streams.append([dict(task=r['task'],fold=r['fold'],method='ours_'+r['mode'],
                        status='ok',metrics=r['metrics']['gate'],scores=r['scores']['gate'])
                        for r in map(json.loads,args.ours.read_text().splitlines())])
    for stream in streams:
        for row in stream:
            key=(row['method'],row['task'],row['fold'])
            if row['status']=='ok':
                data=np.load(args.cases/f"{row['task']}-fold{row['fold']}.npz")
                actual=metrics(row['scores'],data['support'])
                for metric in ['recall_at_k','average_precision','auroc']:
                    if not np.isclose(actual[metric],row['metrics'][metric],atol=1e-12,rtol=0):
                        raise ValueError((key,metric,'metric mismatch'))
                if actual['top_k'] != row['metrics']['top_k']:
                    raise ValueError((key,'selection mismatch'))
                if key in records and records[key]['scores'] != row['scores']:
                    raise ValueError((key,'conflicting successful retries'))
                records[key]=row
            else:
                failures[row['method']].append(dict(task=row['task'],fold=row['fold'],error=row['error']))
    tasks=['xor','ring','ring+xor','ring+xor+sum']
    methods=sorted(set(key[0] for key in records)|set(failures))
    summary={}
    for method in methods:
        rows=[r for key,r in records.items() if key[0]==method]
        complete=len(rows)==24
        task_metrics={}
        for task in tasks:
            rr=[r for r in rows if r['task']==task]
            task_metrics[task]=dict(n=len(rr),**{
                metric:float(np.mean([r['metrics'][metric] for r in rr])) if rr else None
                for metric in ['recall_at_k','average_precision','auroc']})
        summary[method]=dict(status='complete' if complete else 'incomplete',completed=len(rows),
            failed_attempts=len(failures[method]),tasks=task_metrics,
            constant_score_cases=sum(np.unique(r['scores']).size==1 for r in rows),
            min_distinct_scores=min((np.unique(r['scores']).size for r in rows),default=None),
            macro_recall=float(np.mean([task_metrics[t]['recall_at_k'] for t in tasks])) if complete else None,
            macro_ap=float(np.mean([task_metrics[t]['average_precision'] for t in tasks])) if complete else None,
            macro_auroc=float(np.mean([task_metrics[t]['auroc'] for t in tasks])) if complete else None)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (args.output/'verified_results.jsonl').open('w') as f:
        for key in sorted(records):
            f.write(json.dumps(records[key])+'\n')
    (args.output/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    table=['| Method | XOR | Ring | Ring+XOR | Ring+XOR+Sum | Macro | AP | Status |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    def fmt(value):
        if value is None:
            return '—'
        # Remove float noise at exact half-cent boundaries, e.g. 71.875%.
        return str(Decimal(f'{100*value:.10f}').quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
    for method in sorted(methods,key=lambda m:-(summary[m]['macro_recall'] or -1)):
        s=summary[method]
        vals=[fmt(s['tasks'][t]['recall_at_k']) if s['tasks'][t]['n']==6 else f"{s['tasks'][t]['n']}/6" for t in tasks]
        table.append('| '+' | '.join([method,*vals,fmt(s['macro_recall']),fmt(s['macro_ap']),f"{s['completed']}/24"])+ ' |')
    (args.output/'table.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    print(f'Verified {len(records)} successful cases; {sum(s["status"]=="complete" for s in summary.values())} complete methods.')
    print('\n'.join(table))


if __name__=='__main__':
    main()
