"""Verify portable baseline score metrics using only the Python standard library.

This verifies evidence consistency, not training correctness or model quality.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def score_metrics(scores, support, expected_dim=None):
    values=[float(x) for x in scores]
    dimension=len(values)
    if not dimension or (expected_dim is not None and dimension!=expected_dim) or not all(math.isfinite(x) for x in values):
        raise ValueError('Invalid score vector')
    positive=set(support)
    if not 0<len(positive)<dimension or not positive.issubset(range(dimension)):
        raise ValueError('Invalid support')
    order=sorted(range(dimension),key=lambda j:(-values[j],j))
    k=len(positive)
    recall=sum(j in positive for j in order[:k])/k
    ap,tp,offset=0.,0,0
    while offset<len(order):
        end=offset+1
        while end<len(order) and values[order[end]]==values[order[offset]]:
            end+=1
        group_tp=sum(j in positive for j in order[offset:end])
        tp+=group_tp
        ap+=(group_tp/k)*(tp/end)
        offset=end
    ascending=sorted(range(dimension),key=lambda j:values[j])
    rank_sum,offset=0.,0
    while offset<dimension:
        end=offset+1
        while end<dimension and values[ascending[end]]==values[ascending[offset]]:
            end+=1
        rank_sum+=sum(j in positive for j in ascending[offset:end])*(offset+1+end)/2
        offset=end
    auroc=(rank_sum-k*(k+1)/2)/(k*(dimension-k))
    return dict(recall_at_k=recall,average_precision=ap,auroc=auroc,top_k=order[:k])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence',type=Path)
    args=p.parse_args()
    cases=json.loads((args.evidence/'cases.json').read_text(encoding='utf-8'))
    summary=json.loads((args.evidence/'summary.json').read_text(encoding='utf-8'))
    source_manifest=args.evidence/'record_sources.json'
    if source_manifest.exists():
        names=json.loads(source_manifest.read_text(encoding='utf-8'))
        if not isinstance(names,list) or not 0<len(names)<=10:
            raise ValueError('Invalid record source manifest')
        sources=[(args.evidence/name).resolve() for name in names]
        repository=Path(__file__).resolve().parents[1]
        if any(p.suffix!='.jsonl' or not p.is_relative_to(repository) for p in sources):
            raise ValueError('Referenced evidence must be JSONL inside this repository')
    else:
        sources=[args.evidence/'verified_results.jsonl']
    records=[json.loads(line) for source in sources for line in source.read_text(encoding='utf-8').splitlines()]
    by_method=defaultdict(list)
    seen=set()
    def close(a,b,label):
        if not math.isclose(a,b,abs_tol=1e-12,rel_tol=0):
            raise ValueError((label,a,b))
    for row in records:
        key=(row['method'],row['task'],row['fold'])
        if key in seen:
            raise ValueError(('Duplicate case',key))
        seen.add(key)
        case=cases[f"{row['task']}-fold{row['fold']}"]
        actual=score_metrics(row['scores'],case['support'],expected_dim=case.get('dimensions',128))
        for metric in ['recall_at_k','average_precision','auroc']:
            close(actual[metric],row['metrics'][metric],(key,metric))
        if actual['top_k']!=row['metrics']['top_k']:
            raise ValueError(('Top-k mismatch',key))
        by_method[row['method']].append(row)
    for method,report in summary.items():
        rows=by_method[method]
        if report['completed']!=len(rows):
            raise ValueError(('Completed count mismatch',method))
        for task,task_report in report['tasks'].items():
            selected=[r for r in rows if r['task']==task]
            if task_report['n']!=len(selected):
                raise ValueError(('Task count mismatch',method,task))
            for metric in ['recall_at_k','average_precision','auroc']:
                if selected:
                    close(sum(r['metrics'][metric] for r in selected)/len(selected),task_report[metric],(method,task,metric))
        for overall,metric in [('macro_recall','recall_at_k'),('macro_ap','average_precision'),('macro_auroc','auroc')]:
            if report['status']=='complete':
                if len(rows)!=6*len(report['tasks']):
                    raise ValueError(('Incomplete method marked complete',method))
                close(sum(report['tasks'][task][metric] for task in report['tasks'])/len(report['tasks']),report[overall],(method,overall))
            elif report[overall] is not None:
                raise ValueError(('Partial method has macro score',method))
    count=sum(bool(rows) for rows in by_method.values())
    print(f'Verified {len(records)} score vectors across {count} methods; metrics and aggregate tables agree.')


if __name__=='__main__':
    main()
