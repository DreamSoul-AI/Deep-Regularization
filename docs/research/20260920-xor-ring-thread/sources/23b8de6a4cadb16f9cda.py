"""Read-only review of 72 audited R61 XOR traces; no model or training import."""
from pathlib import Path
import hashlib
import json
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
R61 = ROOT / 'results/ablation/five_version_transfer_r61_20260913'
OUT = ROOT / 'docs/analysis/data/r61_xor_trace_review_20260915.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    closure = json.loads((R61/'closeout.json').read_text(encoding='utf-8'))
    source_sha = sha(R61/'run.jsonl')
    assert source_sha == closure['artifacts_sha256'][(R61/'run.jsonl').relative_to(ROOT).as_posix()]
    rows = []
    with (R61/'run.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line)
            if r['event'] != 'final_result':
                continue
            assert r['dataset'] == 'xor'
            truth = set(r['truths']['support'])
            counts, true_ranks = [], []
            for step in r['epoch_trace']:
                scores = step['effective_gate']
                ranked = sorted((j for j,v in enumerate(scores) if v != 0), key=lambda j:(-abs(scores[j]),j))
                assert ranked[:2] == step['topk']
                counts.append(len(truth.intersection(ranked[:2])))
                true_ranks.append([ranked.index(j)+1 if j in ranked else len(scores)+1 for j in sorted(truth)])
            assert len(counts) == 100
            assert counts[-1] == r['evaluation']['metrics']['support_top2']['recovered_count']
            rows.append(dict(p=r['n_features'],arm=r['arm'],instance=r['instance'],fold=r['fold'],
                counts=counts,true_ranks=true_ranks,first=counts[0],final=counts[-1],
                ever_any=any(v>0 for v in counts),ever_exact=2 in counts,
                first_train_bce=r['epoch_trace'][0]['train_supervised_loss'],
                train_metrics=r['full_model_train'],test_metrics=r['full_model_test']))
            if len(rows) == 72:
                break
    assert len({(r['p'],r['arm'],r['instance'],r['fold']) for r in rows}) == 72
    summaries = []
    for p in (128,512,2048):
        for arm in ('Base','Five-v1'):
            sub = [r for r in rows if r['p']==p and r['arm']==arm]
            assert len(sub)==12
            summaries.append(dict(p=p,arm=arm,first_coordinates=sum(r['first'] for r in sub),
                final_coordinates=sum(r['final'] for r in sub),
                trajectories_ever_any=sum(r['ever_any'] for r in sub),
                trajectories_ever_exact=sum(r['ever_exact'] for r in sub),
                first_exact_later_lost=sum(r['first']==2 and r['final']<2 for r in sub),
                train_auroc=mean(r['train_metrics']['auroc'] for r in sub),
                test_auroc=mean(r['test_metrics']['auroc'] for r in sub),
                first_train_bce=mean(r['first_train_bce'] for r in sub),
                epoch_coordinate_counts=[sum(r['counts'][e] for r in sub) for e in range(100)]))
    report = dict(status='pass',read_only=True,new_training=0,new_inference=0,verified_trajectories=72,
        verified_admm_rankings=7200,source_jsonl_sha256=source_sha,script_sha256=sha(__file__),
        summary=summaries,rows=rows,
        boundary='First record is after warmup AND first ADMM epoch. No warmup snapshot or causal attribution; oracle ranks are post hoc only.')
    with OUT.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps([{k:v for k,v in r.items() if k!='epoch_coordinate_counts'} for r in summaries]))


if __name__=='__main__':
    main()
