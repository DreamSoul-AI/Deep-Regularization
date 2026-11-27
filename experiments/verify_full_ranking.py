"""Audit saved 512-D public-reference and Localized Lasso model readouts."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from legacy_ranking_reference import load_reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=Path('.tmp/all-baselines-512-20261128'))
    p.add_argument('--evidence',type=Path,default=Path('docs/ranking-dashboard/512'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    rows=list(map(json.loads,(args.evidence/'verified_results.jsonl').read_text().splitlines()))
    namespace,_=load_reference()
    public=[];localized=[];attention=[]
    for row in rows:
        case=f"{row['task']}-fold{row['fold']}"
        if row['method']=='public_512_defaults_rho200':
            state=torch.load(args.runs/'public-defaults'/row['checkpoint'],weights_only=True,map_location='cpu')
            model=namespace['GatedFeatureSelectionMLP'](512,2)
            model.load_state_dict(state['model'])
            np.testing.assert_array_equal(model.gate.detach().abs().numpy(),row['scores'])
            data=np.load(args.runs/'cases'/f'{case}.npz')
            raw=data['x'][:,data['permutation']][data['train']]
            np.testing.assert_array_equal(raw.mean(0),state['mean'].numpy())
            np.testing.assert_array_equal(raw.std(0)+1e-8,state['std'].numpy())
            public.append(case)
        elif row['method']=='localized_lasso':
            name=f"{row['task']}-{row['fold']}"
            candidates=list((args.runs/'results').rglob(name+'-localized-weights.npy'))
            if len(candidates)!=1:
                raise ValueError('Localized checkpoint identity is ambiguous: '+case)
            w=np.load(candidates[0])
            assert w.shape[0]==512 and np.isfinite(w).all()
            np.testing.assert_array_equal(np.linalg.norm(w,axis=1),row['scores'])
            diagnostics=json.loads(candidates[0].with_name(name+'-linear-solves.json').read_text())
            assert len(diagnostics)==10
            assert [r['iteration'] for r in diagnostics]==list(range(10))
            residual=max(r['relative_residual'] for r in diagnostics)
            assert residual<=1e-11
            localized.append(dict(case=case,iterations=10,max_relative_residual=residual))
        elif row['method']=='attention_maps':
            file=args.runs/'results/attention_maps'/f"{row['task']}-{row['fold']}-attention_maps.pt"
            state=torch.load(file,map_location='cpu',weights_only=True)
            assert state['effective_batch']==512 and state['attention_microbatch']==16
            assert state['device']=='cuda' and state['history']
            assert all(np.isfinite(h['validation_loss']) for h in state['history'])
            attention.append(case)
    assert len(public)==len(localized)==len(attention)==24
    report=dict(status='passed',public_model_readouts_exact=len(public),localized_model_readouts_exact=len(localized),
        localized_linear_solves=sum(r['iterations'] for r in localized),
        max_localized_relative_residual=max(r['max_relative_residual'] for r in localized),
        attention_execution_checkpoints=len(attention),attention_effective_batch=512,attention_microbatch=16,
        localized=localized)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print({k:v for k,v in report.items() if k!='localized'})


if __name__=='__main__':
    main()
