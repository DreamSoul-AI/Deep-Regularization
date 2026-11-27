"""Reload local LassoFlexNet runs; audit knots, selection, scores and hierarchy."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import torch
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.lassoflexnet import FlexConfig,LassoFlexNet,PiecewiseLinearEncoding,IMPLEMENTATION_REVISION


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    checked=[]
    for file in sorted(args.runs.rglob('results.jsonl')):
        manifest=json.loads((file.parent/'manifest.json').read_text(encoding='utf-8'))
        for source,expected in manifest['source_hashes'].items():
            raw=Path(source).read_bytes()
            if expected not in (hashlib.sha256(raw).hexdigest(),hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()):
                raise ValueError('Training source differs from recorded snapshot: '+source)
        for row in map(json.loads,file.read_text(encoding='utf-8').splitlines()):
            case=args.cases/f"{row['task']}-fold{row['fold']}.npz"
            if hashlib.sha256(case.read_bytes()).hexdigest()!=row['data_sha256']:
                raise ValueError('Training data hash differs')
            data=np.load(case)
            state=torch.load(file.parent/f"{row['task']}-{row['fold']}.pt",map_location='cpu',weights_only=True)
            config=FlexConfig(**state['config'])
            assert state['architecture_revision']==manifest['architecture_revision']==IMPLEMENTATION_REVISION
            assert asdict(config)==manifest['model']
            raw=data['x'][:,data['permutation']][data['train']]
            x=(raw-raw.mean(0))/(raw.std(0)+1e-8)
            y=data['y'][data['train']]
            fit,valid=train_test_split(np.arange(len(x)),test_size=.2,random_state=manifest['seed'],stratify=y)
            np.testing.assert_array_equal(fit,state['fit_indices'])
            np.testing.assert_array_equal(valid,state['validation_indices'])
            assert set(fit).isdisjoint(valid)
            encoder=PiecewiseLinearEncoding.fit(x[fit],y[fit],bins=config.bins,seed=manifest['seed'])
            model=LassoFlexNet(encoder,config)
            for name,value in encoder.state_dict().items():
                torch.testing.assert_close(value,state['model']['encoder.'+name],rtol=0,atol=0)
            model.load_state_dict(state['model'])
            model.eval()
            scores=model.beta.detach().abs().numpy()
            np.testing.assert_array_equal(scores,row['scores'])
            path=state['history'][1:]
            selected=min(path,key=lambda r:r['validation_bce'])
            assert selected['penalty']==state['selected_lambda']==row['selected_lambda']
            assert selected['validation_bce']==state['selected_validation_bce']
            with torch.no_grad():
                loss=float(torch.nn.functional.binary_cross_entropy_with_logits(
                    model(torch.tensor(x[valid],dtype=torch.float32)),torch.tensor(y[valid],dtype=torch.float32)))
                violation=float(model.hierarchy_violation())
                difference=abs(loss-state['selected_validation_bce'])
                if difference>5e-6:
                    raise ValueError(f'CPU checkpoint validation BCE mismatch: {difference}')
                zero=torch.nonzero(model.beta==0).flatten()
                tested_zero=False
                if state['mode']=='sequential_feasible':
                    assert violation<=1e-6
                    assert max(r['max_update_hierarchy_violation'] for r in path)<=1e-6
                    if len(zero):
                        sample=torch.tensor(x[:8],dtype=torch.float32)
                        changed=sample.clone();changed[:,zero[0]]+=123.
                        torch.testing.assert_close(model(sample),model(changed),rtol=0,atol=0)
                        tested_zero=True
            checked.append(dict(task=row['task'],fold=row['fold'],mode=state['mode'],dimensions=x.shape[1],
                checkpoint_beta_scores_exact=True,training_only_tree_knots_exact=True,
                best_positive_path_checkpoint_verified=True,cpu_validation_bce_error=difference,
                hierarchy_violation=violation,zero_feature_deletion_tested=tested_zero,
                actual_training_epochs=sum(r['epochs'] for r in state['history'])))
    result=dict(status='passed',cases=len(checked),checks=checked,
                max_cpu_validation_bce_error=max(r['cpu_validation_bce_error'] for r in checked),
                max_selected_hierarchy_violation=max(r['hierarchy_violation'] for r in checked),
                deletion_tests=sum(r['zero_feature_deletion_tested'] for r in checked))
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print({k:v for k,v in result.items() if k!='checks'})


if __name__=='__main__':
    main()
