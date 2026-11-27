"""Reload report-reference models and verify functional gate/ADMM evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.effective_gate import EffectiveGateMLP,load_retained_prox


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    manifest=json.loads((args.runs/'manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['source_hashes'].items():
        raw=Path(name).read_bytes()
        assert expected in (hashlib.sha256(raw).hexdigest(),hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest())
    prox,topk=load_retained_prox()
    checks=[]
    for row in map(json.loads,(args.runs/'results.jsonl').read_text().splitlines()):
        case=args.cases/f"{row['task']}-fold{row['fold']}.npz"
        assert hashlib.sha256(case.read_bytes()).hexdigest()==row['data_sha256']
        data=np.load(case)
        assert set(data['train']).isdisjoint(data['test'])
        raw=data['x'][:,data['permutation']][data['train']]
        state=torch.load(args.runs/row['checkpoint'],map_location='cpu',weights_only=True)
        np.testing.assert_array_equal(raw.mean(0),state['mean'].numpy())
        np.testing.assert_array_equal(raw.std(0)+1e-8,state['std'].numpy())
        x=torch.tensor((raw-state['mean'].numpy())/state['std'].numpy(),dtype=torch.float32)
        model=EffectiveGateMLP(512,x.square().mean(0),state['config']['hidden_size'])
        torch.testing.assert_close(model.square_mean,state['model']['square_mean'],rtol=0,atol=0)
        model.load_state_dict(state['model']);model.eval()
        g=model.effective_gate().detach()
        np.testing.assert_array_equal(g.abs().numpy(),row['scores'])
        q=torch.cat([model.linear.weight,model.square_weight],dim=0)*model.gate
        torch.testing.assert_close(q.norm(dim=0),g.abs(),rtol=2e-6,atol=1e-7)
        assert torch.count_nonzero(state['initial']['square_weight'])==0
        assert torch.all(state['initial']['gate']==1)
        assert len(state['trace'])==state['config']['epochs']==120
        assert all(t['prox']['certified'] for t in state['trace'])
        # u_new=u_old+g-z gives the exact final proximal center z+u_new.
        z,info=prox(state['z']+state['dual'],state['config']['penalty']/state['config']['rho'])
        torch.testing.assert_close(z,state['z'],rtol=2e-5,atol=1e-7)
        selected,selection=topk(g,len(data['support']))
        assert selected.tolist()==row['selected']
        # Exact structural deletion holds for both channels, independently of
        # whether this final soft training endpoint happens to contain zeros.
        with torch.no_grad():
            model.gate[0]=0
            changed=x[:8].clone();changed[:,0]+=13
            torch.testing.assert_close(model(x[:8]),model(changed),rtol=0,atol=0)
        checks.append(dict(task=row['task'],fold=row['fold'],effective_scores_exact=True,
            train_only_scaler_exact=True,zero_square_initialization=True,
            certified_prox_updates=len(state['trace']),final_prox_recomputed=True,
            structural_deletion_both_channels=True,selection=selection))
    report=dict(status='passed',cases=len(checks),certified_updates=sum(c['certified_prox_updates'] for c in checks),checks=checks)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print({k:v for k,v in report.items() if k!='checks'})


if __name__=='__main__':
    main()
