"""Read audited results and draw meeting figures; no models or new experiments."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/reports/meeting_20260918'
# The numerical report is retained byte-for-byte as the technical appendix.
# The 09/06-aligned meeting narrative is verified separately.
REPORT = ROOT/'docs/reports/MEETING_REPORT_20260918_TECHNICAL_APPENDIX.md'
INPUTS = {}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read(relative):
    path=ROOT/relative
    INPUTS[relative]=sha(path)
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm,LinearSegmentedColormap
    OUT.mkdir(parents=True,exist_ok=True)
    grid=read('results/ablation/five_version_full_grid_r62_20260915/full_grid.json')
    transfer=read('results/ablation/five_version_transfer_r61_20260913/summary.json')['results']
    small={name:read(f'results/ablation/legacy_wd_family_r58_20260912/selection/{name}/summary.json')['results']
        for name in ('Base','M31')}
    other128=read('results/ablation/warm_square_factorial_r59_20260913/summary.json')['results']['aggregate']['W20_immediate']
    refit=read('results/ablation/ring_top2_refit_r60_20260913/summary.json')['aggregate']
    path=read('results/ablation/xor_lassonet_path_r63_20260915/summary.json')['results']['table']
    duration=read('results/ablation/xor_duration_r64_20260916/summary.json')['results']['table']
    read('results/ablation/xor_pair_perturbation_r65_20260916/summary.json')
    read('results/ablation/legacy_wn_highdim_source_r66_20260917/audit.json')
    warm_phases=read('results/ablation/five_stage_warmup_r70_20260918/ranking_curves.json')['phase_rows']
    legacy=read('results/main/synthetic_m_sweep_weightnorm_current_1000_legacywn_highm.json')
    dirs={
        'R67':'empirical_combinations_r67_20260917','R68':'five_forward_ladder_r68_20260917',
        'R69':'five_stage_state_r69_20260918','R70':'five_stage_warmup_r70_20260918'}
    arms=[('R67','SlowPath'),('R67','EarlySlowPath'),('R68','LegacyC'),('R68','LegacyCStage50'),
          ('R69','AdamReset'),('R69','AdamZUReset'),('R70','StageWarm20')]
    summaries,contracts={},{}
    for tag,directory in dirs.items():
        prefix='results/ablation/'+directory+'/'
        summaries[tag]=read(prefix+'summary.json');contracts[tag]=read(prefix+'contract.json')
        audit=read(prefix+'audit.json')
        assert audit['status']=='pass' and audit['validated_checkpoints']==summaries[tag]['trajectories']
        assert audit['summary_sha256']==INPUTS[prefix+'summary.json']
        assert audit['contract_sha256']==INPUTS[prefix+'contract.json']
        assert summaries[tag]['selected_for_fresh_confirmation'] is None
    columns=[('xor',256),('xor',512),('ring',256),('ring',512)]
    rows=[]
    baseline=[]
    for task,p in columns:
        item=next(v for v in grid['table'] if (v['dataset'],v['n_features'],v['arm'])==(task,p,'Five-v1'))
        baseline.append(item['metrics']['support_top2'])
    rows.append(dict(round='R61/R62',arm='Five-v1',epochs=120,coordinates=[v['coordinates'] for v in baseline],
        exact_folds=[v['exact_folds'] for v in baseline],joint_screen=None))
    for tag,name in arms:
        group=[next(v for v in summaries[tag]['table'] if (v['dataset'],v['n_features'],v['arm'])==(t,p,name)) for t,p in columns]
        for item,ref in zip(group,baseline):
            assert item['reference']['coordinates']==ref['coordinates']
            assert item['reference']['exact_folds']==ref['exact_folds']
            assert item['reference']['instance_coordinates']=={k:round(v*12) for k,v in ref['instance_recalls'].items()}
        arm=contracts[tag]['arms'][name]
        epochs=arm['warmup_epochs']+5*arm['stage_epochs']+4*arm.get('stage_warmup_epochs',0)
        rows.append(dict(round=tag,arm=name,epochs=epochs,
            coordinates=[v['actual']['coordinates'] for v in group],
            exact_folds=[v['actual']['exact_folds'] for v in group],
            instance_coordinates=[v['actual']['instance_coordinates'] for v in group],
            joint_screen=summaries[tag]['decisions'][name]['worth_fresh_confirmation']))
    assert [v['coordinates'] for v in rows]==[[2,1,18,18],[3,0,17,23],[0,0,15,16],
        [1,0,18,21],[2,0,19,20],[1,0,18,20],[1,0,18,21],[2,0,18,21]]
    assert [v['epochs'] for v in rows]==[120,520,505,120,270,120,120,200]
    assert sum(v['trajectories'] for v in summaries.values())==336
    assert sum(v['epochs'] for v in summaries.values())==89040
    legacy_short={t:{p:round(sum(legacy['results'][t][str(p)]['fold_best_k'])*2)
        for p in (256,512)} for t in ('xor','ring')}
    assert legacy_short=={'xor':{256:12,512:10},'ring':{256:1,512:0}}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.7,4.0),sharey=True)
    dims=[4,8,16,32,64,128,256,512,1024,2048]
    for ax,task in zip(axes,('xor','ring')):
        five=[next(v for v in grid['table'] if (v['dataset'],v['n_features'],v['arm'])==(task,p,'Five-v1'))['metrics']['support_top2']['coordinates'] for p in dims]
        base=[v for v in grid['table'] if v['dataset']==task and v['arm']=='Base']
        ax.plot(dims,five,'o-',color='#2563eb',linewidth=2,label='Five-v1')
        ax.scatter([v['n_features'] for v in base],[v['metrics']['support_top2']['coordinates'] for v in base],
            marker='s',s=53,color='#666666',zorder=4,label='Base (measured points only)')
        ax.set_xscale('log',base=2);ax.set_xticks(dims,[str(p) for p in dims],rotation=35)
        ax.set_ylim(-1,26);ax.set_yticks([0,6,12,18,24]);ax.set_title(task.upper());ax.grid(alpha=.18)
        ax.set_xlabel('Input dimension p')
    axes[0].set_ylabel('True coordinates recovered / 24');axes[1].legend(loc='lower left',fontsize=9)
    fig.suptitle('Five-v1 retains Ring recovery as high-dimensional XOR becomes weak',fontsize=13)
    fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'dimension_recovery.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    counts=np.array([v['coordinates'] for v in rows]);delta=counts-counts[0]
    cmap=LinearSegmentedColormap.from_list('changes',['#efba8c','#ffffff','#82b9db'])
    fig,ax=plt.subplots(figsize=(10.7,5.5))
    ax.imshow(delta,cmap=cmap,norm=TwoSlopeNorm(vmin=-5,vcenter=0,vmax=5),aspect='auto')
    labels=[f"{v['round']}  {v['arm']}   ({v['epochs']} epochs)" for v in rows]
    ax.set_yticks(range(len(rows)),labels);ax.set_xticks(range(4),['XOR 256','XOR 512','Ring 256','Ring 512'])
    ax.xaxis.tick_top();ax.tick_params(length=0,pad=8)
    for i in range(len(rows)):
        for j in range(4):
            change='reference' if i==0 else f'{delta[i,j]:+d} vs Five'
            ax.text(j,i,f'{counts[i,j]}/24\n{change}',ha='center',va='center',fontsize=10,color='#202020')
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
    ax.grid(which='minor',color='#dddddd',linewidth=.9);ax.tick_params(which='minor',length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle('Every candidate is compared with the same Five-v1 reference',fontsize=13,y=.995)
    fig.text(.5,.016,'Blue: more coordinates. Orange: fewer. Rows are separate recipes, not one cumulative path.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95))
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'candidate_comparison.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    doc=REPORT.read_text(encoding='utf-8')
    expected_lines=[]
    for row in rows:
        expected_lines.append('| '+row['arm']+' | '+str(row['epochs'])+' | '+' | '.join(f'{v}/24' for v in row['coordinates'])+' |')
    for line in expected_lines:assert line in doc,line
    for p in (128,256,512):
        vals=[next(v for v in path if v['n_features']==p and v['arm']==name)['coordinates'] for name in ('Base','Five-v1','LassoNet')]
        assert '| '+str(p)+' | '+' | '.join(f'{v}/24' for v in vals)+' |' in doc
    for p in (4,8,16,32,64,128,256,512,1024,2048):
        vals=[next(v for v in grid['table'] if (v['dataset'],v['n_features'],v['arm'])==(t,p,'Five-v1'))['metrics']['support_top2']['coordinates'] for t in ('xor','ring')]
        assert f"| {p} | {vals[0]}/24 | {vals[1]}/24 |" in doc
    for task in ('xor','ring','ring+xor','ring+xor+sum'):
        a,b=small['Base'][task],small['M31'][task]
        label={'xor':'XOR','ring':'Ring','ring+xor':'Ring+XOR','ring+xor+sum':'Ring+XOR+Sum'}[task]
        assert f"| {label} | {a['coordinates']}/{a['opportunities']} | {b['coordinates']}/{b['opportunities']} |" in doc
    for p,stage,title in [(256,2,'第二阶段'),(256,5,'第五阶段'),(512,5,'第五阶段')]:
        row=next(v for v in warm_phases if (v['dataset'],v['n_features'],v['C_stage'])==('xor',p,stage))
        points=[row[k] for k in ('before','after','after_ADMM')]
        ranks=' → '.join(f"{v['median_worse_true_rank']:.1f}" for v in points)
        hits=' → '.join(str(v['coordinates']) for v in points)
        assert f'| XOR{p} {title} | {ranks} | {hits} /24 |' in doc
    # All numerical sources remain unchanged throughout this read-only build.
    for relative,digest in INPUTS.items():assert sha(ROOT/relative)==digest
    data=dict(date='2026-09-18',role='read_only_reporting',new_training=0,new_inference=0,new_selectors=0,
        audited_selector_ledger=4080,source_sha256=INPUTS,candidate_rows=rows,
        grid_table=grid['table'],R58=small,R59=other128,R60=refit,R63=path,R64=duration,
        legacy_unpaired=legacy_short,dag=[v for v in transfer['table'] if v['dataset']=='dag'],
        R61_macro_recall=transfer['macro_recall'],report_sha256=sha(REPORT),
        figure_sha256={p.name:sha(p) for p in OUT.glob('*.png')},
        checked_report_table_rows=len(expected_lines)+3+10+4+3,visual_review='pending')
    (OUT/'evidence.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='pass',source_files=len(INPUTS),checked_table_rows=data['checked_report_table_rows'],
        new_training=0,output=str(OUT)),ensure_ascii=False))

if __name__=='__main__':main()
