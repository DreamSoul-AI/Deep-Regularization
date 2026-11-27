"""Refresh the single current ranking dashboard from verified, paired scores."""
import argparse
from datetime import datetime,timezone,timedelta
from decimal import Decimal,ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import sys
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baseline_suite import metrics

TASKS=['xor','ring','ring+xor','ring+xor+sum']
LABELS={'raw_snapshot_column_norm':'列范数对照',
 'raw_snapshot_taylor_first':'一阶',
 'raw_snapshot_taylor_fisher':'一阶＋二阶',
 'gate_only':'仅 gate', 'column_norm_gate':'A × gate', 'taylor_second':'纯二阶',
 'column_norm_taylor_first':'A × 一阶', 'column_norm_taylor_second':'A × 纯二阶',
 'column_norm_taylor_fisher':'A ×（一阶＋二阶）',
 'public_512_defaults_rho200':'原方案',
 'effective_gate_oct9_reference':'有效 gate',
 'lassoflexnet_reimpl_sequential_feasible':'LassoFlexNet',
 'mrmr_fullrank':'mRMR','localized_lasso':'Localized Lasso',
 'lassonet':'LassoNet','rf':'Random Forest','treeshap':'TreeSHAP','random':'Random',
 'mi':'MI','stg':'STG','relief':'Relief','fsnet':'FSNet','e2efs':'E2E-FS',
 'cae_keras':'CAE','deeppink':'DeepPINK','xgboost':'XGBoost','lasso':'Lasso',
 'univariate':'Univariate','attention_maps':'Attention Maps','deep_lasso':'Deep Lasso',
 'first_layer_lasso':'First-layer Lasso','adaptive_group_lasso':'Adaptive Group Lasso',
 'canceloutsigmoid':'CancelOut Sigmoid','canceloutsoftmax':'CancelOut Softmax'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=Path('.tmp/all-baselines-512-20261128'))
    p.add_argument('--cases',type=Path,default=Path('.tmp/all-baselines-512-20261128/cases'))
    p.add_argument('--output',type=Path,default=Path('docs/ranking-dashboard/512'))
    p.add_argument('--dashboard',type=Path,default=Path('docs/实验看板 Deep Regularization.md'))
    args=p.parse_args()
    previous=json.loads(Path('docs/研究记录 Research Record/20261128-baseline-repairs/128/summary.json').read_text())
    planned=[name.replace('ours_','raw_snapshot_') for name in previous]
    planned+=['effective_gate_oct9_reference','public_512_defaults_rho200']
    cases={}
    for file in sorted(args.cases.glob('*.npz')):
        data=np.load(file)
        cases[file.stem]=dict(support=data['support'].tolist(),dimensions=512,
            sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    if len(cases)!=24:
        raise ValueError('The four-task dashboard requires all 24 paired cases')
    files=list((args.runs/'results').rglob('*.jsonl'))
    files += list((args.runs/'public-defaults').glob('results.jsonl'))
    files += list((args.runs/'effective-mixed').glob('results.jsonl'))
    files += list((args.runs/'flex-mixed-gpu').glob('results.jsonl'))
    files += list(Path('.tmp/effective-gate-512-20261128/run').glob('results.jsonl'))
    files += list(Path('.tmp/baseline-repairs-20261128/flex-v2-512').rglob('results.jsonl'))
    own=args.cases/'results.jsonl'
    if own.exists():files.append(own)
    records={};failures={};sources=[]
    device_audit=args.runs/'lassonet-device-audit.json'
    native_device=json.loads(device_audit.read_text())['native_default_device'] if device_audit.exists() else None
    for file in sorted(files):
        raw=file.read_bytes()
        sources.append(dict(path=file.as_posix(),sha256=hashlib.sha256(raw).hexdigest()))
        # A running worker may be midway through its final append.
        complete_bytes=raw if raw.endswith(b'\n') else raw.rsplit(b'\n',1)[0] if b'\n' in raw else b''
        for line in complete_bytes.decode('utf-8').splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if 'mode' in row and 'method' not in row:
                row=dict(task=row['task'],fold=row['fold'],method='raw_snapshot_'+row['mode'],status='ok',
                    scores=row['scores']['gate'],metrics=row['metrics']['gate'],
                    seconds=row['seconds'],reference_max_parameter_difference=row['reference_max_parameter_difference'])
            if row.get('method') not in planned or row.get('task') not in TASKS:continue
            if row['method']=='lassonet' and native_device and not row.get('device_bound_explicitly',False):
                row=dict(row,requested_device=row.get('device'),device=native_device,
                         device_correction='Native original constructor auto-selection audited on this host; scores unchanged')
            key=(row['method'],row['task'],row['fold'])
            if row.get('status')!='ok':
                failures.setdefault(row['method'],[]).append(dict(task=row['task'],fold=row['fold'],error=row.get('error','unknown')))
                continue
            case=cases[f"{row['task']}-fold{row['fold']}"]
            if 'data_sha256' in row and row['data_sha256']!=case['sha256']:
                raise ValueError('Paired data bytes differ: '+str(key))
            actual=metrics(row['scores'],case['support'],expected_dim=512)
            for name,value in actual.items():
                if name=='top_k':assert value==row['metrics'][name],key
                else:assert abs(value-row['metrics'][name])<1e-12,key
            if key in records and records[key]['scores']!=row['scores']:
                raise ValueError('Conflicting successful records: '+str(key))
            records[key]=row
    summary={}
    for method in planned:
        rows=[r for k,r in records.items() if k[0]==method]
        task_summary={}
        for task in TASKS:
            selected=[r for r in rows if r['task']==task]
            task_summary[task]=dict(n=len(selected),**{m:sum(r['metrics'][m] for r in selected)/len(selected) if selected else None
                for m in ['recall_at_k','average_precision','auroc']})
        complete=len(rows)==24
        summary[method]=dict(status='complete' if complete else 'incomplete',completed=len(rows),tasks=task_summary,
            failed_attempts=len(failures.get(method,[])),**{key:sum(task_summary[t][m] for t in TASKS)/4 if complete else None
                for key,m in [('macro_recall','recall_at_k'),('macro_ap','average_precision'),('macro_auroc','auroc')]})
    args.output.mkdir(parents=True,exist_ok=True)
    def write(name,data):
        (args.output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if device_audit.exists():
        write('lassonet-device-audit.json',json.loads(device_audit.read_text()))
        sources.append(dict(path=device_audit.as_posix(),sha256=hashlib.sha256(device_audit.read_bytes()).hexdigest()))
    write('summary.json',summary);write('cases.json',cases);write('failures.json',failures);write('sources.json',sources)
    (args.output/'verified_results.jsonl').write_text(''.join(json.dumps(records[k])+'\n' for k in sorted(records)),encoding='utf-8')
    complete=sum(s['status']=='complete' for s in summary.values())
    ablation_file=args.dashboard.parent/'ranking-dashboard/score-ablation-512/summary.json'
    ablation=json.loads(ablation_file.read_text(encoding='utf-8')) if ablation_file.exists() else {}
    readout_dir=args.dashboard.parent/'ranking-dashboard/score-readout'
    corrected={dim:json.loads((readout_dir/dim/'summary.json').read_text(encoding='utf-8')) for dim in ('128','512')}
    for table, dimension in ((previous,'128'),(summary,'512'),(ablation,'512')):
        for method in table.keys() & corrected[dimension].keys():
            if table[method]['status']=='complete':
                table[method]=corrected[dimension][method]
    def percent(value):
        return str(Decimal(f'{100*value:.10f}').quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
    now=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')
    head=['# Deep Regularization 实验看板','',f'更新时间：{now}（北京时间）。这是当前唯一汇总入口；旧日期记录只作方法与证据追溯。','',
      f'512维全量任务：**{complete}/{len(planned)}条方法线完成**，已核验{len(records)}/{len(planned)*24}个逐折结果。范围为 XOR、Ring、Ring+XOR、Ring+XOR+Sum，各1000样本、普通六折，只做ranking，不调参或下游重训。全量指本轮声明的方法清单，不扩到原CLI的attr-t、10模型bootstrap attr-b、普通NN预测或DAG2000维。','',
      '128维也已完成36条方法线、864个逐折结果。下方并列保留两张完整表；方法名只写名称，实现和参数差异统一放在表后。有效 gate 重建参照尚未跑128维，不用旧报告数字填入本轮表。','',
      '## 当前判断','',
      '主表默认按阈值评分从高到低排。关闭软阈值时使用同一评分公式。分数在每折训练终点计算：原列范数方案按 A 排，梯度组按各自的梯度分排，乘 A 的组按乘积排。共用阈值的仅 gate 组按 |s| 排。各外部 baseline 保留自己的评分。','',
      ('512维四任务平均为原仓库默认 '+percent(summary['public_512_defaults_rho200']['macro_recall'])+'%、一阶 '+percent(summary['raw_snapshot_taylor_first']['macro_recall'])+'%、一阶＋二阶 '+percent(summary['raw_snapshot_taylor_fisher']['macro_recall'])+'%。' if all(summary[m]['status']=='complete' for m in ['public_512_defaults_rho200','raw_snapshot_taylor_first','raw_snapshot_taylor_fisher']) else '完整结论等待对应四任务齐全。'),'',
      ('全部方法已跑齐。本轮四任务均值最高为 '+LABELS.get(max(summary,key=lambda m:summary[m]['macro_recall']),max(summary,key=lambda m:summary[m]['macro_recall']))+'，'+percent(max(s['macro_recall'] for s in summary.values()))+'%；结论只适用于本轮固定设置，不替代多种子或充分调参后的比较。' if complete==len(planned) else '其余方法尚未齐全，暂不宣布全表最优方法。'),'',
      '## 怎么读结果','',
      '每列是 Top-k 相关特征恢复率：k依次为2、2、4、6。100%表示找回该任务全部真实特征，0%表示一个也没找回。它不是分类准确率。四任务均完成后才计算 Macro；缺折不当作0，也不生成部分任务总排名。','',
      '此前的三方法小表只有 XOR/Ring：有效 gate 擅长 Ring、XOR失败；现有一阶＋Fisher在XOR成功、Ring较弱。那张两任务均值不能代表四任务总体，也不能证明只改score有收益，因为骨干、正则求解器和训练预算不同。','',
      '## 512维','',
      '| 方法 | XOR | Ring | Ring+XOR | Ring+XOR+Sum | 四任务 Macro | 完成状态 |',
      '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    sweep_file=args.dashboard.parent/'ranking-dashboard/seed-sweep/summary.json'
    if sweep_file.exists():
        sweep=json.loads(sweep_file.read_text(encoding='utf-8'))
        modes=('column_norm_gate','column_norm_taylor_first','column_norm_taylor_second','column_norm_taylor_fisher')

        def pp(value):
            value=round(value,12)
            return ('−' if value<0 else '+' if value>0 else '')+percent(abs(value))

        dim512=sweep['by_dimension']['512']
        against_gate=dim512['contrasts']['fisher_minus_gate']
        against_first=dim512['contrasts']['fisher_minus_first']
        section=['## 原方法训练，多种子排名对照', '',
            f'使用{len(sweep["independent_data_seeds"])}个独立数据种子、四个任务、每任务六折。共核验{sweep["models_verified"]}个模型与{sweep["rankings_verified"]}个排名读出。每个模型训练一次，四种方法只更换最终排序的 q。旧512维种子0留作复现参考。', '',
            f'512维上，一阶＋二阶相对 Gate 平均变化 {pp(against_gate["mean"])} 个百分点，相对单用一阶平均变化 {pp(against_first["mean"])} 个百分点。', '',
            '| 维数 | A × Gate | A × 一阶 | A × 纯二阶 | A ×（一阶＋二阶） | 一阶＋二阶对 Gate | 二阶加到一阶后的变化 |',
            '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for d in sweep['dimensions']:
            dim=sweep['by_dimension'][str(d)]
            values=[percent(dim['macro'][m])+'%' for m in modes]
            deltas=[pp(dim['contrasts'][key]['mean'])+'点' for key in ('fisher_minus_gate','fisher_minus_first')]
            section.append('| '+' | '.join([str(d),*values,*deltas])+' |')
        section += ['', '### 512维逐种子', '',
            '| 数据种子 | A × Gate | A × 一阶 | A × 纯二阶 | A ×（一阶＋二阶） | 一阶＋二阶对 Gate | 二阶加到一阶后的变化 |',
            '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for seed,report in dim512['by_seed'].items():
            values=[percent(report['macro'][m])+'%' for m in modes]
            deltas=[pp(report['deltas'][key])+'点' for key in ('fisher_minus_gate','fisher_minus_first')]
            section.append('| '+' | '.join([seed,*values,*deltas])+' |')
        section += ['',
            f'512维上，一阶＋二阶相对 Gate 的逐种子差值范围为 {pp(against_gate["minimum"])} 至 {pp(against_gate["maximum"])} 个百分点，胜／平／负为 {against_gate["wins"]}/{against_gate["ties"]}/{against_gate["losses"]}。相对一阶为 {against_first["wins"]}/{against_first["ties"]}/{against_first["losses"]}。统计单位是数据种子。', '',
            '指标是四任务等权的 Top-k 真实特征恢复率。跨维数沿用同一批样本和标签，增减输入噪声列并重新置换。固定模型随机种子0、原方法500轮、warm-up 120、C=0.05、初始ρ50，并沿用原有ρ更新。训练的 soft 左边为 s+u，右边为 C/(ρA)。训练结束后按 soft(Aq, C/(ρA)) 的正分输出排名。逐任务、逐折结果和复跑入口见 [多种子证据](ranking-dashboard/seed-sweep/README.md)。', '']
        position=head.index('## 当前判断')
        head[position]='## 单种子全量对照'
        head[position:position]=section
    priority=['public_512_defaults_rho200','effective_gate_oct9_reference','raw_snapshot_column_norm',
              'gate_only','column_norm_gate','raw_snapshot_taylor_first','taylor_second',
              'raw_snapshot_taylor_fisher','column_norm_taylor_first','column_norm_taylor_second',
              'column_norm_taylor_fisher']
    if ablation:
        if set(summary).intersection(ablation):
            raise ValueError('Score ablation method names overlap existing results')
        summary.update(ablation)
        planned.extend(ablation)
        count=sum(r['completed'] for r in ablation.values())
        head[6:6]=[f'512维新增 score 对照{len(ablation)}条方法线，已核验{count}个逐折结果，合计{len(planned)}条方法线。', '']
        focus=['gate_only','taylor_second','column_norm_taylor_fisher']
        if all(ablation.get(method,{}).get('status')=='complete' for method in focus):
            gate,second,weighted=(percent(ablation[method]['macro_recall']) for method in focus)
            position=head.index('## 怎么读结果')
            head[position:position]=[
                f'新增对照中，仅 gate 的四任务平均为{gate}%，纯二阶为{second}%，A ×（一阶＋二阶）为{weighted}%。同配方的列范数对照为{percent(summary["raw_snapshot_column_norm"]["macro_recall"])}%。', '',
            ]
    priority=[method for method in priority if method in summary]
    order=priority+sorted([m for m in planned if m not in priority],key=lambda m:(summary[m]['status']!='complete',-(summary[m]['macro_recall'] or 0),m))
    for method in order:
        s=summary[method]
        values=[percent(s['tasks'][t]['recall_at_k']) if s['tasks'][t]['n']==6 else f"—（{s['tasks'][t]['n']}/6）" for t in TASKS]
        macro=percent(s['macro_recall']) if s['macro_recall'] is not None else '—'
        status='完成' if s['status']=='complete' else ('待首折/排队' if s['completed']==0 else '未齐')
        if s['failed_attempts']:status+=f"；{s['failed_attempts']}次失败记录"
        head.append('| '+' | '.join([LABELS.get(method,method),*values,macro,f"{s['completed']}/24 {status}"])+' |')
    head += ['', '## 128维','',
      '| 方法 | XOR | Ring | Ring+XOR | Ring+XOR+Sum | 四任务 Macro | 完成状态 |',
      '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for method in sorted(previous,key=lambda m:-(previous[m]['macro_recall'] or 0)):
        report=previous[method]
        key=method.replace('ours_','raw_snapshot_')
        label='原方案' if method=='ours_column_norm' else LABELS.get(key,key)
        values=[percent(report['tasks'][t]['recall_at_k']) for t in TASKS]
        head.append('| '+' | '.join([label,*values,percent(report['macro_recall']),f"{report['completed']}/24 完成"])+ ' |')
    threshold_file=args.dashboard.parent/'ranking-dashboard/threshold-ablation-512/summary.json'
    if threshold_file.exists():
        threshold=json.loads(threshold_file.read_text(encoding='utf-8'))
        original_readout=json.loads(threshold_file.with_name('manifest.json').read_text(encoding='utf-8'))['original_rho50_models_same_readout']
        pairs=[('Gate','gate_adaptive','column_norm_gate'),
               ('一阶','taylor_first','column_norm_taylor_first'),
               ('纯二阶','taylor_second','column_norm_taylor_second'),
               ('一阶＋二阶','taylor_fisher','column_norm_taylor_fisher')]
        comparison=['### 旧对照：改变阈值分母', '',
            '四任务平均恢复率（%）。每个格子按该行 score 排名，同一列范数设置下，训练时有无软阈值的最终排名公式相同。', '',
            '| 评分 | 无软阈值 | 有软阈值 | 无软阈值 × A | 有软阈值 × A |',
            '| --- | ---: | ---: | ---: | ---: |']
        for label,plain,weighted in pairs:
            keys=[f'{kind}_{mode}' for mode in (plain,weighted) for kind in ('none','soft')]
            comparison.append('| '+' | '.join([label,*[percent(threshold[key]['macro_recall']) for key in keys]])+' |')
        comparison += ['',
            '上一轮的仅 gate 训练用了共用软阈值 C/ρ。此处无软阈值组直接跳过软阈值步骤，保留 ADMM 的公共缩放和对偶更新，所以每折只训练一个模型，再按八种 score 读出排名。有软阈值组由对应 score 调阈值。', '',
            f'原列范数方案（初始ρ50）按 A 排名为{percent(summary["raw_snapshot_column_norm"]["macro_recall"])}%。同一模型按 |s| 和 A|s| 排分别为{percent(original_readout["gate_adaptive"]["macro_recall"])}%和{percent(original_readout["column_norm_gate"]["macro_recall"])}%。软阈值对照证据见 [逐折结果](ranking-dashboard/threshold-ablation-512/README.md)。', '',
            '### 有软阈值时，最终排名是否乘 A', '',
            '每一对数字来自同一训练模型。左半边是训练用 q 调阈值，右半边是训练用 Aq 调阈值。q 表示该行基础评分，包含 Gate 的 |s|。', '',
            '| 评分 | 训练 q，按 q 排 | 训练 q，按 Aq 排 | 训练 Aq，按 q 排 | 训练 Aq，按 Aq 排 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for label,plain,weighted in pairs:
            values=[percent(corrected['512'][f'soft_{mode}__rank_{weight}']['macro_recall'])
                    for mode in (plain,weighted) for weight in ('q','aq')]
            comparison.append('| '+' | '.join([label,*values])+' |')
        comparison += ['']
        position=head.index('## 128维')
        head[position:position]=comparison
    fixed_dir=args.dashboard.parent/'ranking-dashboard/fixed-threshold-512'
    if (fixed_dir/'score/summary.json').exists():
        arms=('readout','score','weighted_consensus')
        fixed={arm:json.loads((fixed_dir/arm/'summary.json').read_text(encoding='utf-8')) for arm in arms}
        pairs=[('Gate','gate_adaptive','column_norm_gate'),('一阶','taylor_first','column_norm_taylor_first'),
               ('纯二阶','taylor_second','column_norm_taylor_second'),('一阶＋二阶','taylor_fisher','column_norm_taylor_fisher')]
        comparison=['### 512维固定软阈值，比较左边输入', '',
            '各组右边阈值都使用原方法 C/(ρA)。左边的 q 和 Aq 不归一化。有训练改动的组仍沿用原公共缩放和对偶更新。', '',
            '| 评分 | 原训练，soft(q) 排名 | 原训练，soft(Aq) 排名 | z 用 q | z 用 Aq | z 用 A(s+u)，soft(q) 排名 | z 用 A(s+u)，soft(Aq) 排名 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for label,base,weighted in pairs:
            values=[]
            for arm in arms:
                for mode in (base,weighted):
                    report=fixed[arm][f'{arm}_{mode}_soft']
                    value=report['macro_positive_recall']
                    values.append(percent(value) if value is not None else f'—（{report["completed"]}/24）')
            comparison.append('| '+' | '.join([label,*values])+' |')
        comparison += ['',
            '表中是四任务平均恢复率（%）。按软阈值后的值排名，只用正分项填 Top-k，剩余名额按未找回计。z 用 q / Aq 两列最终用各自相同的左边公式做 soft 再排名。原训练在 z 步使用 s+u。三个训练轨迹的 A 和 ρ 可以随参数变化，阈值公式一致。', '',
            '未做最终 soft 的读出、全零折和训练诊断见 [固定阈值对照证据](ranking-dashboard/fixed-threshold-512/README.md)。', '']
        uniform_training_file=fixed_dir/'uniform_training/summary.json'
        if uniform_training_file.exists():
            uniform_training=json.loads(uniform_training_file.read_text(encoding='utf-8'))
            comparison += ['#### 训练中左边固定 Aq，右边是否除以 A', '',
                '512维配对重训。左右两列只改变训练时的软阈值。训练完成后统一用 soft(Aq, C/(ρA)) 排名。', '',
                '| 评分 | 训练用 C/(ρA) | 训练用 C/ρ |',
                '| --- | ---: | ---: |']
            for label,_,weighted in pairs:
                original=fixed['score'][f'score_{weighted}_soft']
                shared=uniform_training[f'uniform_training_{weighted}_soft']
                values=[percent(r['macro_positive_recall']) if r['status']=='complete'
                        else f'—（{r["completed"]}/24）' for r in (original,shared)]
                comparison.append('| '+' | '.join([label,*values])+' |')
            comparison += ['', '恢复率只计正分 Top-k。各折软阈值前的评分、全零折和训练状态见 [重训证据](ranking-dashboard/fixed-threshold-512/uniform_training/summary.json)。', '']
        uniform_file=fixed_dir/'uniform-readout/comparison.json'
        if uniform_file.exists():
            uniform_records=json.loads(uniform_file.read_text(encoding='utf-8'))
            comparison += ['#### 仅最终排名时，右边是否除以 A', '',
                '512维原训练 checkpoint 配对比较，四任务平均恢复率（%）。最终 ρ 都为50，共用阈值 C/ρ 为0.001。', '',
                '| 评分 | C/(ρA) | C/ρ | 直接按 Aq 排 |',
                '| --- | ---: | ---: | ---: |']
            for label,_,weighted in pairs:
                matched=[r for r in uniform_records if r['mode']==weighted]
                if len(matched)!=24:
                    raise ValueError(f'Expected 24 uniform-threshold cases for {weighted}')
                values=[percent(sum(r[f'{variant}_positive_recall'] for r in matched)/len(matched))
                        for variant in ('divided','uniform','raw')]
                comparison.append('| '+' | '.join([label,*values])+' |')
            comparison += ['', '同一批 checkpoint 的逐折对比和 Top-k 变更见 [读出证据](ranking-dashboard/fixed-threshold-512/uniform-readout/comparison.json)。', '']
        position=head.index('### 旧对照：改变阈值分母') if '### 旧对照：改变阈值分母' in head else head.index('## 128维')
        head[position:position]=comparison
        position=head.index('## 怎么读结果')
        if uniform_training_file.exists():
            head[position:position]=['另有训练阈值消融，固定左边 Aq，在训练中配对比较右边 C/(ρA) 和 C/ρ。见 [训练中右边是否除以 A](#训练中左边固定-aq右边是否除以-a)。', '']
        else:
            head[position:position]=['512维消融固定原方法右边的软阈值。左边乘 A 的三种位置对照见 [固定软阈值，比较左边输入](#512维固定软阈值比较左边输入)。', '']
    head += ['', '## 表外说明','',
      'LassoFlexNet这一行是非官方自实现，采用论文正文的可行更新；mRMR使用原生完整贪心顺序评价排名。这些实现说明不再放进方法名。两张表中的原方案分别采用该维数的公开默认值：128维初始ρ50，512维初始ρ200；512维的列范数对照则保持ρ50。','',
      '## 原实验仓库到底是什么基线','',
      '固定来源为 [feature-selection@555ba11（2026-07-10）](https://github.com/DreamSoul-AI/Deep-Regularization/tree/555ba11e8872bd6618cd721bd3b08f7128db9e6b)。main/dev根模型与研究分支不同，9月模型以证据快照保留；未发现10月9日或11月A—L的新源码分支。','',
      '- 原公开512默认采用 Mish32×32、500轮/120 warm-up、C=.05、lr=.005、初始ρ200、batch64、WD=.001、feature dropout=.6。训练直接调用固定Git对象内的原函数，主表按本轮约定用最终 A 排名。维数分段ρ见 [原包装器](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/src/admm_input_group_wrapper.py#L1338)。','- 先前公开快照三条线是固定128维配方扩到512，初始ρ50。','- 10月有效gate参照采用共享线性/平方父gate、ReLU32×32、120轮、全局C=.003172、ρ=.18857、lr=.002084、canonical prox、零warm/WD/dropout。该行沿用有效 gate 的排序。','- 一阶和一阶＋Fisher沿用 raw gate 训练，每折按对应梯度评分排序。','',
      '旧版按 |s| 排时，ρ50和ρ200的四任务恢复率恰好一致。旧结果见 [逐折核对](ranking-dashboard/512/rho-reference-comparison.json)。','',
      '原清单含RF、TreeSHAP、Relief、FSNet、E2E-FS、mRMR、MI、LassoNet、CAE、两种CancelOut、DeepPINK及10种归因；普通NN是归因骨干。STG、随机控制、XGBoost、Localized Lasso、LassoFlexNet及论文补充选择器也纳入本轮。LassoFlexNet是明确标记的本地文本可行变体，官方代码未取得，不另造官方成绩。','',
      '## 原commit保存的历史512维结果','',
      '原commit携带的历史均值文件仅作追溯；缺少执行版本/环境和逐折分数，不混入上面两张本轮结果表。','',
      '来源例：[LassoNet XOR原始结果文件](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/Feature-Selection-Benchmark/results/lassonet-xor-1000.txt)。历史mRMR Ring100%与本轮Pearson估计器不同，执行来源尚未闭合，不能直接说方法退化。','',
      '## 比较边界与当前证据','',
      ('本次未显式绑定的旧LassoNet记录，原生构造器实际自动使用CUDA；旧日志的CPU是请求值，Windows上的空CUDA可见性未禁用GPU。当前汇总已更正设备元数据，原分数不改，[设备审计](ranking-dashboard/512/lassonet-device-audit.json)保留依据。未来入口显式传入设备。' if any(r.get('device_correction') for r in records.values()) else 'LassoNet设备按逐折显式记录，不从空CUDA可见性推断。'),'',
      'Attention Maps的512维物理batch512曾占用约26GB内存，已保留停止记录并改为GPU上16样本分块累积：有效batch仍为512，每批仍只更新一次，模型/学习率不变。实际上游FT模型在关闭dropout时的完整/分块梯度一致；dropout开启会改变随机数次序，不声称逐位同轨迹。Localized Lasso仅按不同任务/折分片并行，求解目标和数值门限不变。','',
      '本轮baseline统一训练折z-score；原benchmark先2*x-1、各wrapper自行预处理。Relief保留原生符号；原benchmark统一取abs。mRMR用原生完整贪心顺序，原k集合包装器内部随机分数只留作诊断。LassoNet排名来自路径 feature_importances_，不是最低val_loss checkpoint的β；best-val模型只用于预测。见 [原LassoNet调用](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/src/core.py#L175)。','',
      '所有方法复用相同保存的数据/划分/支持真值。六折训练集重叠且来自一个生成实例，不能冒充多个独立数据生成种子。不同结构与预算是方法间比较，不能作为只换评分的消融。真实相关坐标不参与训练或选正则参数；评价使用真值k，FSNet、CAE和E2E-FS等原生预算选择器也预先给定相同k。128维和512维并非只追加噪声列的嵌套数据，跨维数差异不能全部归因于新增噪声。','',
      '[当前分数读出证据](ranking-dashboard/score-readout/README.md)、[512维旧读出与外部baseline](ranking-dashboard/512/summary.json)、[数据身份](ranking-dashboard/512/cases.json)。刷新看板用 `python experiments/update_ranking_dashboard.py`。','',
      '128维原始汇总见 [既有证据](<研究记录 Research Record/20261128-baseline-repairs/128/table.md>)，不与512维成绩合并。方法推导与旧运行细节从 [研究计划](<研究计划 Research Plan Deep Regularization.md>) 查阅。']
    if ablation:
        position=head.index('## 原实验仓库到底是什么基线')
        head[position:position]=[
            '512维 score 对照统一使用已有的初始ρ50、500轮及120轮 warm-up 配方。仅 gate 组共用软阈值 C/ρ，ρ仍按原 ADMM 规则全局更新。列范数对照用第一层列范数 A 调节各特征阈值。A × gate 组用 A|s|，其余组用一阶、纯二阶或一阶加二阶的删除评分，可再乘 A。', '',
            '令 t 为逐样本的 gate 值乘上任务损失对 gate 的梯度。一阶取 |mean(-t)|，纯二阶取 mean(t²)/2，两项合用时先相加和求均值再取绝对值。A × gate 和各梯度评分在求阈值前归一化到平均为1，列范数对照按原方法使用原始尺度。乘 A 的各组先求乘积，再归一化。最终用同一评分公式，在训练终点计算。', '',
            '逐折验证与运行配置见 [score 对照证据](ranking-dashboard/score-ablation-512/README.md)。', '',
        ]
    args.dashboard.write_text('\n'.join(head)+'\n',encoding='utf-8')
    extra=sum(r['completed'] for r in ablation.values())
    print(f'Dashboard refreshed: {complete+sum(r["status"]=="complete" for r in ablation.values())}/{len(planned)} methods; {len(records)+extra}/{len(planned)*24} records.')


if __name__=='__main__':
    main()
