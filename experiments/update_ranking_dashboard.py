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
LABELS={'raw_snapshot_column_norm':'固定128配方扩维：列范数（ρ50）',
 'raw_snapshot_taylor_first':'我们的方案：一阶（ρ50）',
 'raw_snapshot_taylor_fisher':'我们的方案：一阶＋Fisher（ρ50）',
 'public_512_defaults_rho200':'原仓库512默认（ρ200）',
 'effective_gate_oct9_reference':'10月有效 gate 报告参照',
 'lassoflexnet_reimpl_sequential_feasible':'LassoFlexNet 本地文本变体',
 'mrmr_fullrank':'mRMR 原生完整排名','localized_lasso':'Localized Lasso',
 'lassonet':'LassoNet','rf':'Random Forest','treeshap':'TreeSHAP','random':'随机排名控制'}


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
    def percent(value):
        return str(Decimal(f'{100*value:.10f}').quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
    now=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')
    head=['# Deep Regularization 实验看板','',f'更新时间：{now}（北京时间）。这是当前唯一汇总入口；旧日期记录只作方法与证据追溯。','',
      f'512维全量任务：**{complete}/{len(planned)}条方法线完成**，已核验{len(records)}/{len(planned)*24}个逐折结果。范围为 XOR、Ring、Ring+XOR、Ring+XOR+Sum，各1000样本、普通六折，只做ranking，不调参或下游重训。全量指本轮声明的方法清单，不扩到原CLI的attr-t、10模型bootstrap attr-b、普通NN预测或DAG2000维。','',
      '## 当前判断','',
      ('四任务已完成的原仓库默认平均为'+percent(summary['public_512_defaults_rho200']['macro_recall'])+'%，一阶为'+percent(summary['raw_snapshot_taylor_first']['macro_recall'])+'%，一阶＋Fisher为'+percent(summary['raw_snapshot_taylor_fisher']['macro_recall'])+'%。这批固定数据与配方下，换评分没有改善；不能认为二阶总不会差。' if all(summary[m]['status']=='complete' for m in ['public_512_defaults_rho200','raw_snapshot_taylor_first','raw_snapshot_taylor_fisher']) else '完整结论等待对应四任务齐全，不使用未齐方法的部分均值作总排名。'),'',
      ('全部方法已跑齐。本轮四任务均值最高为 '+LABELS.get(max(summary,key=lambda m:summary[m]['macro_recall']),max(summary,key=lambda m:summary[m]['macro_recall']))+'，'+percent(max(s['macro_recall'] for s in summary.values()))+'%；结论只适用于本轮固定设置，不替代多种子或充分调参后的比较。' if complete==len(planned) else '其余方法尚未齐全，暂不宣布全表最优方法。'),'',
      '## 怎么读结果','',
      '每列是 Top-k 相关特征恢复率：k依次为2、2、4、6。100%表示找回该任务全部真实特征，0%表示一个也没找回。它不是分类准确率。四任务均完成后才计算 Macro；缺折不当作0，也不生成部分任务总排名。','',
      '此前的三方法小表只有 XOR/Ring：有效 gate 擅长 Ring、XOR失败；现有一阶＋Fisher在XOR成功、Ring较弱。那张两任务均值不能代表四任务总体，也不能证明只改score有收益，因为骨干、正则求解器和训练预算不同。','',
      '## 512维当前完整矩阵','',
      '| 方法 | XOR | Ring | Ring+XOR | Ring+XOR+Sum | 四任务 Macro | 完成状态 |',
      '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    priority=['public_512_defaults_rho200','effective_gate_oct9_reference','raw_snapshot_column_norm','raw_snapshot_taylor_first','raw_snapshot_taylor_fisher']
    order=priority+sorted([m for m in planned if m not in priority],key=lambda m:(summary[m]['status']!='complete',-(summary[m]['macro_recall'] or 0),m))
    for method in order:
        s=summary[method]
        values=[percent(s['tasks'][t]['recall_at_k']) if s['tasks'][t]['n']==6 else f"—（{s['tasks'][t]['n']}/6）" for t in TASKS]
        macro=percent(s['macro_recall']) if s['macro_recall'] is not None else '—'
        status='完成' if s['status']=='complete' else ('待首折/排队' if s['completed']==0 else '未齐')
        if s['failed_attempts']:status+=f"；{s['failed_attempts']}次失败记录"
        head.append('| '+' | '.join([LABELS.get(method,method),*values,macro,f"{s['completed']}/24 {status}"])+' |')
    head += ['', '## 原实验仓库到底是什么基线','',
      '固定来源为 [feature-selection@555ba11（2026-07-10）](https://github.com/DreamSoul-AI/Deep-Regularization/tree/555ba11e8872bd6618cd721bd3b08f7128db9e6b)。main/dev根模型与研究分支不同，9月模型以证据快照保留；未发现10月9日或11月A—L的新源码分支。','',
      '- 原公开512默认：Mish32×32、500轮/120 warm-up、C=.05、lr=.005、初始ρ200、batch64、WD=.001、feature dropout=.6。维数分段ρ见 [原包装器](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/src/admm_input_group_wrapper.py#L1338)。本轮新增该行，直接执行固定Git对象内的原模型/训练函数','- 先前公开快照三条线是固定128维配方扩到512，初始ρ50；旧结果保留，不能冒充原512默认','- 10月有效gate参照：共享线性/平方父gate、ReLU32×32、120轮、全局C=.003172、ρ=.18857、lr=.002084；canonical prox，零warm/WD/dropout。按报告恢复，未取得历史原驱动','- 我们的一阶和一阶＋Fisher保持原已实现方法：在raw gate路径上更新score scaling，最终仍按训练后gate排名；本轮没有移植到有效gate骨干或改近端目标','',
      'ρ50和ρ200本轮四任务的恢复率恰好一致，但不是重复结果：24折中只有13折的Top-k集合相同，分数向量全部不同，见 [逐折核对](ranking-dashboard/512/rho-reference-comparison.json)。','',
      '原清单含RF、TreeSHAP、Relief、FSNet、E2E-FS、mRMR、MI、LassoNet、CAE、两种CancelOut、DeepPINK及10种归因；普通NN是归因骨干。STG、随机控制、XGBoost、Localized Lasso、LassoFlexNet及论文补充选择器也纳入本轮。LassoFlexNet是明确标记的本地文本可行变体，官方代码未取得，不另造官方成绩。','',
      '## 原commit保存的历史512维结果','',
      '下表来自原commit携带的均值文件，仅作历史背景；缺少执行版本/环境和逐折分数，不能混入本轮配对成绩。','',
      '| 历史方法 | XOR | Ring | Ring+XOR | Ring+XOR+Sum | Macro |',
      '| --- | ---: | ---: | ---: | ---: | ---: |',
      '| RF | 16.67 | 100 | 75 | 63.89 | 63.89 |',
      '| mRMR | 0 | 100 | 70.83 | 55.56 | 56.60 |',
      '| TreeSHAP | 0 | 100 | 58.33 | 50 | 52.08 |',
      '| LassoNet | 100 | 0 | 33.33 | 47.22 | 45.14 |',
      '| MI | 0 | 100 | 33.33 | 13.89 | 36.81 |',
      '| Relief | 16.67 | 0 | 0 | 0 | 4.17 |','',
      '来源例：[LassoNet XOR原始结果文件](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/Feature-Selection-Benchmark/results/lassonet-xor-1000.txt)。历史mRMR Ring100%与本轮Pearson估计器不同，执行来源尚未闭合，不能直接说方法退化。','',
      '## 比较边界与当前证据','',
      ('本次未显式绑定的旧LassoNet记录，原生构造器实际自动使用CUDA；旧日志的CPU是请求值，Windows上的空CUDA可见性未禁用GPU。当前汇总已更正设备元数据，原分数不改，[设备审计](ranking-dashboard/512/lassonet-device-audit.json)保留依据。未来入口显式传入设备。' if any(r.get('device_correction') for r in records.values()) else 'LassoNet设备按逐折显式记录，不从空CUDA可见性推断。'),'',
      'Attention Maps的512维物理batch512曾占用约26GB内存，已保留停止记录并改为GPU上16样本分块累积：有效batch仍为512，每批仍只更新一次，模型/学习率不变。实际上游FT模型在关闭dropout时的完整/分块梯度一致；dropout开启会改变随机数次序，不声称逐位同轨迹。Localized Lasso仅按不同任务/折分片并行，求解目标和数值门限不变。','',
      '本轮baseline统一训练折z-score；原benchmark先2*x-1、各wrapper自行预处理。Relief保留原生符号；原benchmark统一取abs。mRMR用原生完整贪心顺序，原k集合包装器内部随机分数只留作诊断。LassoNet排名来自路径 feature_importances_，不是最低val_loss checkpoint的β；best-val模型只用于预测。见 [原LassoNet调用](https://github.com/DreamSoul-AI/Deep-Regularization/blob/555ba11e8872bd6618cd721bd3b08f7128db9e6b/src/core.py#L175)。','',
      '所有方法复用相同保存的数据/划分/支持真值。六折训练集重叠且来自一个生成实例，不能冒充多个独立数据生成种子。不同结构与预算是方法间比较，不能作为只换评分的消融。真实相关坐标不参与训练或选正则参数；评价使用真值k，FSNet、CAE和E2E-FS等原生预算选择器也预先给定相同k。128维和512维并非只追加噪声列的嵌套数据，跨维数差异不能全部归因于新增噪声。','',
      '[逐折分数与机器汇总](ranking-dashboard/512/summary.json)、[原始分数](ranking-dashboard/512/verified_results.jsonl)、[数据身份](ranking-dashboard/512/cases.json)、[失败记录](ranking-dashboard/512/failures.json)。标准库核验：`python experiments/verify_baseline_evidence.py docs/ranking-dashboard/512`。刷新看板：`python experiments/update_ranking_dashboard.py`。','',
      '128维已完成表另留在 [既有证据](<研究记录 Research Record/20261128-baseline-repairs/128/table.md>)，不与512维成绩合并。方法推导与旧运行细节从 [研究计划](<研究计划 Research Plan Deep Regularization.md>) 查阅。']
    args.dashboard.write_text('\n'.join(head)+'\n',encoding='utf-8')
    print(f'Dashboard refreshed: {complete}/{len(planned)} methods; {len(records)}/{len(planned)*24} records.')


if __name__=='__main__':
    main()
