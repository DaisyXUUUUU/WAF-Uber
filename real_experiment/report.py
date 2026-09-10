"""Paired descriptive results and compact LaTeX table, never invented gains."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .common import ROOT,digest,write_json


def run(folder=None,allow_fixture=False):
    if folder is None:
        choices=sorted((ROOT/'results_real').glob('*_test'))
        if not choices:
            raise ValueError('No test run. Run validate and test first.')
        folder=choices[-1]
    from pathlib import Path
    folder=Path(folder)
    manifest=json.loads((folder/'run_manifest.json').read_text())
    fixture=manifest.get('source_kind')!='official_tlc_completed_trips'
    if fixture and not allow_fixture:
        raise ValueError('Synthetic fixture results cannot be exported as research results.')
    if manifest.get('mode')!='test' or not manifest.get('complete'):
        raise ValueError('Report requires a completed test run.')
    if digest(folder/'summary.csv')!=manifest['summary_sha256']:
        raise ValueError('Test summary checksum mismatch.')
    selected=json.loads((folder/'selection_used.json').read_text())['selected_weights']
    data=pd.read_csv(folder/'summary.csv')
    aggregates=[]; pairs=[]; intervals=[]
    for fleet in sorted(data.fleet.unique()):
        w=selected[str(fleet)]
        for name,weight in [('Baseline',0),('Selected lookahead',w)]:
            g=data.loc[(data.fleet==fleet)&(data.weight==weight)]
            served=g.served.sum()
            aggregates.append(dict(fleet=int(fleet),policy=name,weight=weight,service_rate=g.served.sum()/g.requests.sum(),
                mean_wait_min=(g.mean_wait_min.fillna(0)*g.served).sum()/served if served else None,
                idle_share=g.idle_share.mean(),passenger_share=g.passenger_share.mean(),
                mean_decision_ms=np.average(g.mean_decision_ms,weights=g.decision_epochs) if g.decision_epochs.sum() else 0))
        b=data.loc[(data.fleet==fleet)&(data.weight==0)].set_index(['date','seed'])
        a=data.loc[(data.fleet==fleet)&(data.weight==w)].set_index(['date','seed'])
        if set(a.index)!=set(b.index):
            raise ValueError('Unpaired test conditions.')
        for key in b.index:
            pairs.append(dict(fleet=int(fleet),date=key[0],seed=int(key[1]),
                service_rate_change_pp=100*(a.loc[key,'service_rate']-b.loc[key,'service_rate']),
                mean_wait_change_min=a.loc[key,'mean_wait_min']-b.loc[key,'mean_wait_min'],
                idle_share_change_pp=100*(a.loc[key,'idle_share']-b.loc[key,'idle_share'])))
        daily=pd.DataFrame([r for r in pairs if r['fleet']==fleet]).groupby('date').service_rate_change_pp.mean()
        # Cluster by date: initialization seeds are not independent demand days.
        rng=np.random.default_rng(817)
        boots=rng.choice(daily.to_numpy(),size=(2000,len(daily)),replace=True).mean(axis=1)
        lo,hi=np.quantile(boots,[.025,.975]) if len(daily)>=2 else (None,None)
        intervals.append(dict(fleet=int(fleet),test_days=len(daily),mean_daily_change_pp=float(daily.mean()),
            exploratory_day_bootstrap_low_pp=None if lo is None else float(lo),
            exploratory_day_bootstrap_high_pp=None if hi is None else float(hi)))
    pd.DataFrame(aggregates).to_csv(folder/'comparison.csv',index=False)
    pd.DataFrame(pairs).to_csv(folder/'paired_differences.csv',index=False)
    write_json(folder/'uncertainty.json',intervals)
    lines=[r'\begin{table}[htbp]',r'\centering',r'\small',
           r'\caption{Simulation on sampled TLC Uber trip records. Waiting time is conditional on service.}',
           r'\label{tab:dispatch_results}',r'\begin{tabular}{rlrrr}',r'\hline',
           r'Fleet & Policy & Served (\%) & Wait (min) & Idle (\%) \\',r'\hline']
    for r in aggregates:
        wait='--' if r['mean_wait_min'] is None else f"{r['mean_wait_min']:.2f}"
        label=r['policy'] if r['weight'] else ('Baseline' if r['policy']=='Baseline' else r'Selected ($\lambda=0$)')
        lines.append(f"{r['fleet']} & {label} & {100*r['service_rate']:.1f} & {wait} & {100*r['idle_share']:.1f} " + r'\\')
    lines += [r'\hline',r'\end{tabular}',r'\end{table}']
    if fixture:
        lines.insert(0,'% SYNTHETIC TEST FIXTURE -- NOT RESEARCH RESULTS')
    (folder/'results_table.tex').write_text('\n'.join(lines),encoding='utf-8')
    note=['# 实验结果说明','', '这些是公开已完成行程驱动的模拟，不是真实 Uber 的服务率。',
          '验证期选择可能得到零权重，此时两行相同是正确结果。',
          'comparison.csv 合并全部测试日和初始化；paired_differences.csv 是同日同种子的配对差值。',
          '不确定性按日期重采样，先对同日不同初始化求均值；默认仅三天，区间仅供探索，不能作强显著性结论。',
          '平均等待针对已服务乘客，不同策略服务集合可能不同。候选搜索仅改变司机，不改变基线选中的当前订单子集。','']
    for r in intervals:
        note.append(f"Fleet {r['fleet']}: mean daily service-rate change = {r['mean_daily_change_pp']:.3f} percentage points over {r['test_days']} test days.")
    (folder/'RESULTS_README.md').write_text('\n'.join(note),encoding='utf-8')
    print(pd.DataFrame(aggregates).to_string(index=False))
    print(f'LaTeX table and paired results saved in {folder}',flush=True)
