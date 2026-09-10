"""Common simulator, validation-only parameter selection, held-out testing."""
from __future__ import annotations
import csv
from dataclasses import replace
from datetime import datetime
import json
import math
import platform
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
import simulator as sim
from .common import ROOT, load_config, prepared_dir, digest, fingerprint, write_json, utc_now


def fast_min_cost_matching(costs):
    """Exact lexicographic objective via a proven finite dummy-cost bound."""
    n=len(costs)
    m=len(costs[0]) if n else 0
    if not n or not m:
        return {}
    largest=max((c for row in costs for c in row if c is not None),default=0)
    # One less unmatched row saves P, exceeding any possible n-edge cost change.
    penalty=(n+1)*(largest+1)
    matrix=np.full((n,m+n),penalty,dtype=float)
    matrix[:,:m]=2*penalty
    for i,row in enumerate(costs):
        for j,cost in enumerate(row):
            if cost is not None:
                matrix[i,j]=cost
    rows,cols=linear_sum_assignment(matrix)
    return {int(i):int(j) for i,j in zip(rows,cols) if j<m and costs[i][j] is not None}


def fast_cardinality(adjacency,right_size):
    if not adjacency or not right_size:
        return 0
    rows=[]; cols=[]
    for i,neighbors in enumerate(adjacency):
        rows.extend([i]*len(neighbors)); cols.extend(neighbors)
    if not rows:
        return 0
    graph=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(adjacency),right_size))
    return int(np.count_nonzero(maximum_bipartite_matching(graph,perm_type='column')>=0))


def enable_fast_solvers():
    # Changes solver implementation, not either policy's objective/constraints.
    sim.min_cost_maximum_matching=fast_min_cost_matching
    sim.maximum_matching=fast_cardinality


class EmpiricalTravel:
    def __init__(self,table):
        self.values={(int(r.origin),int(r.destination)):float(r.minutes) for r in table.itertuples()}
    def minutes(self,origin,destination,departure):
        # Fixed pre-test OD table; test trip durations are never used here.
        return self.values[origin,destination]


def read_requests(path):
    with Path(path).open(newline='') as f:
        return [sim.Request(int(r['id']),float(r['arrival']),int(r['origin']),int(r['destination'])) for r in csv.DictReader(f)]


def code_hash():
    files=[ROOT/'simulator.py']+sorted((ROOT/'real_experiment').glob('*.py'))
    return fingerprint({p.name:digest(p) for p in files})


def load_inputs(c,folder=None,allow_fixture=False):
    folder=Path(folder) if folder else prepared_dir(c)
    report=json.loads((folder/'data_report.json').read_text())
    if report['source']['kind']!='official_tlc_completed_trips' and not allow_fixture:
        raise ValueError('Fixture data cannot be run through the official experiment command.')
    for name,sha in report['artifact_hashes'].items():
        if digest(folder/name)!=sha:
            raise ValueError(f'Prepared artifact changed: {name}')
    from .common import processing_signature
    if report['processing_signature']!=processing_signature(c):
        raise ValueError('Data processing config changed; prepare again.')
    historical=[read_requests(folder/f'requests_{d}.csv') for d in c['history_dates']]
    weights=json.loads((folder/'initial_distribution.json').read_text())
    travel=EmpiricalTravel(pd.read_csv(folder/'travel_times.csv'))
    for day in c['history_dates']+c['validation_dates']+c['test_dates']:
        with (folder/f'requests_{day}.csv').open() as handle:
            count=sum(1 for _ in handle)-1
        if count>c['max_requests_per_day']:
            raise ValueError(f'{day} has {count} requests, above the configured pilot guard. Explicitly revise scale and prepare before proceeding; no hidden truncation.')
    return folder,report,historical,weights,travel


def new_run(mode,c,report):
    name=datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+mode
    out=ROOT/'results_real'/name
    out.mkdir(parents=True)
    import scipy
    write_json(out/'run_manifest.json',dict(mode=mode,created_utc=utc_now(),config=c,
        config_hash=fingerprint(c),code_hash=code_hash(),data_hash=fingerprint(report),
        source_kind=report['source']['kind'],python=sys.version,platform=platform.platform(),
        versions=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
        note='Simulation using sampled completed-trip records; not actual Uber service performance.'))
    return out


def run_case(c,folder,historical,initial_weights,travel,date,seed,fleet,weight,out):
    cfg=sim.Config(duration=c['duration_minutes'],fleet=fleet,delta=c['matching_interval_minutes'],
        max_wait=c['max_wait_minutes'],horizon=c['horizon_minutes'],scenarios=c['scenario_count'],
        weight=weight,search_passes=c['search_passes'],candidate_budget=c['candidate_budget'],
        history_days=len(historical))
    stream=read_requests(folder/f'requests_{date}.csv')
    rng=random.Random(seed*100000+int(date.replace('-','')))
    initial=rng.choices(c['zones'],weights=[initial_weights[str(z)] for z in c['zones']],k=fleet)
    history=sim.HistoricalScenarios(historical,cfg)
    policy='baseline' if weight==0 else 'lookahead'
    diagnostics=[]
    metrics,logs=sim.simulate(policy,stream,initial,history,cfg,travel,diagnostics)
    metrics=dict(date=date,seed=seed,fleet=fleet,weight=weight,**metrics)
    key=f'{date}_seed{seed}_fleet{fleet}_w{weight:g}'
    sim.write_csv(out/f'{key}_trips.csv',logs)
    sim.write_csv(out/f'{key}_decisions.csv',diagnostics)
    write_json(out/f'{key}_initial.json',initial)
    print(f"{date} seed={seed} fleet={fleet} weight={weight:g}: "
          f"served={metrics['served']}/{metrics['requests']} "
          f"changed={metrics['changed_from_same_state_baseline']} "
          f"mean decision={metrics['mean_decision_ms']:.1f} ms",flush=True)
    return metrics


def choose_weights(rows,c):
    result={}
    for fleet in c['fleet_sizes']:
        candidates=[]
        for w in c['weight_grid']:
            group=[r for r in rows if r['fleet']==fleet and r['weight']==w]
            served=sum(r['served'] for r in group)
            requests=sum(r['requests'] for r in group)
            wait=sum((r['mean_wait_min'] or 0)*r['served'] for r in group)/served if served else math.inf
            candidates.append((-(served/requests),wait,w))
        result[str(fleet)]=min(candidates)[2]
    return result


def run(c,mode,selection_path=None):
    enable_fast_solvers()
    folder,report,historical,initial_weights,travel=load_inputs(c)
    if mode=='test':
        if selection_path is None:
            matches=sorted((ROOT/'results_real').glob('*_validate/selection.json'))
            matches=[p for p in matches if json.loads(p.read_text())['config_hash']==fingerprint(c)]
            if not matches:
                raise ValueError('Run validation first; no compatible frozen selection exists.')
            selection_path=matches[-1]
        selection_path=Path(selection_path)
        selection=json.loads(selection_path.read_text())
        for key,expected in [('config_hash',fingerprint(c)),('code_hash',code_hash()),('data_hash',fingerprint(report))]:
            if selection[key]!=expected:
                raise ValueError(f'{key} changed after validation; rerun validation.')
        if selection['validation_summary_sha256']!=digest(selection_path.parent/'summary.csv'):
            raise ValueError('Validation summary changed after selection.')
    out=new_run(mode,c,report)
    rows=[]
    dates=c['test_dates'] if mode=='test' else c['validation_dates']
    fleets=c['fleet_sizes']
    seeds=c['initialization_seeds']
    if mode=='smoke':
        dates, fleets, seeds = dates[:1],fleets[:1],seeds[:1]
    for fleet in fleets:
        weights=c['weight_grid'] if mode=='validate' else [0,next((w for w in c['weight_grid'] if w>0),0)] if mode=='smoke' else sorted(set([0,selection['selected_weights'][str(fleet)]]))
        for day in dates:
            for seed in seeds:
                for weight in weights:
                    rows.append(run_case(c,folder,historical,initial_weights,travel,day,seed,fleet,weight,out))
                    sim.write_csv(out/'summary.csv',rows)
    if mode=='validate':
        write_json(out/'selection.json',dict(config_hash=fingerprint(c),code_hash=code_hash(),data_hash=fingerprint(report),
            selected_weights=choose_weights(rows,c),validation_summary_sha256=digest(out/'summary.csv'),
            rule='Maximize pooled validation service rate; tie: smaller served-weighted mean wait, then smaller weight.',
            selected_on_dates=dates,created_utc=utc_now()))
        print('Frozen validation selection:',choose_weights(rows,c),flush=True)
    elif mode=='test':
        write_json(out/'selection_used.json',selection)
    manifest=json.loads((out/'run_manifest.json').read_text())
    manifest['complete']=True
    manifest['summary_sha256']=digest(out/'summary.csv')
    write_json(out/'run_manifest.json',manifest)
    print(f'Run complete: {out}',flush=True)
    return out
