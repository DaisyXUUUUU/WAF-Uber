"""Exploratory waiting-limit sensitivity; reuse frozen inputs for 8/10/12 min.
Run: python sensitivity.py
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
import simulator as sim
from real_experiment import experiment, report
from real_experiment.common import ROOT, load_config, fingerprint, digest, utc_now, write_json

WAIT_LIMITS = (8, 10, 12)


def phase_manifest(folder, mode, c, data_report, shared_folder):
    folder.mkdir(parents=True)
    write_json(folder/'run_manifest.json', dict(mode=mode,complete=False,created_utc=utc_now(),
        config=c,config_hash=fingerprint(c),code_hash=experiment.code_hash(),
        sensitivity_code_sha256=digest(Path(__file__)),data_hash=fingerprint(data_report),
        source_kind=data_report['source']['kind'],shared_prepared_folder=str(shared_folder),
        design='Exploratory sensitivity added after inspecting primary 8-minute test results. All three conditions reported.'))


def finish(folder, rows):
    sim.write_csv(folder/'summary.csv',rows)
    manifest=json.loads((folder/'run_manifest.json').read_text())
    manifest.update(complete=True,summary_sha256=digest(folder/'summary.csv'))
    write_json(folder/'run_manifest.json',manifest)


def main():
    base=load_config()
    experiment.enable_fast_solvers()
    inputs=experiment.load_inputs(base)
    folder,data_report,historical,initial,travel=inputs
    parent=ROOT/'results_real'/('sensitivity_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    parent.mkdir(parents=True)
    protocol=dict(created_utc=utc_now(),wait_limits=list(WAIT_LIMITS),base_config=base,
        shared_prepared_folder=str(folder),shared_data_hash=fingerprint(data_report),
        travel_sha256=digest(folder/'travel_times.csv'),
        sensitivity_code_sha256=digest(Path(__file__)),
        selection_rule='Per fleet and wait limit: pooled validation service rate, then served-weighted wait, then smaller weight.',
        exploratory=True,complete=False,
        note='Only max_wait changes. Do not refit travel table or resample requests. Primary test has already been inspected; this is sensitivity, not a new untouched confirmation set.')
    write_json(parent/'sensitivity_manifest.json',protocol)
    variants={}; selections={}
    # Freeze every condition using validation before running any test condition.
    for wait in WAIT_LIMITS:
        c=deepcopy(base); c['max_wait_minutes']=wait; variants[wait]=c
        out=parent/f'wait_{wait}'/'validate'
        phase_manifest(out,'validate',c,data_report,folder)
        rows=[]
        for fleet in c['fleet_sizes']:
            for day in c['validation_dates']:
                for seed in c['initialization_seeds']:
                    for weight in c['weight_grid']:
                        rows.append(experiment.run_case(c,folder,historical,initial,travel,day,seed,fleet,weight,out))
                        sim.write_csv(out/'summary.csv',rows)
        selections[wait]=dict(selected_weights=experiment.choose_weights(rows,c),
            selected_on_dates=c['validation_dates'],config_hash=fingerprint(c),
            data_hash=fingerprint(data_report),code_hash=experiment.code_hash(),
            validation_summary_sha256=digest(out/'summary.csv'))
        write_json(out/'selection.json',selections[wait]); finish(out,rows)
        print(f'WAIT={wait}: frozen weights {selections[wait]["selected_weights"]}',flush=True)
    comparisons=[]; differences=[]; uncertainty=[]
    for wait,c in variants.items():
        out=parent/f'wait_{wait}'/'test'
        phase_manifest(out,'test',c,data_report,folder)
        write_json(out/'selection_used.json',selections[wait])
        rows=[]
        for fleet in c['fleet_sizes']:
            for day in c['test_dates']:
                for seed in c['initialization_seeds']:
                    for weight in sorted(set([0,selections[wait]['selected_weights'][str(fleet)]])):
                        rows.append(experiment.run_case(c,folder,historical,initial,travel,day,seed,fleet,weight,out))
                        sim.write_csv(out/'summary.csv',rows)
        finish(out,rows); report.run(out)
        for filename,target in [('comparison.csv',comparisons),('paired_differences.csv',differences)]:
            frame=pd.read_csv(out/filename); frame.insert(0,'max_wait_minutes',wait); target.append(frame)
        for item in json.loads((out/'uncertainty.json').read_text()):
            uncertainty.append(dict(max_wait_minutes=wait,**item))
    comp=pd.concat(comparisons,ignore_index=True)
    comp.to_csv(parent/'sensitivity_comparison.csv',index=False)
    pd.concat(differences,ignore_index=True).to_csv(parent/'sensitivity_paired_differences.csv',index=False)
    write_json(parent/'sensitivity_uncertainty.json',uncertainty)
    summary=[]
    for wait in WAIT_LIMITS:
        for fleet in base['fleet_sizes']:
            g=comp.loc[(comp.max_wait_minutes==wait)&(comp.fleet==fleet)].set_index('policy')
            b,a=g.loc['Baseline'],g.loc['Selected lookahead']
            summary.append(dict(max_wait_minutes=wait,fleet=fleet,selected_weight=float(a.weight),
                baseline_service_rate=float(b.service_rate),lookahead_service_rate=float(a.service_rate),
                service_gain_pp=100*float(a.service_rate-b.service_rate),
                wait_change_seconds=60*float(a.mean_wait_min-b.mean_wait_min),
                idle_change_pp=100*float(a.idle_share-b.idle_share)))
    table=pd.DataFrame(summary); table.to_csv(parent/'sensitivity_effects.csv',index=False)
    times=pd.read_csv(folder/'travel_times.csv')
    reach=[]
    for wait in WAIT_LIMITS:
        unreachable=[int(z) for z in base['zones'] if times.loc[times.destination==z,'minutes'].min()>wait]
        reach.append(dict(max_wait_minutes=wait,od_pairs=len(times),
            pairs_reachable_with_zero_assignment_delay=int((times.minutes<=wait).sum()),
            origins_unreachable_from_every_zone=unreachable))
    write_json(parent/'reachability.json',reach)
    lines=[r'\begin{table}[htbp]',r'\centering\small',
        r'\caption{Exploratory sensitivity to the maximum waiting time. Shared demand and travel estimates are held fixed.}',
        r'\label{tab:wait_sensitivity}',r'\begin{rrrrrr}',r'\end{rrrrrr}']
    # Standard LaTeX tabular, no extra package required.
    lines=lines[:-2]+[r'\begin{tabular}{rrrrrr}',r'\hline',
        r'$W_{\max}$ & Fleet & $\lambda$ & Base (\%) & Ahead (\%) & Gain (pp) \\',r'\hline']
    for r in summary:
        lines.append(f"{r['max_wait_minutes']} & {r['fleet']} & {r['selected_weight']:g} & "
                     f"{100*r['baseline_service_rate']:.1f} & {100*r['lookahead_service_rate']:.1f} & {r['service_gain_pp']:+.2f} " + r'\\')
    lines += [r'\hline',r'\end{tabular}',r'\end{table}']
    (parent/'sensitivity_table.tex').write_text('\n'.join(lines),encoding='utf-8')
    protocol.update(complete=True,effects_sha256=digest(parent/'sensitivity_effects.csv'))
    write_json(parent/'sensitivity_manifest.json',protocol)
    print('\nALL CONDITIONS (including zero/negative changes):\n'+table.to_string(index=False))
    print(f'\nOutputs: {parent}',flush=True)
    return parent


if __name__=='__main__':
    main()
