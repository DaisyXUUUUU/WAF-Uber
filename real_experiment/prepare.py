"""Streaming source read, explicit filtering, train-only travel estimation."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
import numpy as np
import pandas as pd
from .common import ROOT, digest, prepared_dir, processing_signature, write_json, utc_now

COLUMNS = ['hvfhs_license_num','request_datetime','pickup_datetime','dropoff_datetime',
           'PULocationID','DOLocationID','shared_request_flag','shared_match_flag']


def clean_chunk(frame, c, audit):
    f = frame.copy()
    audit['source_rows_scanned'] += len(f)
    def keep(mask, name):
        nonlocal f
        mask = mask.fillna(False)
        audit[name] += int((~mask).sum())
        f = f.loc[mask].copy()
    keep(f['hvfhs_license_num'].eq('HV0003'), 'removed_not_uber')
    keep(f['shared_request_flag'].eq('N') & f['shared_match_flag'].eq('N'), 'removed_pooled_or_unknown')
    for key in ['request_datetime','pickup_datetime','dropoff_datetime']:
        f[key] = pd.to_datetime(f[key], errors='coerce')
        if f[key].dt.tz is not None:
            # TLC times are NYC local; do not silently interpret UTC as local.
            raise ValueError('Unexpected timezone-aware timestamps; inspect source schema.')
    keep(f[['request_datetime','pickup_datetime','dropoff_datetime']].notna().all(axis=1), 'removed_missing_timestamps')
    f['observed_trip_min'] = (f.dropoff_datetime-f.pickup_datetime).dt.total_seconds()/60
    observed_wait = (f.pickup_datetime-f.request_datetime).dt.total_seconds()/60
    keep(observed_wait.between(0,180) & f.observed_trip_min.between(0.5,180), 'removed_invalid_times')
    keep(f.PULocationID.isin(c['zones']) & f.DOLocationID.isin(c['zones']), 'removed_outside_closed_area')
    f['date'] = f.request_datetime.dt.strftime('%Y-%m-%d')
    selected_dates = c['history_dates']+c['validation_dates']+c['test_dates']
    keep(f['date'].isin(selected_dates), 'removed_other_dates')
    return f


def fit_travel(history, c):
    """Static empirical OD medians. Fallback hierarchy is recorded per pair."""
    zones = c['zones']
    if history.empty:
        raise ValueError('No historical trips for travel estimation.')
    groups = history.groupby(['PULocationID','DOLocationID']).observed_trip_min
    medians, counts = groups.median().to_dict(), groups.size().to_dict()
    n = len(zones)
    paths = np.full((n,n), np.inf)
    np.fill_diagonal(paths, 0)
    for i,a in enumerate(zones):
        for j,b in enumerate(zones):
            if a != b and counts.get((a,b),0) >= c['minimum_od_observations']:
                paths[i,j] = medians[a,b]
    for k in range(n):
        paths = np.minimum(paths, paths[:,k,None] + paths[None,k,:])
    intra = history.loc[history.PULocationID.eq(history.DOLocationID), 'observed_trip_min']
    inter = history.loc[history.PULocationID.ne(history.DOLocationID), 'observed_trip_min']
    if inter.empty:
        raise ValueError('No inter-zone training trips; choose a data-rich region before testing.')
    rows=[]
    for i,a in enumerate(zones):
        for j,b in enumerate(zones):
            count=counts.get((a,b),0)
            if count >= c['minimum_od_observations']:
                value,source=medians[a,b],'observed_od_median'
            elif a == b:
                value,source=(float(intra.median()),'pooled_intra_median') if not intra.empty else (3.0,'assumed_intra_3min')
            elif math.isfinite(paths[i,j]):
                value,source=float(paths[i,j]),'historical_graph_path'
            else:
                value,source=float(inter.median()),'pooled_inter_median'
            rows.append(dict(origin=a,destination=b,minutes=max(.5,float(value)),observations=int(count),source=source))
    return pd.DataFrame(rows)


def prepare_frames(frames, c, out, source_info, lookup=None):
    """Shared production/fixture path. Fixtures must identify themselves."""
    if out.exists():
        raise FileExistsError(f'{out} already exists. Existing prepared data will not be overwritten.')
    audit=Counter()
    selected=[]
    offset=0
    for batch_number, frame in enumerate(frames,1):
        missing=set(COLUMNS)-set(frame.columns)
        if missing:
            raise ValueError(f'Missing required TLC columns: {sorted(missing)}')
        frame=frame.copy()
        frame['source_row']=np.arange(offset,offset+len(frame),dtype=np.int64)
        offset+=len(frame)
        clean=clean_chunk(frame,c,audit)
        if not clean.empty:
            selected.append(clean)
        if batch_number%10==0:
            print(f'  Scanned {offset:,} source rows; retained {sum(len(x) for x in selected):,}',flush=True)
    if not selected:
        raise ValueError('No records survived the stated filters.')
    f=pd.concat(selected,ignore_index=True)
    duplicate_keys=['request_datetime','pickup_datetime','dropoff_datetime','PULocationID','DOLocationID']
    # No unique trip ID: identical signatures can be legitimate simultaneous rides.
    # Count them but do not silently remove them.
    audit['duplicate_signature_rows_retained']=int(f.duplicated(duplicate_keys).sum())
    start=c['start_hour']*60
    request_minutes=(f.request_datetime-f.request_datetime.dt.normalize()).dt.total_seconds()/60
    pickup_minutes=(f.pickup_datetime-f.pickup_datetime.dt.normalize()).dt.total_seconds()/60
    tail=start+c['duration_minutes']+c['max_wait_minutes']+180
    train=f.loc[f.date.isin(c['history_dates']) & pickup_minutes.between(start,tail)].copy()
    travel=fit_travel(train,c)
    f['arrival']=request_minutes-start
    window=f.loc[(f.arrival>=0)&(f.arrival<c['duration_minutes'])].copy()
    fraction=c['sample_fraction']
    def retained(row):
        key=f"{c['sampling_seed']}:{int(row)}".encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:8],'big')/2**64 < fraction
    sampled=window.loc[window.source_row.map(retained)].copy()
    daily=[]
    dates=c['history_dates']+c['validation_dates']+c['test_dates']
    for d in dates:
        group=sampled.loc[sampled.date.eq(d)]
        if group.empty:
            raise ValueError(f'No sampled requests on {d}. Inspect coverage or predeclare a larger sample fraction.')
        daily.append(dict(date=d,split='history' if d in c['history_dates'] else 'validation' if d in c['validation_dates'] else 'test',
                          raw_window_requests=int(window.date.eq(d).sum()),sampled_requests=len(group)))
    initial=sampled.loc[sampled.date.isin(c['history_dates'])].groupby('PULocationID').size()
    if initial.empty:
        raise ValueError('No historical requests for initial fleet distribution.')
    out.mkdir(parents=True)
    for d in dates:
        group=sampled.loc[sampled.date.eq(d)].sort_values(['arrival','source_row'])
        group[['source_row','arrival','PULocationID','DOLocationID']].rename(columns={
            'source_row':'id','PULocationID':'origin','DOLocationID':'destination'}).to_csv(out/f'requests_{d}.csv',index=False)
    travel.to_csv(out/'travel_times.csv',index=False)
    pd.DataFrame(daily).to_csv(out/'daily_counts.csv',index=False)
    if lookup is not None:
        lookup.loc[lookup.LocationID.isin(c['zones'])].to_csv(out/'selected_zones.csv',index=False)
    write_json(out/'initial_distribution.json',{str(z):int(initial.get(z,0)) for z in c['zones']})
    report=dict(created_utc=utc_now(),processing_signature=processing_signature(c),source=source_info,
        config=c,filter_audit=dict(audit),travel_training_rows=len(train),
        travel_sources=travel.source.value_counts().to_dict(),sampling='deterministic hash Bernoulli thinning, same fraction for every split',
        initial_fleet='all initially idle; locations sampled from history pickup distribution',
        limitations=['Completed-trip records omit full unmet/cancelled demand.',
                     '10% sample if default config is used; fixed simulated fleet is not actual Uber supply.',
                     'Passenger trip medians proxy empty pickup travel; unobserved OD fallbacks are assumptions.',
                     'Static time table, no congestion feedback, no online/offline drivers.',
                     'Exact timestamp duplicates retained because public records have no unique trip ID.'])
    report['artifact_hashes']={p.name:digest(p) for p in sorted(out.glob('*')) if p.is_file()}
    write_json(out/'data_report.json',report)
    print(f'Prepared data saved: {out}',flush=True)
    return report


def run(c):
    out=prepared_dir(c)
    raw=ROOT/'data'/'raw'/f"fhvhv_tripdata_{c['month']}.parquet"
    if out.exists():
        report=out/'data_report.json'
        if not report.exists():
            raise RuntimeError(f'Incomplete prepared folder: {out}; move it aside and retry.')
        print(f'Prepared data already exists: {out}',flush=True)
        return
    if not raw.exists():
        raise FileNotFoundError('Run the download step first.')
    import pyarrow.parquet as pq
    meta=json.loads(raw.with_suffix('.parquet.source.json').read_text())
    if digest(raw)!=meta['sha256']:
        raise RuntimeError('Raw checksum mismatch.')
    parquet=pq.ParquetFile(raw)
    missing=set(COLUMNS)-set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f'TLC schema missing: {sorted(missing)}')
    lookup=pd.read_csv(ROOT/'data'/'raw'/'taxi_zone_lookup.csv')
    if not set(c['zones'])<=set(lookup.LocationID):
        raise ValueError('Unknown zone IDs in configuration.')
    print(lookup.loc[lookup.LocationID.isin(c['zones']),['LocationID','Borough','Zone']].to_string(index=False))
    frames=(batch.to_pandas() for batch in parquet.iter_batches(batch_size=200000,columns=COLUMNS))
    prepare_frames(frames,c,out,{'kind':'official_tlc_completed_trips','download':meta},lookup)
