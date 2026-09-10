"""Shared configuration, provenance, and immutable output helpers."""
from __future__ import annotations
import hashlib
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_PAGE = 'https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page'
DICTIONARY = 'https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_hvfhs.pdf'
LOOKUP_URL = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def load_config(path=None):
    path = Path(path).resolve() if path else ROOT / 'experiment_config.json'
    c = json.loads(path.read_text(encoding='utf-8'))
    groups = [c[k] for k in ('history_dates','validation_dates','test_dates')]
    if any(not g or len(g) != len(set(g)) or g != sorted(g) for g in groups):
        raise ValueError('Date lists must be nonempty, unique, and sorted.')
    dates = [d for g in groups for d in g]
    if len(set(dates)) != len(dates) or not max(groups[0]) < min(groups[1]) or not max(groups[1]) < min(groups[2]):
        raise ValueError('History, validation, test must be disjoint and strictly chronological.')
    for d in dates:
        date.fromisoformat(d)
        if d[:7] != c['month']:
            raise ValueError('This first version supports one month only.')
    if len(set(c['zones'])) != len(c['zones']) or len(c['zones']) < 2:
        raise ValueError('Use at least two distinct zone IDs.')
    numeric = ['sample_fraction','duration_minutes','horizon_minutes','matching_interval_minutes',
               'max_wait_minutes','candidate_budget','search_passes','scenario_count',
               'minimum_od_observations','max_requests_per_day']
    if any(not math.isfinite(c[k]) or c[k] <= 0 for k in numeric):
        raise ValueError('Configuration sizes and time limits must be finite and positive.')
    if not 0 < c['sample_fraction'] <= 1 or not 0 <= c['start_hour'] < 24:
        raise ValueError('Invalid sampling fraction or start hour.')
    if c['start_hour']*60 + c['duration_minutes'] + c['max_wait_minutes'] + 180 > 1440:
        raise ValueError('Choose a window whose completion tail stays in the same day.')
    if c['scenario_count'] > len(c['history_dates']):
        raise ValueError('Too few historical days for scenario count.')
    if not c['weight_grid'] or 0 not in c['weight_grid'] or any(not math.isfinite(w) or w < 0 for w in c['weight_grid']):
        raise ValueError('Weight grid must include zero and finite nonnegative weights.')
    if not c['fleet_sizes'] or any(int(n)!=n or n<1 for n in c['fleet_sizes']):
        raise ValueError('Invalid fleet sizes.')
    if not c['initialization_seeds']:
        raise ValueError('At least one initialization seed is required.')
    return c


def processing_signature(c):
    keys = ('month','zones','start_hour','duration_minutes','history_dates','validation_dates',
            'test_dates','sample_fraction','sampling_seed','max_wait_minutes','minimum_od_observations')
    return fingerprint({k:c[k] for k in keys})


def prepared_dir(c):
    return ROOT / 'data' / 'processed' / processing_signature(c)[:12]


def utc_now():
    return datetime.now(timezone.utc).isoformat()
