"""Download official TLC files; stream to disk and record checksums."""
from __future__ import annotations
import json
import time
import urllib.request
from pathlib import Path
from .common import ROOT, LOOKUP_URL, OFFICIAL_PAGE, DICTIONARY, digest, write_json, utc_now


def parquet_footer_ok(path):
    with Path(path).open('rb') as f:
        if f.read(4) != b'PAR1':
            return False
        f.seek(-4, 2)
        return f.read(4) == b'PAR1'


def download_file(url, path):
    path = Path(path)
    meta = path.with_suffix(path.suffix + '.source.json')
    if path.exists():
        if not meta.exists():
            raise RuntimeError(f'{path} exists without provenance. Move it aside or supply its source metadata.')
        record = json.loads(meta.read_text())
        if record['url'] != url or digest(path) != record['sha256']:
            raise RuntimeError(f'Cached source changed: {path}')
        print(f'Using verified cache: {path.name}', flush=True)
        return record
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    # A partial attempt is replaced on retry; completed cache is never overwritten.
    request = urllib.request.Request(url, headers={'User-Agent':'WAF-Uber-Research/1.0'})
    print(f'Downloading official source: {url}', flush=True)
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as out:
            expected = int(response.headers.get('Content-Length', 0))
            received, last_print = 0, time.monotonic()
            while True:
                block = response.read(4*1024*1024)
                if not block:
                    break
                out.write(block)
                received += len(block)
                if time.monotonic()-last_print >= 3:
                    print(f'  {received/1024**2:.0f} MB / {expected/1024**2:.0f} MB', flush=True)
                    last_print = time.monotonic()
            if not received or (expected and received != expected):
                raise RuntimeError('Incomplete download; rerun to retry.')
            etag = response.headers.get('ETag')
        if path.suffix == '.parquet' and not parquet_footer_ok(temporary):
            raise RuntimeError('Downloaded file is not a complete Parquet file.')
        record = dict(url=url, retrieved_utc=utc_now(), bytes=received, etag=etag,
                      sha256=digest(temporary), official_page=OFFICIAL_PAGE, dictionary=DICTIONARY)
        temporary.replace(path)
        write_json(meta, record)
        return record
    except Exception as e:
        raise RuntimeError(f'Download failed for {path.name}: {e}. Rerun when connected; do not use synthetic replacements.') from e


def run(c):
    raw = ROOT/'data'/'raw'
    download_file(LOOKUP_URL, raw/'taxi_zone_lookup.csv')
    name = f"fhvhv_tripdata_{c['month']}.parquet"
    download_file(f'https://d37ci6vzurychx.cloudfront.net/trip-data/{name}', raw/name)
