"""Synthetic fixtures test data plumbing, not research effectiveness."""
from __future__ import annotations
import contextlib
from copy import deepcopy
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import simulator as sim
from real_experiment import experiment, report
from real_experiment.common import ROOT, load_config, digest, write_json
from real_experiment.prepare import clean_chunk, fit_travel, prepare_frames
from collections import Counter


def fixture_config():
    c=load_config()
    c.update(history_dates=['2024-01-02','2024-01-03'],validation_dates=['2024-01-16'],
             test_dates=['2024-01-22','2024-01-23'],zones=[161,162,163],
             sample_fraction=1.0,duration_minutes=30,fleet_sizes=[2],initialization_seeds=[7],
             weight_grid=[0,3],scenario_count=2,candidate_budget=4)
    return c


def fixture_frame(c,test_duration=5):
    rows=[]
    for day in c['history_dates']+c['validation_dates']+c['test_dates']:
        for i in range(18):
            arrival=pd.Timestamp(f'{day} 17:00:00')+pd.Timedelta(minutes=i)
            pickup=arrival+pd.Timedelta(minutes=2)
            duration=test_duration if day in c['test_dates'] else 3+i%4
            rows.append(dict(hvfhs_license_num='HV0003',request_datetime=arrival,
                pickup_datetime=pickup,dropoff_datetime=pickup+pd.Timedelta(minutes=duration),
                PULocationID=c['zones'][i%3],DOLocationID=c['zones'][(i//3)%3],
                shared_request_flag='N',shared_match_flag='N'))
    return pd.DataFrame(rows)


class RealPipelineTests(unittest.TestCase):
    def test_chronological_config_guard(self):
        c=fixture_config()
        c['test_dates']=c['history_dates']
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as d:
            p=Path(d)/'bad.json'; write_json(p,c)
            with self.assertRaises(ValueError): load_config(p)

    def test_filtering_does_not_use_model_wait_limit(self):
        c=fixture_config(); frame=fixture_frame(c).iloc[:4].copy()
        frame.loc[0,'hvfhs_license_num']='HV0005'
        frame.loc[1,'shared_match_flag']='Y'
        # A long observed wait is retained; do not select only historical successes under Wmax.
        frame.loc[2,'pickup_datetime']=frame.loc[2,'request_datetime']+pd.Timedelta(minutes=20)
        frame.loc[2,'dropoff_datetime']=frame.loc[2,'pickup_datetime']+pd.Timedelta(minutes=5)
        audit=Counter(); cleaned=clean_chunk(frame,c,audit)
        self.assertEqual(len(cleaned),2)
        self.assertEqual(audit['removed_not_uber'],1)
        self.assertEqual(audit['removed_pooled_or_unknown'],1)

    def test_fast_solver_matches_reference(self):
        rng=random.Random(72)
        for _ in range(100):
            costs=[[None if rng.random()<.35 else rng.randint(1,50)/3 for _ in range(5)] for _ in range(4)]
            a=sim.min_cost_maximum_matching(costs); b=experiment.fast_min_cost_matching(costs)
            self.assertEqual(len(a),len(b))
            self.assertAlmostEqual(sum(costs[i][j] for i,j in a.items()),sum(costs[i][j] for i,j in b.items()))
            adjacency=[[j for j,c in enumerate(row) if c is not None] for row in costs]
            self.assertEqual(sim.maximum_matching(adjacency,5),experiment.fast_cardinality(adjacency,5))

    def test_training_artifacts_ignore_test_durations(self):
        c=fixture_config()
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as d, contextlib.redirect_stdout(io.StringIO()):
            a,b=Path(d)/'a',Path(d)/'b'
            prepare_frames([fixture_frame(c)],c,a,{'kind':'synthetic_test_fixture'})
            prepare_frames([fixture_frame(c,test_duration=100)],c,b,{'kind':'synthetic_test_fixture'})
            self.assertEqual((a/'travel_times.csv').read_bytes(),(b/'travel_times.csv').read_bytes())
            self.assertEqual((a/'initial_distribution.json').read_bytes(),(b/'initial_distribution.json').read_bytes())
            with self.assertRaises(ValueError): experiment.load_inputs(c,a)

    def test_fixture_end_to_end_and_report(self):
        c=fixture_config()
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); prepared=root/'prepared'; output=root/'run'; output.mkdir()
            prepare_frames([fixture_frame(c)],c,prepared,{'kind':'synthetic_test_fixture'})
            folder,data,hist,initial,travel=experiment.load_inputs(c,prepared,allow_fixture=True)
            # Keep original solvers for independent comparison; production fast solver is separately checked.
            rows=[experiment.run_case(c,folder,hist,initial,travel,c['validation_dates'][0],7,2,w,output) for w in [0,3]]
            chosen=experiment.choose_weights(rows,c)
            tests=[]
            for day in c['test_dates']:
                for w in sorted(set([0,chosen['2']])):
                    tests.append(experiment.run_case(c,folder,hist,initial,travel,day,7,2,w,output))
            sim.write_csv(output/'summary.csv',tests)
            write_json(output/'selection_used.json',{'selected_weights':chosen})
            write_json(output/'run_manifest.json',{'mode':'test','complete':True,'source_kind':'synthetic_test_fixture',
                       'summary_sha256':digest(output/'summary.csv')})
            # Production report must refuse fixture results, even if layout tests use fixtures.
            with self.assertRaises(ValueError): report.run(output)
            report.run(output,allow_fixture=True)
            self.assertTrue((output/'results_table.tex').exists())
            paired=pd.read_csv(output/'paired_differences.csv')
            self.assertEqual(len(paired),2)
            self.assertTrue(all((pd.read_csv(output/'comparison.csv').service_rate.between(0,1))))
            self.assertIn('SYNTHETIC', (output/'results_table.tex').read_text())

    def test_parquet_read_when_dependency_available(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest('pyarrow not installed: real Parquet reader not locally exercised')
        from real_experiment.download import parquet_footer_ok
        c=fixture_config()
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as d, contextlib.redirect_stdout(io.StringIO()):
            p=Path(d)/'fixture.parquet'
            pq.write_table(pa.Table.from_pandas(fixture_frame(c)),p)
            self.assertTrue(parquet_footer_ok(p))
            frames=(b.to_pandas() for b in pq.ParquetFile(p).iter_batches(batch_size=20))
            result=prepare_frames(frames,c,Path(d)/'processed',{'kind':'synthetic_test_fixture'})
            self.assertGreater(result['travel_training_rows'],0)


if __name__=='__main__': unittest.main()
