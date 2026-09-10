"""Entry point for the real-data experiment. Run from any working directory."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from real_experiment.common import ROOT, load_config


def main():
    parser=argparse.ArgumentParser(description='Official TLC data -> clean -> validate -> test -> report')
    parser.add_argument('step',choices=['download','prepare','smoke','validate','test','report','all'])
    parser.add_argument('--config',type=Path,default=ROOT/'experiment_config.json')
    parser.add_argument('--selection',type=Path,help='Optional frozen validation selection.json for test')
    parser.add_argument('--run-dir',type=Path,help='Completed test directory to summarize')
    args=parser.parse_args()
    try:
        c=load_config(args.config)
        print('REAL-TRIP-DRIVEN SIMULATION; not a reconstruction of actual Uber supply.',flush=True)
        if args.step in ('download','all'):
            from real_experiment.download import run
            run(c)
        if args.step in ('prepare','all'):
            from real_experiment.prepare import run
            run(c)
        if args.step in ('smoke','validate','test','all'):
            from real_experiment.experiment import run
            if args.step=='all':
                validation=run(c,'validate')
                test=run(c,'test',validation/'selection.json')
                from real_experiment.report import run as report
                report(test)
            else:
                run(c,args.step,args.selection)
        if args.step=='report':
            from real_experiment.report import run
            run(args.run_dir)
    except ModuleNotFoundError as e:
        print(f'Missing package: {e.name}\nInstall using this same Python interpreter:\n'
              f'  "{sys.executable}" -m pip install -r "{ROOT / "requirements-real.txt"}"',file=sys.stderr)
        return 1
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError) as e:
        print(f'Cannot continue: {e}',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
