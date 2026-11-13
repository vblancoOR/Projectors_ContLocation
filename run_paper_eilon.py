#!/usr/bin/env python3
"""Five-model strengthened Eilon campaigns for tau=1.5, 2, or 3."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SIZES = (10, 20, 30, 40, 50)
MODELS = ('compact', 'PW', 'LR-D', 'LR-A', 'LR-SA')
FACILITIES = (2, 3, 5)
def campaign_path(tau=2.0, engine="master"):
    return ROOT / 'results' / f'eilon_paper_tau{float(tau):g}_3600_{engine}_v1'


CAMPAIGN = campaign_path()
print(f'Campaign folder: {CAMPAIGN}', flush=True)

def expected_runs(instances=ROOT / 'instances', sizes=SIZES, tau=2.0):
    if float(tau) not in (1.5, 2., 3.):
        raise ValueError('Supported campaign taus: 1.5, 2, 3')
    label = f'{float(tau):g}'
    records = []
    for source in sorted((instances / 'processed' / 'eilon').glob('eilon*.csv')):
        with source.open(newline='', encoding='utf-8') as stream:
            n = sum(1 for _ in csv.DictReader(stream))
        if n not in sizes:
            continue
        for p in FACILITIES:
            for model in MODELS:
                records.append(dict(run_id=f'{source.stem}__p{p}__tau{label}__{model}',
                                    instance=source.stem, n=n, p=p, tau=label, configuration=model))
    return records


def main(size=None, tau=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if size is None:
        parser.add_argument('--size', type=int, choices=SIZES, required=True)
    if tau is None:
        parser.add_argument('--tau', type=float, choices=(1.5, 2., 3.), default=2.)
    parser.add_argument('--instances', type=Path, default=ROOT / 'instances')
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--projector-engine', choices=('master', 'sdp', 'legacy'), default='master')
    parser.add_argument('--sdp-node-time-limit', type=float, default=60.)
    parser.add_argument('--sdp-root-time-limit', type=float, default=300.)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.sdp_node_time_limit <= 0 or args.sdp_root_time_limit <= 0:
        parser.error("SDP root/node limits must be positive")
    size = size if size is not None else args.size
    tau = float(tau if tau is not None else args.tau)
    tau_label = f"{tau:g}"
    args.campaign = args.campaign or campaign_path(tau, args.projector_engine)
    expected = expected_runs(args.instances, (size,), tau=tau)
    if not expected:
        parser.error(f'No size-{size} instances under {args.instances}')
    output = args.campaign / f'size_{size}'
    checkpoint = output / 'runs_large.csv'
    done = set()
    if checkpoint.exists():
        with checkpoint.open(newline='', encoding='utf-8') as stream:
            done = {row['run_id'] for row in csv.DictReader(stream)}
    missing = [row for row in expected if row['run_id'] not in done]
    print(f'tau={tau_label}, n={size}: {len(expected)} planned model solves; {len(expected)-len(missing)} recorded; {len(missing)} missing.', flush=True)
    command = [sys.executable, str(ROOT / 'run_eilon_experiments.py'),
               '--instances', str(args.instances.resolve()), '--output', str(output.resolve()),
               '--profile', 'paper', '--sizes', str(size), '--facilities', '2', '3', '5',
               '--taus', tau_label, '--time-limit', '3600', '--relaxation-time-limit', '120',
               '--sdp-time-limit', '300', '--warm-sdp-time-limit', '300',
               '--warm-random-starts', '10', '--warm-iterations', '50',
               '--mip-gap', '0.0001', '--seed', '0', '--no-figures',
               '--projector-engine', args.projector_engine, '--sdp-node-time-limit', str(args.sdp_node_time_limit),
               '--sdp-root-time-limit', str(args.sdp_root_time_limit)]
    if args.dry_run:
        import shlex
        print(shlex.join(command))
        for row in missing:
            print(row['run_id'])
        return
    # Never silently resume a different experiment under the same IDs.
    settings_file = output / 'settings.json'
    if settings_file.exists():
        settings = json.loads(settings_file.read_text())
        required = dict(profile='paper', sizes=[size], facilities=[2, 3, 5], taus=[tau_label],
                        time_limit=3600, relaxation_time_limit=120, sdp_time_limit=300,
                        warm_sdp_time_limit=300, warm_random_starts=10, warm_iterations=50,
                        mip_gap=0.0001, seed=0, projector_engine=args.projector_engine,
                        sdp_node_time_limit=args.sdp_node_time_limit,
                        sdp_root_time_limit=args.sdp_root_time_limit)
        if any(settings.get(key) != value for key, value in required.items()):
            parser.error(f'Existing settings differ: choose a fresh --campaign folder ({settings_file})')
    if not missing:
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'planned_runs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(expected[0]))
        writer.writeheader()
        writer.writerows(expected)
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env.setdefault(name, '1')
    subprocess.run(command, check=True, env=env)


if __name__ == '__main__':
    main()
