#!/usr/bin/env python3
"""Merge size-group CSVs, check coverage, and summarize the five-model campaign."""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
from statistics import mean

from run_paper_eilon import CAMPAIGN, ROOT, expected_runs, campaign_path


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def average(rows, field):
    values = [number(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return mean(values) if values else ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--projector-engine', choices=('master', 'sdp', 'legacy'), default='master')
    parser.add_argument('--tau', type=float, choices=(1.5, 2., 3.), default=2.)
    parser.add_argument('--instances', type=Path, default=ROOT / 'instances')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    args.campaign = args.campaign or campaign_path(args.tau, args.projector_engine)
    planned = expected_runs(args.instances, tau=args.tau)
    if not planned:
        parser.error('No instances found; check --instances')
    expected = {row['run_id']: row for row in planned}
    by_id = {}
    fields = None
    duplicates = 0
    engines = set()
    sources = sorted(args.campaign.glob(f'size_*/runs_large.csv'))
    if not sources:
        parser.error(f'No size_*/runs_large.csv files in {args.campaign}')
    # Identical duplicated rows are harmless; conflicting repetitions are an error.
    for source in sources:
        with source.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if fields is None:
                fields = reader.fieldnames
            if reader.fieldnames != fields:
                parser.error(f'CSV schema differs: {source}')
            for row in reader:
                engine = row.get('engine', 'legacy')
                if row.get('configuration') != 'compact':
                    engines.add(engine)
                if len(engines) > 1:
                    parser.error('Mixed projector engines: merge each campaign separately')
                key = row.get('run_id')
                if key not in expected:
                    parser.error(f'Unexpected run ID {key!r} in {source}')
                target = expected[key]
                if any(str(row.get(field)) != str(target[field]) for field in ('instance', 'n', 'p', 'tau', 'configuration')):
                    parser.error(f'Run metadata inconsistent: {key} in {source}')
                if row.get('use_sdp_bound', '').lower() != 'true' or row.get('use_symmetry_breaking', '').lower() != 'true':
                    parser.error(f'Strengthening not active for {key}')
                if row.get('status') in (None, '') or None in row or any(value is None for value in row.values()):
                    parser.error(f'Incomplete CSV row for {key}; retry after the writer finishes')
                if key in by_id:
                    if by_id[key] != row:
                        parser.error(f'Conflicting duplicate {key}; resolve before merging')
                    duplicates += 1
                by_id[key] = row
    merged = [by_id[key] for key in expected if key in by_id]
    missing = [row for key, row in expected.items() if key not in by_id]
    destination = args.campaign / f'merged_{args.projector_engine}_tau{args.tau}'
    print(f'Merging {len(sources)} CSV files into {destination}')
    write_csv(destination / f'runs_{args.tau}.csv', merged, fields)
    write_csv(destination / f'missing_runs_{args.tau}.csv', missing, list(planned[0]))
    for grouping, filename in ((('n', 'configuration'), 'summary_by_size_model.csv'),
                               (('n', 'p', 'configuration'), 'summary_by_size_p_model.csv'),
                               (('configuration',), 'summary_by_model.csv')):
        groups = defaultdict(list)
        expected_counts = defaultdict(int)
        for row in planned:
            expected_counts[tuple(str(row[field]) for field in grouping)] += 1
        for row in merged:
            groups[tuple(row[field] for field in grouping)].append(row)
        summary = []
        for key, count in expected_counts.items():
            rows = groups[key]
            optimal = sum(row.get('optimal', '').lower() == 'true' for row in rows)
            summary.append(dict(zip(grouping, key), expected=count, recorded=len(rows),
                                missing=count-len(rows), optimal=optimal,
                                time_limit=sum(row['status'] == 'TIME_LIMIT' for row in rows),
                                other_status=len(rows)-optimal-sum(row['status'] == 'TIME_LIMIT' for row in rows),
                                mean_runtime_seconds=average(rows, 'runtime'),
                                total_runtime_seconds=sum(number(row.get('runtime')) or 0 for row in rows),
                                finite_gap_count=sum(number(row.get('gap')) is not None for row in rows),
                                mean_gap=average(rows, 'gap'),
                                mean_objective=average(rows, 'objective'),
                                mean_lower_bound=average(rows, 'lower_bound'),
                                mean_sdp_root_bound=average(rows, 'sdp_root_bound'),
                                mean_sdp_nodes=average(rows, 'sdp_nodes'),
                                mean_partition_triangle_cuts=average(rows, 'partition_triangle_cuts'),
                                mean_unknown_sdp_nodes=average(rows, 'unknown_sdp_nodes')))
        write_csv(destination / f'{filename}_{args.tau}.csv', summary, list(summary[0]))
    statuses = defaultdict(int)
    for row in merged:
        statuses[(row['n'], row['configuration'], row['status'])] += 1
    status_rows = [dict(n=n, configuration=model, status=status, count=count)
                   for (n, model, status), count in sorted(statuses.items())]
    write_csv(destination / f'status_counts_{args.tau}.csv', status_rows, ['n', 'configuration', 'status', 'count'])
    print(f'{len(sources)} CSV files; {len(merged)}/{len(expected)} recorded; {len(missing)} missing; {duplicates} identical duplicates.')
    print(f'Merged results and summaries: {destination}')
    if args.require_complete and missing:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
