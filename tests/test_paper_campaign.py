"""Run with: python3 -m unittest discover -s tests -p test_paper_campaign.py -v"""
import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from run_paper_eilon import ROOT, expected_runs
from run_eilon_experiments import configurations
from sdp_clusters import (Data, solve_compact, solve_pairwise, solve_lr_d,
                          solve_lr_a, solve_lr_sa, assignment_projectors)


class PaperCampaignTests(unittest.TestCase):
    def test_plan_and_configuration(self):
        plan = expected_runs()
        self.assertEqual(len(plan), 315)
        self.assertEqual(len({row['run_id'] for row in plan}), 315)
        self.assertEqual([c.name for c in configurations('paper', True)],
                         ['compact', 'PW', 'LR-D', 'LR-A', 'LR-SA'])
        self.assertTrue(all(c.use_sdp_bound and c.use_symmetry_breaking
                            for c in configurations('paper', True)))

    def test_sdp_strengthening_does_not_reuse_non_sdp_bound(self):
        from sdp_clusters import Results, _obtain_sdp_bound
        data = Data(np.zeros((2, 2)), facilities=1)
        initial = Results(method="alternating", status="HEURISTIC", objective=1.,
                          lower_bound=.1, runtime=0.)
        with patch('sdp_clusters.solve_sdp_relaxation', return_value=(.2, None, .5)) as solve:
            self.assertEqual(_obtain_sdp_bound(data, initial, True, 5, False), (.2, .5))
            solve.assert_called_once()
        initial.metadata['sdp_lower_bound'] = .2
        with patch('sdp_clusters.solve_sdp_relaxation') as solve:
            self.assertEqual(_obtain_sdp_bound(data, initial, True, 5, False), (.2, 0.))
            solve.assert_not_called()

    def test_expired_location_budget(self):
        from sdp_clusters import SubproblemTimeLimit, solve_single_facility, solve_fixed_partition
        data = Data(np.array([[0., 0.], [.2, 0.], [4.8, 5.], [5., 5.]]), facilities=2)
        with self.assertRaises(SubproblemTimeLimit):
            solve_single_facility(data.points, data.weights, time_limit=0)
        with self.assertRaises(SubproblemTimeLimit):
            solve_fixed_partition(data, [0, 0, 1, 1], time_limit=0)

    def test_subproblem_timeout_preserves_incumbent_without_cut(self):
        from sdp_clusters import SubproblemTimeLimit, Results
        data = Data(np.array([[0., 0.], [.2, 0.], [4.8, 5.], [5., 5.]]), facilities=2)
        warm = Results(method="test", status="FEASIBLE", objective=.4, lower_bound=0.,
                       runtime=0., labels=np.array([0, 0, 1, 1]),
                       centers=np.array([[.1, 0.], [4.9, 5.]]))
        for solver in (solve_pairwise, solve_lr_a, solve_lr_sa, solve_lr_d):
            target = 'solve_single_facility' if solver is solve_lr_d else 'solve_fixed_partition'
            with self.subTest(solver=solver.__name__), patch(
                    'sdp_clusters.' + target, side_effect=SubproblemTimeLimit('injected deadline')):
                result = solver(data, time_limit=5, warm_start=warm,
                                use_sdp_bound=False, compute_relaxation=False)
                self.assertEqual(result.status, 'TIME_LIMIT')
                self.assertEqual(result.objective, .4)
                self.assertTrue(np.array_equal(result.labels, warm.labels))
                self.assertEqual(result.metadata['projection_logic_cuts'], 0)
                self.assertEqual(result.metadata['subproblem_termination'], 'injected deadline')
                self.assertFalse(result.optimal)
                self.assertTrue(math.isfinite(result.lower_bound))

    def test_five_models_known_optimum(self):
        data = Data(np.array([[0., 0.], [.2, 0.], [4.8, 5.], [5., 5.]]), facilities=2)
        for solver in (solve_compact, solve_pairwise, solve_lr_d, solve_lr_a, solve_lr_sa):
            with self.subTest(solver=solver.__name__):
                # No supplied warm solution: exercise optimization and cut generation.
                result = solver(data, time_limit=20, warm_start=None,
                                use_sdp_bound=True, use_symmetry_breaking=True,
                                sdp_bound_time_limit=5)
                self.assertAlmostEqual(result.objective, .4, places=5)
                self.assertLessEqual(result.lower_bound, result.objective + 1e-5)
                self.assertTrue(result.metadata['symmetry_breaking'])
                P = assignment_projectors(result.labels, 2)
                Q = [np.outer(result.labels == j, result.centers[j]) for j in range(2)]
                self.assertTrue(all(np.allclose(p @ q, q) for p, q in zip(P, Q)))
                self.assertTrue(np.allclose(sum(P) @ sum(Q), sum(Q)))

    def test_end_to_end_restart_and_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instances = root / 'instances'
            source = instances / 'processed/eilon/eilon10_test.csv'
            source.parent.mkdir(parents=True)
            with source.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['id', 'x', 'y', 'weight'])
                for i in range(10):
                    writer.writerow([i, float(i // 2), float(i // 2), 1])
            campaign = root / 'campaign'
            output = campaign / 'size_10'
            cmd = [sys.executable, str(ROOT / 'run_eilon_experiments.py'),
                   '--instances', str(instances), '--output', str(output),
                   '--profile', 'paper', '--sizes', '10', '--facilities', '2',
                   '--taus', '2', '--time-limit', '2', '--relaxation-time-limit', '.1',
                   '--sdp-time-limit', '1', '--warm-sdp-time-limit', '1',
                   '--warm-random-starts', '1', '--warm-iterations', '2', '--no-figures']
            first = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            csv_path = output / 'runs_large.csv'
            with csv_path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 5, first.stdout + first.stderr)
            self.assertTrue(all(row['use_sdp_bound'] == 'True' and
                                row['use_symmetry_breaking'] == 'True' for row in rows))
            before = csv_path.read_bytes()
            again = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertIn('skip completed', again.stdout)
            self.assertEqual(csv_path.read_bytes(), before)
            merge = subprocess.run([sys.executable, str(ROOT / 'merge_paper_eilon.py'),
                                    '--campaign', str(campaign), '--instances', str(instances)],
                                   capture_output=True, text=True)
            self.assertEqual(merge.returncode, 0, merge.stdout + merge.stderr)
            with (campaign / 'merged/missing_runs.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 10)
            with (campaign / 'merged/summary_by_model.csv').open() as stream:
                summary = list(csv.DictReader(stream))
            self.assertEqual(sum(int(row['recorded']) for row in summary), 5)
            self.assertEqual(sum(int(row['expected']) for row in summary), 15)


if __name__ == '__main__':
    unittest.main()
