"""Power-norm SDP campaigns: mathematical bounds and end-to-end routing."""
import csv
import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from sdp_clusters import Data, Results, solve_compact
from sdp_projector_search import partition_certificate, solve_node, solve_projector_sdp
from run_paper_eilon import ROOT, expected_runs, campaign_path


class FractionalTauTests(unittest.TestCase):
    def test_plans_are_disjoint(self):
        plans = [expected_runs(tau=tau) for tau in (1.5, 2., 3.)]
        self.assertEqual([len(plan) for plan in plans], [315, 315, 315])
        ids = [{r['run_id'] for r in plan} for plan in plans]
        self.assertEqual(len(set.union(*ids)), 945)
        self.assertEqual(len({campaign_path(tau) for tau in (1.5, 2., 3.)}), 3)
        for tau, tag in ((1.5, '1p5'), (3., '3')):
            for n in (10, 20, 30, 40, 50):
                run = subprocess.run([sys.executable, str(ROOT/f'run_paper_eilon_{n}_tau{tag}.py'), '--dry-run'], capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn(f'--taus {tau:g}', run.stdout)
                self.assertIn(str(campaign_path(tau)), run.stdout)

    def test_all_models_and_conditioned_bounds(self):
        points = np.array([[0., 0.], [.2, .1], [4.8, 5.], [5., 5.1]])
        for tau in (1.5, 3.):
            data = Data(points, 2, tau=tau)
            known = 2*np.linalg.norm([.2, .1], ord=tau)
            evaluated = []
            for tail in itertools.product(range(2), repeat=3):
                labels = np.array((0,)+tail)
                if len(set(labels)) == 2:
                    result = partition_certificate(data, labels, 5)
                    self.assertLessEqual(result['lower_bound'], result['objective']+1e-6)
                    evaluated.append((labels, result['objective']))
            self.assertAlmostEqual(min(v for _, v in evaluated), known, places=5)
            warm = Results(method='test', status='FEASIBLE', objective=known,
                           labels=np.array([0, 0, 1, 1]), centers=np.array([[.1, .05], [4.9, 5.05]]))
            compact = solve_compact(data, time_limit=15, warm_start=warm,
                                    use_sdp_bound=True, use_symmetry_breaking=True,
                                    sdp_bound_time_limit=2)
            self.assertAlmostEqual(compact.objective, known, places=4)
            self.assertLessEqual(compact.lower_bound, known+1e-5)
            for method in ('PW', 'LR-D', 'LR-A', 'LR-SA'):
                for must, cannot in (((), ()), (((0, 2),), ()), ((), ((0, 1),))):
                    with self.subTest(tau=tau, method=method, must=must, cannot=cannot):
                        admissible = [v for labels, v in evaluated
                                      if all(labels[i] == labels[h] for i, h in must)
                                      and all(labels[i] != labels[h] for i, h in cannot)]
                        node = solve_node(data, method, must, cannot, set(), 5)
                        self.assertFalse(node['infeasible'])
                        # Unknown is allowed; it must not claim a false bound.
                        if node['bound'] is not None:
                            self.assertLessEqual(node['bound'], min(admissible)+1e-5)
                result = solve_projector_sdp(data, method, warm_start=warm, time_limit=10,
                                            root_time_limit=2, node_time_limit=1)
                self.assertTrue(result.optimal)
                self.assertAlmostEqual(result.objective, known, places=5)
                self.assertLessEqual(result.lower_bound, known+1e-5)

    def test_new_norm_csv_merge_and_restart(self):
        for tau in (1.5, 3.):
            with self.subTest(tau=tau), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root/'instances/processed/eilon/eilon10_test.csv'
                source.parent.mkdir(parents=True)
                with source.open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['id', 'x', 'y', 'weight'])
                    for i in range(10):
                        writer.writerow([i, i//5*5+i%5*.01, i//5*5+i%5*.02, 1])
                campaign = root/'campaign'
                output = campaign/'size_10'
                cmd = [sys.executable, str(ROOT/'run_eilon_experiments.py'),
                       '--instances', str(root/'instances'), '--output', str(output),
                       '--profile', 'paper', '--sizes', '10', '--facilities', '2',
                       '--taus', str(tau), '--projector-engine', 'sdp', '--time-limit', '1',
                       '--sdp-root-time-limit', '.3', '--sdp-node-time-limit', '.1',
                       '--warm-random-starts', '1', '--warm-iterations', '2',
                       '--warm-sdp-time-limit', '1', '--no-figures']
                run = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
                path = output/'runs_large.csv'
                with path.open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 5, run.stdout+run.stderr)
                self.assertTrue(all(float(r['tau']) == tau for r in rows))
                before = path.read_bytes()
                again = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(again.returncode, 0, again.stdout+again.stderr)
                self.assertEqual(before, path.read_bytes())
                merge = subprocess.run([sys.executable, str(ROOT/'merge_paper_eilon.py'),
                                        '--tau', str(tau), '--campaign', str(campaign),
                                        '--instances', str(root/'instances')], capture_output=True, text=True)
                self.assertEqual(merge.returncode, 0, merge.stdout+merge.stderr)
                with (campaign/'merged/missing_runs.csv').open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 10)


if __name__ == '__main__':
    unittest.main()
