"""Licensed SDP search regression tests; standard-library unittest runner."""
import itertools
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import math
import unittest
from unittest.mock import patch

import numpy as np
from sdp_clusters import Data, Results, assignment_projectors
from sdp_projector_search import (branching_pair, components, partition_certificate,
                                  solve_node, solve_projector_sdp, violated_triangles)


class ProjectorSearchTests(unittest.TestCase):
    def setUp(self):
        self.data = Data(np.array([[0., 0.], [.2, 0.], [4.8, 5.], [5., 5.]]), 2)
        self.warm = Results(method='test', status='FEASIBLE', objective=.4,
                            labels=np.array([0, 0, 1, 1]),
                            centers=np.array([[.1, 0.], [4.9, 5.]]))

    def test_partition_inequalities_and_branch_exhaustiveness(self):
        for labels in itertools.product(range(2), repeat=4):
            if set(labels) != {0, 1}:
                continue
            P = sum(assignment_projectors(labels, 2))
            self.assertFalse(violated_triangles(P, set()))
            self.assertTrue(np.all(P <= np.diag(P)[:, None]+1e-10))
        self.assertIsNone(components(4, ((0, 1),), ((0, 1),), 2))
        groups = components(4, ((0, 1),), (), 2)
        self.assertEqual(groups, ((0, 1), (2,), (3,)))
        pair = branching_pair(None, groups, ((0, 2),))
        self.assertNotEqual(pair, (0, 2))

    def test_bounds_and_perspective_feasibility(self):
        # Enumerate every binary partition and independently solve its SOCP.
        partitions = []
        for tail in itertools.product(range(2), repeat=3):
            labels = np.array((0,)+tail)
            if len(set(labels)) == 2:
                candidate = partition_certificate(self.data, labels, 5)
                self.assertLessEqual(candidate['lower_bound'], candidate['objective']+1e-6)
                partitions.append((labels, candidate['objective']))
        optimum = min(v for _, v in partitions)
        self.assertAlmostEqual(optimum, .4, places=5)
        for method in ('PW', 'LR-D', 'LR-A', 'LR-SA'):
            for must, cannot in (((), ()), (((0, 2),), ()), ((), ((0, 1),))):
                admissible = [v for labels, v in partitions
                              if all(labels[i] == labels[h] for i, h in must)
                              and all(labels[i] != labels[h] for i, h in cannot)]
                with self.subTest(method=method, must=must, cannot=cannot):
                    result = solve_node(self.data, method, must, cannot, set(), 5)
                    self.assertFalse(result['infeasible'])
                    self.assertIsNotNone(result['bound'])
                    self.assertLessEqual(result['bound'], min(admissible)+1e-5)
            result = solve_projector_sdp(self.data, method, warm_start=self.warm,
                                        time_limit=10, root_time_limit=2, node_time_limit=1)
            self.assertTrue(result.optimal)
            self.assertAlmostEqual(result.objective, optimum, places=5)
            self.assertLessEqual(result.lower_bound, optimum+1e-6)
            self.assertLess(result.gap, 1e-4)

    def test_unknown_and_node_limit_do_not_certify(self):
        unknown = dict(infeasible=False, bound=None, P=None, cuts=0, diagnostic='injected unknown')
        with patch('sdp_projector_search.solve_node', return_value=unknown):
            result = solve_projector_sdp(self.data, 'LR-SA', warm_start=self.warm,
                                        maximum_nodes=1, time_limit=5)
        self.assertEqual(result.status, 'NODE_LIMIT')
        self.assertEqual(result.lower_bound, 0.)
        self.assertEqual(result.objective, .4)
        self.assertEqual(result.metadata['open_nodes'], 2)
        self.assertFalse(result.optimal)

    def test_terminal_failure_keeps_open_node(self):
        unknown = dict(infeasible=False, bound=None, P=None, cuts=0, diagnostic='unknown')
        with patch('sdp_projector_search.solve_node', return_value=unknown), patch(
                'sdp_projector_search.partition_certificate', return_value=None):
            result = solve_projector_sdp(self.data, 'PW', warm_start=self.warm,
                                        time_limit=5, node_time_limit=1)
        self.assertEqual(result.status, 'SUBPROBLEM_LIMIT')
        self.assertFalse(result.optimal)
        self.assertEqual(result.lower_bound, 0.)
        self.assertGreater(result.metadata['open_nodes'], 0)

    def test_sdp_runner_csv_restart_and_merge(self):
        from run_paper_eilon import ROOT
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'instances/processed/eilon/eilon10_test.csv'
            source.parent.mkdir(parents=True)
            with source.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['id', 'x', 'y', 'weight'])
                for i in range(10):
                    writer.writerow([i, i//5*5 + i%5*.01, i//5*5, 1])
            campaign = root / 'campaign'
            output = campaign / 'size_10'
            command = [sys.executable, str(ROOT/'run_eilon_experiments.py'),
                       '--instances', str(root/'instances'), '--output', str(output),
                       '--profile', 'paper', '--sizes', '10', '--facilities', '2',
                       '--taus', '2', '--projector-engine', 'sdp', '--time-limit', '1',
                       '--sdp-root-time-limit', '.3', '--sdp-node-time-limit', '.1',
                       '--warm-random-starts', '1', '--warm-iterations', '2',
                       '--warm-sdp-time-limit', '1', '--no-figures']
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
            path = output/'runs_large.csv'
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 5, run.stdout+run.stderr)
            self.assertEqual({r['engine'] for r in rows}, {'gurobi_compact', 'projector-sdp-v1'})
            self.assertTrue(all(float(r['lower_bound']) <= float(r['objective'])+1e-5 for r in rows))
            before = path.read_bytes()
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
            self.assertEqual(path.read_bytes(), before)
            merge = subprocess.run([sys.executable, str(ROOT/'merge_paper_eilon.py'),
                                    '--campaign', str(campaign), '--instances', str(root/'instances')],
                                   capture_output=True, text=True)
            self.assertEqual(merge.returncode, 0, merge.stdout+merge.stderr)
            self.assertTrue((campaign/'merged/summary_by_model.csv').exists())

    def test_triangle_separator_finds_invalid_matrix(self):
        P = np.eye(3)*.4
        P[0, 1] = P[1, 0] = .4
        P[0, 2] = P[2, 0] = .4
        self.assertIn((0, 1, 2), violated_triangles(P, set()))


if __name__ == '__main__':
    unittest.main()
