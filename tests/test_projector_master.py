"""Regression checks for the paper-compatible master--subproblem engine."""
import itertools
import math
from unittest.mock import patch
import unittest
import numpy as np
from sdp_clusters import Data, Results
from projector_master import solve_projector_master
from run_paper_eilon import campaign_path


class MasterTests(unittest.TestCase):
    def test_default_campaign_and_launcher(self):
        import subprocess, sys
        from run_paper_eilon import ROOT
        self.assertIn('master_v1', str(campaign_path()))
        run = subprocess.run([sys.executable, str(ROOT/'run_paper_eilon_20.py'), '--tau', '3', '--dry-run'],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertIn('--projector-engine master',run.stdout)
        self.assertIn('tau3_3600_master_v1',run.stdout)

    def test_matches_enumeration_all_norms_and_methods(self):
        from sdp_projector_search import partition_certificate
        points=np.array([[0.,0.],[1.,.4],[2.,1.7],[4.,.3],[5.,2.]])
        for tau in (1.5,2.,3.):
            data=Data(points,2,tau=tau)
            candidates=[]
            for tail in itertools.product(range(2),repeat=4):
                labels=np.array((0,)+tail)
                if len(set(labels))==2:
                    c=partition_certificate(data,labels,5)
                    candidates.append(c)
            best=min(candidates,key=lambda c:c['objective'])
            # Start at a deliberately poor feasible partition to exercise cuts.
            c=candidates[0]
            warm=Results(method='test',status='FEASIBLE',objective=c['objective'],labels=c['labels'],centers=c['centers'])
            for method in ('PW','LR-D','LR-A','LR-SA'):
                with self.subTest(tau=tau,method=method):
                    r=solve_projector_master(data,method,time_limit=10,warm_start=warm,sdp_time_limit=.5)
                    self.assertTrue(r.optimal,r.summary())
                    self.assertAlmostEqual(r.objective,best['objective'],places=4)
                    self.assertLessEqual(r.lower_bound,best['objective']+1e-5)
                    self.assertFalse(r.metadata['custom_sdp_branching'])
                    self.assertEqual(r.metadata['search'],'gurobi_projector_master')
                    self.assertGreater(r.metadata['subset_cuts'],0)

    def test_runner_and_merge_use_master_engine(self):
        import csv, subprocess, sys, tempfile
        from pathlib import Path
        from run_paper_eilon import ROOT
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/'instances/processed/eilon/eilon10_test.csv'
            source.parent.mkdir(parents=True)
            with source.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['id','x','y','weight'])
                for i in range(10):
                    writer.writerow([i,i//5*5+i%5*.01,i//5*5+i%5*.02,1])
            campaign=root/'campaign';output=campaign/'size_10'
            command=[sys.executable,str(ROOT/'run_eilon_experiments.py'),
                     '--instances',str(root/'instances'),'--output',str(output),
                     '--profile','paper','--projector-engine','master','--sizes','10',
                     '--facilities','2','--taus','3','--time-limit','1',
                     '--sdp-root-time-limit','.2','--warm-random-starts','1',
                     '--warm-iterations','2','--warm-sdp-time-limit','1','--no-figures']
            run=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            with (output/'runs_large.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),5,run.stdout+run.stderr)
            self.assertEqual({r['engine'] for r in rows},{'gurobi_compact','projector-master-v1'})
            merge=subprocess.run([sys.executable,str(ROOT/'merge_paper_eilon.py'),
                                  '--tau','3','--campaign',str(campaign),
                                  '--instances',str(root/'instances')],capture_output=True,text=True)
            self.assertEqual(merge.returncode,0,merge.stdout+merge.stderr)

    def test_missing_dual_certificate_does_not_make_cut_or_certify(self):
        data=Data(np.array([[0.,0.],[1.,.4],[2.,1.7],[4.,.3],[5.,2.]]),2)
        labels=np.array([0,1,0,1,0]); centers=np.array([[0.,0.],[5.,2.]])
        objective=float(np.linalg.norm(data.points-centers[labels],axis=1).sum())
        warm=Results(method='test',status='FEASIBLE',objective=objective,labels=labels,centers=centers)
        with patch('projector_master.partition_certificate',return_value=None):
            r=solve_projector_master(data,'PW',time_limit=5,warm_start=warm,strengthen_sdp=False)
        self.assertEqual(r.status,'SUBPROBLEM_LIMIT')
        self.assertFalse(r.optimal)
        self.assertEqual(r.objective,objective)
        self.assertEqual(r.metadata['projection_logic_cuts'],0)
        self.assertEqual(r.metadata['subset_cuts'],0)


if __name__ == '__main__':
    unittest.main()
