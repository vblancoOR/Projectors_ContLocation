# Paper-compatible strengthened master--subproblem experiments

The five size launchers now default to `--projector-engine master`. This keeps
the paper's exact outer-approximation architecture. The separate custom SDP
branch-and-bound experiment is retained only as an explicitly selected option.

## Algorithm and relation to the manuscript

Reference: `MixedConicProjections_Cluster.tex`, particularly
`mod:pairwise-benders-master`, `eq:pairwise-subproblem-value`,
`mod:cutting-plane-master`, `eq:cutting-plane-value-function`, and
`ctr:optimality-cut`.

1. Build a Gurobi master over partition projectors, encoded exactly by binary
   incidence vectors. Generate a shared warm incumbent.
2. Solve an auxiliary strengthened SDP relaxation with projector inequalities
   and, for LR, matrix-perspective location coupling. Its feasible dual value
   supplies a global objective lower bound in the master.
3. Solve the master. At its incumbent reconstruct the projectors
   `P_j=z_j z_j'/|S_j|` and fix them. MOSEK solves the resulting continuous conic
   location problem, using the exact single-center-per-cluster reduction.
4. Update the feasible objective and add projector logic cuts and reusable
   certified subset cuts to the master. Repeat until the master bound closes
   the requested gap or the time budget is reached.

There is no custom must-link/cannot-link search tree in this engine. Gurobi
handles its normal internal MIP search. For fixed projectors the location
subproblem is conic; a PSD cone is not needed after eliminating Q using Q=PQ.
The SDP is an auxiliary lower-bound relaxation, not a replacement for the
master. No unsupported dual-gradient cuts are introduced.

## Strengthening inside this architecture

- Stronger auxiliary SDP: partition diagonal inequalities, separated triangle
  inequalities, and bounded matrix-perspective blocks for LR.
- Master metric inequalities: customer-distance lower-bound variables obey
  triangle inequalities whenever two customers share a facility.
- Complete first-occurrence ordering eliminates facility-label permutations.
- Evaluated subsets are cached and reused across facility labels. For S and a
  certified lower bound L(S) on its optimal location cost, add
  `theta_j >= L(S) (sum_{i in S} z_ij - |S| + 1)`.
  This remains active when a cluster contains S, because adding nonnegative
  demand costs cannot reduce its optimal cost.
- The paper's full-partition logic cut is also returned after each complete
  evaluation. Only dual-certified lower values enter cuts. Feasible primal
  costs are used for incumbents; they are never silently substituted for a
  certified subproblem value.

PW, LR-D, and LR-A have equivalent fixed-projector value oracles. Their
strengthened incidence masters share infrastructure; their auxiliary SDP
relaxations retain the pairwise / disaggregated / block-diagonal structures.
LR-SA uses co-assignment variables and unlabeled partition logic cuts, with a
sum-aggregated auxiliary SDP. These are algorithmic implementations of the
manuscript's equivalent models, not claims of different integer feasible sets.

Unknown conic statuses never certify optimality. An unresolved subproblem
preserves the incumbent and master bound and returns SUBPROBLEM_LIMIT (or
TIME_LIMIT when the total budget is exhausted). Detailed JSON records the
reason. The previous standalone SDP search remains in `sdp_projector_search.py`;
its reusable continuous relaxation and certificate functions are called by the
master engine, but its branching function is not.

## Launch and counts

In each terminal:

```sh
cd "/Users/vblanco/Library/CloudStorage/OneDrive-UNIVERSIDADDEGRANADA/SDP Clusters Code"
```

Run one size command in each of five terminals:

```sh
python3 run_paper_eilon_10.py
python3 run_paper_eilon_20.py
python3 run_paper_eilon_30.py
python3 run_paper_eilon_40.py
python3 run_paper_eilon_50.py
```

The default is tau=2. Add `--tau 1.5` or `--tau 3` to any of these commands for
another norm. Dedicated `run_paper_eilon_N_tau1p5.py` and
`run_paper_eilon_N_tau3.py` launchers also select the master engine by default.
`--dry-run` inspects the exact command, output location, and missing run IDs.

Each norm has 315 solves: five source instances at each of n=10,20,30,40 and
one at n=50, each with p=2,3,5 and five models (compact, PW, LR-D, LR-A, LR-SA).
Counts are 75,75,75,75,15. All three norms together give 945 solves.

## Output isolation and restarting

The corrected default campaigns are separate:

- `results/eilon_paper_tau1.5_3600_master_v1/size_N/`
- `results/eilon_paper_tau2_3600_master_v1/size_N/`
- `results/eilon_paper_tau3_3600_master_v1/size_N/`

Previous `...sdp_v1/` custom-search and original campaign results are preserved
and not reused as completion checkpoints. Restarting a launcher skips recorded
rows in its own campaign, including TIME_LIMIT rows. Errors without result rows
remain eligible for retry. Existing settings must match before resuming.
Do not launch duplicate copies of the same size into the same campaign folder.

Each folder contains a plan, settings, flat results CSV, detailed JSON records,
and errors if any. The engine column distinguishes `projector-master-v1` from
`projector-sdp-v1` and `gurobi_compact`. Master JSON additionally records subset
cuts, metric inequalities, logic cuts, cached clusters, and auxiliary SDP status.

## Budgets and norm representations

Total budget: 3600 seconds per model, including its auxiliary SDP and master
relaxation. Shared warm-start generation is separate. Auxiliary strengthened
SDP cap: 300 seconds, adjustable with `--sdp-root-time-limit`. Master continuous
relaxation cap: 120 seconds. Remaining time is passed to Gurobi and the fixed-
projector conic evaluations. Model construction, solver preprocessing and
cleanup can cause wall-clock overruns.

Relative stopping tolerance is 1e-4 and the master engine's absolute stopping
tolerance is 1e-6. OPTIMAL means within numerical and requested tolerances.
Warm start settings remain 10 random starts, 50 iterations, 300-second warm-SDP
cap, seed 0. MOSEK SDP/evaluation calls use one thread; Gurobi retains its own
thread policy. Simultaneous size groups compete for CPU and memory.

Tau=2 uses exact SOC distance epigraphs. Tau=1.5 and 3 use exact MOSEK power-cone
epigraphs; compact retains the rational-norm rotated-SOC lifting. Non-Euclidean
center bounds use the coordinate box, justified by coordinatewise clipping;
the corresponding PSD trace bound and compact big-M use that box. Tau=2 keeps
the Euclidean convex-hull restriction. The compact baseline retains its shared
initial SDP floor and symmetry constraints.

## Merge

```sh
python3 merge_paper_eilon.py
python3 merge_paper_eilon.py --tau 1.5
python3 merge_paper_eilon.py --tau 3
```

The merger defaults to the corrected master campaign. Add `--require-complete`
to exit with a nonzero status while any expected run is missing. Output under
`merged/`: runs.csv, summaries by size/model, size/p/model and model, status
counts, and missing_runs.csv. Partial campaigns can be summarized. Gaps are
fractions; blank/nonfinite values are excluded from means. Conflicting
duplicates, incompatible schemas, wrong norms, and mixed engines are rejected.

To inspect the earlier custom SDP-search campaign explicitly:

```sh
python3 merge_paper_eilon.py --projector-engine sdp --tau 2
```

## Optional alternative engines

`run_paper_eilon_N.py --projector-engine sdp` explicitly selects the earlier
custom SDP branch-and-bound and writes to `...sdp_v1/`. Only that engine uses
`--sdp-node-time-limit`. `--projector-engine legacy` selects the original
unstrengthened master loop. The main low-level runner keeps its legacy default;
the five paper launchers explicitly select master. Do not pool engines into one
campaign when comparing algorithms.

## Validation

```sh
python3 -m unittest discover -s tests -p test_projector_master.py -v
python3 -m unittest discover -s tests -p test_paper_campaign.py -v
python3 -m unittest discover -s tests -p test_sdp_projector_search.py -v
python3 -m unittest discover -s tests -p test_fractional_tau_campaigns.py -v
```

Checks include enumeration of small-instance optima for all norms, bound
validity, all four projector methods, missing-certificate safety, default
launcher routing, end-to-end CSV output, restarting and merging. Temporary
checks do not launch production campaigns. No performance advantage over
compact is assumed before experiments.
