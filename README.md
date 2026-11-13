# Projector-based formulations for continuous multifacility location

This repository contains the code, benchmark instances, and complete numerical results
for the paper *Projector-Based Conic Formulations and Exact Decompositions for
Continuous Multifacility Location*.

It provides the material needed to reproduce the formulations, computational
campaign, tables, and figures. It also contains the extended run-level data
used to construct the averages reported in the paper. Large restart
checkpoints and solver logs are omitted because their relevant fields are
already consolidated in the machine-readable result files.

Repository URL: <https://github.com/vblancoOR/Projectors_ContLocation>

## Implemented methods

The code compares five exact formulations.

- `compact` is the standard mixed-integer second-order cone formulation
  (MISOCO).
- `PW` is the pairwise projector formulation.
- `LR-D` is the facility-wise disaggregated low-rank formulation.
- `LR-A` is the facility-wise aggregated low-rank formulation.
- `LR-SA` is the sum-aggregated low-rank formulation.

The projector formulations use a Gurobi master problem and MOSEK conic or SDP
subproblems. The supported distances are weighted $\ell_\tau$ distances for
`tau` equal to `1`, `3/2`, `2`, `3`, or infinity. The rational norm cases use
second-order-cone liftings rather than power-cone constraints.

Every method returns its best feasible solution when the time limit is reached.
The result records distinguish the incumbent objective, certified lower bound,
final optimality gap, runtime, and termination status.

## Repository contents

```text
.
├── sdp_clusters.py                  Core models and common data structures
├── projector_master.py              Strengthened projector master problems
├── sdp_projector_search.py          SDP-based projector search
├── run_eilon_experiments.py         Checkpointed experiment driver
├── run_paper_eilon.py               Paper campaign launcher
├── run_all_eilon_experiments.sh     Terminal launcher for a complete campaign
├── plot_relaxation_gap_choices.py   Table and figure generation library
├── all_v1.xlsx                      Consolidated data used by the paper notebook
├── instances/processed/eilon/       Eilon benchmark instances
├── notebooks/
│   ├── 01_quick_start.ipynb
│   ├── 02_computational_experiments.ipynb
│   ├── 03_concise_paper_results.ipynb
│   ├── 04_relaxation_gap_plot_choices.ipynb
│   └── 05_automatic_table_and_figures.ipynb
├── results/
│   ├── all_v1.csv                   Extended run-level numerical results
│   ├── merged_master_all_tau.csv    Merged source records
│   └── paper_summary/               Matched summaries and supplementary outputs
├── tables/eilon_main_results.tex    Automatically generated paper table
├── figures/                         PDF and PNG publication/supplementary figures
└── tests/                            Model and solver integration tests
```

Large checkpoint directories, solver logs, temporary files, and the individual
JSON checkpoint for every run are intentionally excluded. Their relevant
numerical fields are collected in `results/all_v1.csv`,
`results/merged_master_all_tau.csv`, and `all_v1.xlsx`.

## Installation

Python 3.10 or newer is recommended. Gurobi and MOSEK must be installed and
licensed separately.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[notebooks,test]"
```

The following commands provide a short license check.

```bash
python -c "import gurobipy as gp; model = gp.Model(); model.dispose()"
python -c "import mosek; mosek.Env()"
```

## Quick example

```python
from sdp_clusters import Data, WarmStartOptions, solve_all

data = Data.from_csv(
    "instances/processed/eilon/eilon20_1.csv",
    facilities=3,
    tau=2,
)

warm_start = WarmStartOptions(
    strategy="best",
    random_starts=10,
    alternating_iterations=50,
    sdp_time_limit=60,
    seed=0,
)

results, best = solve_all(data, time_limit=300, warm_start=warm_start)
for method, result in results.items():
    print(method, result.summary())
```

## Notebook guide

- `01_quick_start.ipynb` constructs a small example, solves the available
  formulations, and visualizes the assignments and facility locations.
- `02_computational_experiments.ipynb` illustrates a configurable experiment
  grid and comparison of methods.
- `03_concise_paper_results.ipynb` produces compact summaries from campaign
  files.
- `04_relaxation_gap_plot_choices.ipynb` displays alternative relaxation-gap
  visualizations.
- `05_automatic_table_and_figures.ipynb` reproduces the final table and all
  figures used in the computational section.

## Reproducing the paper table and figures

Open and run:

```text
notebooks/05_automatic_table_and_figures.ipynb
```

The notebook reads `all_v1.xlsx`, retains the cases available for every
formulation, and creates:

- the three-block table with runtime, final MIP gap, and unsolved percentage
- the relative runtime performance profile
- the absolute CPU-time profile with the percentage solved to optimality
- the continuous-relaxation comparison with MISOCO
- the final MIP-gap comparison with MISOCO
- the diagnostic analysis of cut counts and time spent computing SDP bounds

Generated files are written to `tables/` and `figures/`. Both PDF and PNG
versions are provided. The PDFs are suitable for inclusion in LaTeX, while
the PNG files allow quick inspection on GitHub.

## Running the Eilon experiments

A single paper campaign can be started with, for example,

```bash
python run_paper_eilon.py --size 20 --tau 2
```

The tested facility counts are 2, 3, and 5. The paper campaign uses
$\tau\in\{3/2,2,3\}$ and sizes 10, 20, 30, 40, and 50. Runs are checkpointed,
so an interrupted campaign can be resumed without repeating completed cases.
See `PAPER_EXPERIMENTS.md` for the complete configuration and output fields.

## Extended numerical results

`results/all_v1.csv` contains one row per method and test case. The same data
are available in `all_v1.xlsx`, which is read directly by the publication
notebook. The records include:

- instance size, number of facilities, and norm order
- configuration and strengthening options
- status, incumbent, lower bound, final gap, and runtime
- continuous-relaxation and SDP-root bounds
- warm-start information
- decomposition iterations and cut counts
- time spent computing SDP bounds
- node and partition diagnostics when available

`results/merged_master_all_tau.csv` contains the merged campaign records before
the final presentation layer is constructed. The `results/paper_summary/`
folder provides matched records, aggregate summaries, and supplementary
figures. The CSV files are the preferred sources for independent analysis.

## Tests

```bash
pytest
```

Some tests call Gurobi or MOSEK and therefore require valid licenses. Tests
that only check data handling and projector identities do not require a
solver run.

## Citation

If this repository is used in other work, please cite the accompanying paper.
The final journal citation and repository DOI can be added here after
publication and archival release.

## License

The authors' original software and repository content are distributed under
the BSD 3-Clause License. See `LICENSE` for the complete terms.

The license does not cover Gurobi, MOSEK, or other third-party software. Users
must install those packages and obtain the corresponding licenses separately.
Benchmark files originating from external collections remain subject to their
original terms and attribution requirements, as recorded in
`instances/manifest.csv`.
