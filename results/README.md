# Extended results

`all_v1.csv` and the root-level `all_v1.xlsx` contain the run-level data used
by `notebooks/05_automatic_table_and_figures.ipynb`. Each row corresponds to
one instance, number of facilities, norm order, and formulation.

The principal identifiers are `run_id`, `instance`, `n`, `p`, `tau`, and
`configuration`. The main outcome fields are `status`, `optimal`, `objective`,
`lower_bound`, `gap`, `runtime`, and `Unsolved`.

Relaxation information is stored in the fields beginning with
`continuous_relaxation_` and `sdp_root_`. Decomposition diagnostics include
`iterations`, `projection_logic_cuts`, `partition_triangle_cuts`,
`evaluated_partitions`, `evaluated_clusters`, and `sdp_bound_time`. Some
method-specific fields are not applicable to every formulation and are left
empty in those rows.

The publication notebook forms matched groups before comparing methods. A
test case is included in a matched group only when all five configurations are
available for the same `(instance, p, tau)` combination.

`merged_master_all_tau.csv` is the merged campaign file from which the final
workbook was assembled. `paper_summary/` contains matched records, aggregate
tables, and supplementary plots. These files allow readers to check the
reported averages without running Gurobi or MOSEK again.
