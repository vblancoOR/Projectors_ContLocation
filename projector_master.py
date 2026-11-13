"""Paper-compatible projector master + fixed-projector conic evaluation.

Incidence vectors encode partition projectors exactly. The auxiliary SDP is
used only for a global lower bound; it does not run a separate search tree.
"""
import math
from time import perf_counter
import numpy as np

from sdp_clusters import (Data, Results, WarmStartOptions, build_warm_start,
                          _build_partition_master, _build_aggregate_partition_master,
                          _set_partition_start, _extract_labels, _gurobi_status,
                          _solve_gurobi_relaxation, _relative_bound_gap)
from sdp_projector_search import solve_node, partition_certificate

ENGINE_VERSION = 'projector-master-v1'


def solve_projector_master(data, method, time_limit=3600., warm_start=None,
                           sdp_time_limit=300., relaxation_time_limit=120.,
                           mip_gap=1e-4, absolute_gap=1e-6, log_output=False,
                           maximum_iterations=None, strengthen_sdp=True):
    """Exact outer approximation with dual-certified logic and subset cuts."""
    if method not in ('PW', 'LR-D', 'LR-A', 'LR-SA') or data.tau not in (1.5, 2., 3.):
        raise ValueError('Master engine supports PW/LR-D/LR-A/LR-SA and tau=1.5/2/3')
    if time_limit <= 0:
        raise ValueError('time_limit must be positive')
    started = perf_counter()
    def remaining():
        return time_limit-(perf_counter()-started)
    initial = warm_start if warm_start is not None else build_warm_start(
        data, WarmStartOptions(strategy='multistart', random_starts=1, alternating_iterations=3))
    best = math.inf if initial is None else initial.objective
    best_labels = None if initial is None or initial.labels is None else initial.labels.copy()
    best_centers = None if initial is None or initial.centers is None else initial.centers.copy()
    root = 0. if initial is None else float(initial.metadata.get('sdp_lower_bound', 0.))
    if not math.isfinite(root):
        root = 0.
    root = max(0., root)
    triangles = set()
    sdp_started = perf_counter()
    root_diagnostic = 'auxiliary SDP disabled'
    if strengthen_sdp and remaining() > 0:
        relaxed = solve_node(data, method, (), (), triangles,
                             min(sdp_time_limit, max(0., remaining())), log_output=log_output)
        root_diagnostic = relaxed['diagnostic']
        if relaxed['bound'] is not None:
            root = max(root, relaxed['bound'])
    root_time = perf_counter()-sdp_started
    same = None
    if method == 'LR-SA':
        gp, GRB, model, z, same = _build_aggregate_partition_master(data, 'LR-SA_master', log_output, True)
    else:
        gp, GRB, model, z = _build_partition_master(data, method+'_master', log_output, True)
    model.Params.MIPGap = mip_gap
    model.Params.MIPGapAbs = absolute_gap
    # Complete first-occurrence ordering: every unlabeled partition retains
    # one representative, while the manuscript's incidence encoding is kept.
    for i in range(data.n):
        for j in range(1, data.facilities):
            model.addConstr(z[i, j] <= gp.quicksum(z[h, j-1] for h in range(i)),
                            name=f'first_occurrence_{i}_{j}')
    theta = model.addVar(lb=root, name='theta')
    cluster_theta = model.addVars(data.facilities, lb=0., name='cluster_theta')
    model.addConstr(theta >= gp.quicksum(cluster_theta[j] for j in range(data.facilities)))
    distances = np.linalg.norm(data.points[:, None]-data.points[None, :], ord=data.tau, axis=2)
    geometry_cuts = 0
    if same is None:
        radial = model.addVars(data.n, data.facilities, lb=0., name='radial')
        for j in range(data.facilities):
            model.addConstr(cluster_theta[j] >= gp.quicksum(float(data.weights[i])*radial[i, j] for i in range(data.n)))
            for i in range(data.n):
                for h in range(i+1, data.n):
                    model.addConstr(radial[i, j]+radial[h, j] >= float(distances[i, h])*(z[i, j]+z[h, j]-1))
                    geometry_cuts += 1
    else:
        radial = model.addVars(data.n, lb=0., name='radial')
        model.addConstr(theta >= gp.quicksum(float(data.weights[i])*radial[i] for i in range(data.n)))
        for i in range(data.n):
            for h in range(i+1, data.n):
                model.addConstr(radial[i]+radial[h] >= float(distances[i, h])*same[i, h])
                geometry_cuts += 1
    model.setObjective(theta, GRB.MINIMIZE)
    _set_partition_start(z, best_labels, data)
    relaxation, relaxation_time = root, 0.
    if remaining() > 0:
        relaxation, relaxation_time = _solve_gurobi_relaxation(
            model, min(relaxation_time_limit, remaining()), log_output)
    lower = root
    if math.isfinite(relaxation):
        lower = max(lower, relaxation)
    history, cache, cut_keys, seen = [], {}, set(), set()
    logic_cuts = 0
    status, diagnostic = 'TIME_LIMIT', None
    limit = maximum_iterations if maximum_iterations is not None else 1_000_000
    def tolerance():
        return max(absolute_gap, mip_gap*abs(best)) if math.isfinite(best) else absolute_gap
    try:
        for iteration in range(1, limit+1):
            if remaining() <= 0:
                break
            model.Params.TimeLimit = max(.001, remaining())
            model.optimize()
            bound = float(model.ObjBound)
            if math.isfinite(bound):
                lower = max(lower, bound)
            if math.isfinite(best) and best-lower <= tolerance():
                status = 'OPTIMAL'
                break
            if model.SolCount == 0:
                status = _gurobi_status(model.Status, GRB)
                break
            labels = _extract_labels(z, data)
            # Equivalent projector reconstruction used by the paper.
            # P_j = z_j z_j' / |S_j|; the conic value then separates by S_j.
            subsets = [tuple(int(i) for i in np.flatnonzero(labels == j)) for j in range(data.facilities)]
            projectors = [np.outer(labels == j, labels == j).astype(float)/len(subsets[j])
                          for j in range(data.facilities)]
            assert np.allclose(sum(P@np.ones(data.n) for P in projectors), 1.)
            key = tuple(subsets) if same is None else tuple(sorted(subsets))
            repeated = key in seen
            seen.add(key)
            centers = np.empty((data.facilities, data.dimension))
            upper_value, certified_value = 0., 0.
            valid = True
            for j, subset in enumerate(subsets):
                if subset not in cache:
                    if remaining() <= 0:
                        valid = False
                        break
                    if float(data.weights[list(subset)].sum()) == 0.:
                        candidate = dict(centers=data.points[list(subset)].mean(axis=0)[None, :],
                                         objective=0., lower_bound=0.)
                    else:
                        cluster = Data(data.points[list(subset)], 1, data.weights[list(subset)], tau=data.tau)
                        candidate = partition_certificate(cluster, np.zeros(len(subset), dtype=int), remaining(), log_output)
                    if candidate is None or candidate['centers'] is None or candidate['lower_bound'] is None:
                        diagnostic = 'No feasible primal/dual certificate for location subproblem'
                        valid = False
                        break
                    cache[subset] = (candidate['centers'][0].copy(), candidate['objective'],
                                     max(0., candidate['lower_bound']))
                center, cost, certified = cache[subset]
                centers[j] = center
                upper_value += cost
                certified_value += certified
                # Monotonicity: every cluster containing S costs at least c(S).
                # Reuse a subset across all facility labels. Only dual lower
                # bounds enter these cuts, never feasible primal objectives.
                for h in range(data.facilities):
                    if (subset, h) not in cut_keys:
                        model.addConstr(cluster_theta[h] >= certified*(gp.quicksum(z[i, h] for i in subset)-len(subset)+1),
                                        name=f'subset_{len(cut_keys)}')
                        cut_keys.add((subset, h))
            if not valid:
                status = 'TIME_LIMIT' if remaining() <= 0 else 'SUBPROBLEM_LIMIT'
                break
            if upper_value < best:
                best, best_labels, best_centers = upper_value, labels.copy(), centers.copy()
            if same is None:
                mismatch = gp.quicksum(1-z[i, int(labels[i])] for i in range(data.n))
            else:
                mismatch = gp.quicksum((1-same[i,h]) if labels[i] == labels[h] else same[i,h]
                                       for i in range(data.n) for h in range(i+1, data.n))
            model.addConstr(theta >= certified_value*(1-mismatch), name=f'projector_logic_{iteration}')
            logic_cuts += 1
            history.append(dict(iteration=float(iteration), lower_bound=lower, upper_bound=best))
            if best-lower <= tolerance():
                status = 'OPTIMAL'
                break
            if repeated:
                # A loose conic certificate must not cause an endless repeat
                # or an unsupported optimality claim.
                status = 'SUBPROBLEM_LIMIT'
                diagnostic = 'Repeated partition with unresolved certification gap'
                break
        else:
            status = 'ITERATION_LIMIT'
    finally:
        model.dispose()
    return Results(method=method, status=status, objective=best, lower_bound=min(lower, best),
                   runtime=perf_counter()-started, labels=best_labels, centers=best_centers,
                   iterations=len(history), optimal=status=='OPTIMAL', history=history,
                   metadata=dict(engine=ENGINE_VERSION, relaxation='master_with_'+method+'_auxiliary_sdp',
                                 tau=data.tau, symmetry_breaking=True,
                                 sdp_root_bound=root, sdp_bound_time=root_time,
                                 sdp_root_gap=_relative_bound_gap(best, root),
                                 continuous_relaxation_bound=relaxation,
                                 continuous_relaxation_time=relaxation_time,
                                 continuous_relaxation_gap=_relative_bound_gap(best, relaxation),
                                 partition_triangle_cuts=len(triangles), master_geometry_cuts=geometry_cuts,
                                 projection_logic_cuts=logic_cuts, subset_cuts=len(cut_keys),
                                 evaluated_clusters=len(cache), evaluated_partitions=len(seen),
                                 subproblem_termination=diagnostic, auxiliary_sdp_status=root_diagnostic,
                                 search='gurobi_projector_master', custom_sdp_branching=False))
