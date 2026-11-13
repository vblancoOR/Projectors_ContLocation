"""Cut-strengthened SDP branch-and-bound for ell-tau partition projectors.

PW uses the pairwise objective. LR-D/A use facility-block matrix perspectives;
LR-SA uses a single sum-aggregated matrix perspective.
Only feasible dual objectives are used for bounds; unknown solutions never
prune nodes. This module does not manufacture gradients from partition solves.
"""
from __future__ import annotations

import heapq
import itertools
import math
import sys
from time import perf_counter

import numpy as np
from mosek.fusion import (AccSolutionStatus, Domain, Expr, Matrix, Model,
                          ObjectiveSense, ProblemStatus, SolutionStatus, OptimizeError, SolutionError)
from sdp_clusters import (Results, WarmStartOptions, build_warm_start,
                          _repair_nonempty, _relative_bound_gap)

ENGINE_VERSION = 'projector-sdp-v1'


def components(n, must, cannot, p):
    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, h in must:
        parent[root(h)] = root(i)
    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    groups = tuple(tuple(v) for v in groups.values())
    if len(groups) < p or any(root(i) == root(h) for i, h in cannot):
        return None
    return groups


def branching_pair(P, groups, cannot):
    forbidden = {tuple(sorted((i, h))) for i, h in cannot}
    candidates = []
    for a, b in itertools.combinations(groups, 2):
        if any(tuple(sorted((i, h))) in forbidden for i in a for h in b):
            continue
        i, h = a[0], b[0]
        score = .5 if P is None else float(P[i, h] / max(1e-12, np.sqrt(max(0., P[i, i] * P[h, h]))))
        candidates.append((abs(score - .5), i, h))
    return min(candidates)[1:] if candidates else None


def violated_triangles(P, existing, count=150, tolerance=1e-6):
    candidates = []
    for i in range(len(P)):
        for h in range(len(P)):
            if h == i:
                continue
            for k in range(h + 1, len(P)):
                if k == i or (i, h, k) in existing:
                    continue
                violation = P[i, h] + P[i, k] - P[i, i] - P[h, k]
                if violation > tolerance:
                    candidates.append((violation, (i, h, k)))
    candidates.sort(reverse=True)
    return [key for _, key in candidates[:count]]


def _dual_bound(model):
    # A feasible dual point, not a primal relaxation objective, is a bound.
    if model.getDualSolutionStatus() not in (SolutionStatus.Optimal, SolutionStatus.Feasible):
        return None
    model.acceptedSolutionStatus(AccSolutionStatus.Feasible)
    value = float(model.dualObjValue())
    return value - 1e-7 * max(1., abs(value)) if math.isfinite(value) else None


def norm_epigraph(model, t, residual, dimension, tau, name):
    """Exact ell-2 SOC or ell-3/2, ell-3 power-cone epigraph."""
    if tau == 2.:
        model.constraint(Expr.vstack(t, residual), Domain.inQCone())
    elif tau in (1.5, 3.):
        scale = model.variable(name, dimension, Domain.greaterThan(0.))
        model.constraint(Expr.sub(Expr.sum(scale), t), Domain.lessThan(0.))
        for k in range(dimension):
            model.constraint(Expr.vstack(scale.index(k), t, residual.index(k)),
                             Domain.inPPowerCone(1./tau))
    else:
        raise ValueError('SDP norm epigraph supports tau=1.5, 2, 3')


def box_center_constraints(model, Q, alpha, points):
    """Coordinatewise clipping is non-increasing for every ell-p norm."""
    n, m = points.shape
    for k in range(m):
        column = Expr.flatten(Q.slice([0, k], [n, k+1]))
        lower, upper = float(points[:, k].min()), float(points[:, k].max())
        if alpha is None:
            model.constraint(column, Domain.inRange(lower, upper))
        else:
            model.constraint(Expr.sub(column, Expr.mul(lower, alpha)), Domain.greaterThan(0.))
            model.constraint(Expr.sub(Expr.mul(upper, alpha), column), Domain.greaterThan(0.))


def partition_certificate(data, labels, budget, log_output=False):
    """One joint conic solve; actual feasible cost and a feasible dual bound."""
    if budget <= 0:
        return None
    labels = _repair_nonempty(np.asarray(labels, dtype=int), data)
    with Model('partition_certificate') as model:
        model.setSolverParam('numThreads', 1)
        model.setSolverParam('optimizerMaxTime', max(.001, budget))
        if log_output:
            model.setLogHandler(sys.stdout)
        x = model.variable('centers', [data.facilities, data.dimension], Domain.unbounded())
        for k in range(data.dimension):
            column = Expr.flatten(x.slice([0, k], [data.facilities, k+1]))
            model.constraint(column, Domain.inRange(float(data.points[:, k].min()),
                                                    float(data.points[:, k].max())))
        t = model.variable('distances', data.n, Domain.greaterThan(0.))
        for i in range(data.n):
            residual = Expr.sub(x.slice([int(labels[i]), 0], [int(labels[i])+1, data.dimension]).reshape(data.dimension), data.points[i].tolist())
            norm_epigraph(model, t.index(i), residual, data.dimension, data.tau, f'norm_{i}')
        model.objective(ObjectiveSense.Minimize, Expr.dot(data.weights.tolist(), t))
        try:
            model.solve()
        except (OptimizeError, SolutionError):
            return None
        dual = _dual_bound(model)
        if model.getPrimalSolutionStatus() not in (SolutionStatus.Optimal, SolutionStatus.Feasible):
            return dict(lower_bound=dual, labels=labels, centers=None, objective=math.inf)
        model.acceptedSolutionStatus(AccSolutionStatus.Feasible)
        centers = np.asarray(x.level()).reshape(data.facilities, data.dimension)
        value = float(np.dot(data.weights, np.linalg.norm(data.points-centers[labels], ord=data.tau, axis=1)))
        return dict(lower_bound=dual, labels=labels, centers=centers, objective=value)


def solve_node(data, method, must, cannot, triangles, budget, cut_rounds=3, log_output=False):
    """Continuous node relaxation; the pairwise bound is retained in LR models."""
    started = perf_counter()
    n, m, p = data.n, data.dimension, data.facilities
    groups = components(n, must, cannot, p)
    if groups is None:
        return dict(infeasible=True, bound=None, P=None, cuts=0, diagnostic='inconsistent links')
    location = method != 'PW'
    g = len(groups)
    U = np.zeros((n, g))
    for j, group in enumerate(groups):
        U[list(group), j] = 1. / np.sqrt(len(group))
    indices_i, indices_j = np.nonzero(U)
    values = U[indices_i, indices_j].tolist()
    left = Matrix.sparse(n, g, indices_i.tolist(), indices_j.tolist(), values)
    right = Matrix.sparse(g, n, indices_j.tolist(), indices_i.tolist(), values)
    distances = np.linalg.norm(data.points[:, None] - data.points[None, :], ord=data.tau, axis=2)
    cost = .5 * np.minimum(data.weights[:, None], data.weights[None, :]) * distances
    with Model('projector_node_' + method) as model:
        model.setSolverParam('numThreads', 1)
        if log_output:
            model.setLogHandler(sys.stdout)
        S = model.variable('compressed_P', Domain.inPSDCone(g))
        if g == n:
            P = S
        else:
            P = model.variable('expanded_P', [n, n], Domain.unbounded())
            model.constraint(Expr.sub(P, Expr.mul(Expr.mul(left, S), right)), Domain.equalsTo(0.))
        model.constraint(S, Domain.greaterThan(0.))
        model.constraint(Expr.sum(S.diag()), Domain.equalsTo(float(p)))
        model.constraint(Expr.mul(P, np.ones(n).tolist()), Domain.equalsTo(np.ones(n).tolist()))
        # Nonnegative symmetric stochastic PSD matrices already satisfy P<=I.
        # Adding that redundant PSD cone would only increase solve cost.
        for i in range(n):
            for h in range(n):
                if i != h:
                    model.constraint(Expr.sub(P.index([i, i]), P.index([i, h])), Domain.greaterThan(0.))
        for i, h in cannot:
            model.constraint(P.index([i, h]), Domain.equalsTo(0.))
        def add_triangle(key):
            i, h, k = key
            model.constraint(Expr.sub(Expr.add(P.index([i, h]), P.index([i, k])),
                                      Expr.add(P.index([i, i]), P.index([h, k]))), Domain.lessThan(0.))
        for key in triangles:
            add_triangle(key)
        theta = model.variable('theta', 1, Domain.greaterThan(0.))
        model.constraint(Expr.sub(theta, Expr.dot(Matrix.dense(cost), P)), Domain.greaterThan(0.))
        if location:
            shifted = data.points-data.points.mean(axis=0)
            radius2 = (float(np.max(np.sum(shifted**2, axis=1))) if data.tau == 2.
                       else float(np.sum(np.max(np.abs(shifted), axis=0)**2)))
            # The hull's extreme points suffice; interior demand points need
            # not create center-bounding variables in every facility block.
            hull_points = np.unique(shifted, axis=0)
            if len(hull_points) > m:
                from scipy.spatial import ConvexHull, QhullError
                try:
                    hull_points = hull_points[ConvexHull(hull_points).vertices]
                except QhullError:
                    pass  # Collinear/degenerate input: retain every point.
            nh = len(hull_points)
            if method == 'LR-SA':
                block = model.variable('perspective', Domain.inPSDCone(g+m))
                model.constraint(Expr.sub(block.slice([0, 0], [g, g]), S), Domain.equalsTo(0.))
                R = block.slice([0, g], [g, g+m])
                if g == n:
                    Q = R
                else:
                    Q = model.variable('expanded_Q', [n, m], Domain.unbounded())
                    model.constraint(Expr.sub(Q, Expr.mul(left, R)), Domain.equalsTo(0.))
                T = block.slice([g, g], [g+m, g+m])
                model.constraint(Expr.sum(T.diag()), Domain.lessThan(n * radius2))
                if data.tau == 2.:
                    hull = model.variable('hull', [n, nh], Domain.greaterThan(0.))
                    model.constraint(Expr.mul(hull, np.ones(nh).tolist()), Domain.equalsTo(np.ones(n).tolist()))
                    model.constraint(Expr.sub(Q, Expr.mul(hull, Matrix.dense(hull_points))), Domain.equalsTo(0.))
                else:
                    box_center_constraints(model, Q, None, shifted)
                r = model.variable('r', n, Domain.greaterThan(0.))
                for i in range(n):
                    q = Expr.flatten(Q.slice([i, 0], [i+1, m]))
                    norm_epigraph(model, r.index(i), Expr.sub(q, shifted[i].tolist()), m, data.tau, f'norm_{i}')
                model.constraint(Expr.sub(theta, Expr.dot(data.weights.tolist(), r)), Domain.greaterThan(0.))
            else:
                # MPCO-D/A: one normalized projector and perspective per
                # facility. LR-A's block diagonal PSD cone is represented by
                # its independent diagonal cones, avoiding a giant sparse cone.
                projectors, traces, location_costs, cluster_epigraphs = [], [], [], []
                assignments = []
                for j in range(p):
                    block = model.variable(f'perspective_{j}', Domain.inPSDCone(g+m))
                    C = block.slice([0, 0], [g, g])
                    R = block.slice([0, g], [g, g+m])
                    if g == n:
                        B, Q = C, R
                    else:
                        B = model.variable(f'expanded_B_{j}', [n, n], Domain.unbounded())
                        Q = model.variable(f'expanded_Q_{j}', [n, m], Domain.unbounded())
                        model.constraint(Expr.sub(B, Expr.mul(Expr.mul(left, C), right)), Domain.equalsTo(0.))
                        model.constraint(Expr.sub(Q, Expr.mul(left, R)), Domain.equalsTo(0.))
                    T = block.slice([g, g], [g+m, g+m])
                    projectors.append(C)
                    traces.append(Expr.sum(T.diag()))
                    model.constraint(B, Domain.greaterThan(0.))
                    model.constraint(Expr.sum(C.diag()), Domain.equalsTo(1.))
                    alpha = Expr.mul(B, np.ones(n).tolist())
                    assignments.append(alpha)
                    model.constraint(alpha, Domain.inRange(0., 1.))
                    for i in range(n):
                        for h in range(n):
                            if h != i:
                                model.constraint(Expr.sub(B.index([i, i]), B.index([i, h])), Domain.greaterThan(0.))
                    if data.tau == 2.:
                        hull = model.variable(f'hull_{j}', [n, nh], Domain.greaterThan(0.))
                        model.constraint(Expr.sub(Expr.mul(hull, np.ones(nh).tolist()), alpha), Domain.equalsTo(0.))
                        model.constraint(Expr.sub(Q, Expr.mul(hull, Matrix.dense(hull_points))), Domain.equalsTo(0.))
                    else:
                        box_center_constraints(model, Q, alpha, shifted)
                    r = model.variable(f'r_{j}', n, Domain.greaterThan(0.))
                    for i in range(n):
                        q = Expr.flatten(Q.slice([i, 0], [i+1, m]))
                        residual = Expr.sub(q, Expr.mul(shifted[i].tolist(), alpha.index(i)))
                        norm_epigraph(model, r.index(i), residual, m, data.tau, f'norm_{j}_{i}')
                    cluster_cost = Expr.dot(data.weights.tolist(), r)
                    location_costs.append(cluster_cost)
                    if method == 'LR-D':
                        epigraph = model.variable(f'theta_{j}', 1, Domain.greaterThan(0.))
                        model.constraint(Expr.sub(epigraph, cluster_cost), Domain.greaterThan(0.))
                        model.constraint(Expr.sub(epigraph, Expr.dot(Matrix.dense(cost), B)), Domain.greaterThan(0.))
                        cluster_epigraphs.append(epigraph)
                model.constraint(Expr.sub(S, Expr.add(projectors)), Domain.equalsTo(0.))
                model.constraint(Expr.add(traces), Domain.lessThan(n * radius2))
                for i in range(min(n, p)):
                    model.constraint(Expr.add([assignments[j].index(i) for j in range(i+1)]), Domain.equalsTo(1.))
                total = Expr.add(cluster_epigraphs if method == 'LR-D' else location_costs)
                model.constraint(Expr.sub(theta, total), Domain.greaterThan(0.))
        model.objective(ObjectiveSense.Minimize, theta)
        bound, levels, added = None, None, 0
        diagnostic = ''
        for round_index in range(cut_rounds + 1):
            remaining = budget-(perf_counter()-started)
            if remaining <= 0:
                break
            model.setSolverParam('optimizerMaxTime', max(.001, remaining))
            try:
                model.solve()
            except (OptimizeError, SolutionError) as error:
                diagnostic = f"{type(error).__name__}: {error}"
                break
            # A dual infeasibility certificate proves primal infeasibility.
            if (model.getProblemStatus() == ProblemStatus.PrimalInfeasible
                    and model.getDualSolutionStatus() == SolutionStatus.Certificate):
                return dict(infeasible=True, bound=None, P=None, cuts=added, diagnostic='SDP infeasibility certificate')
            value = _dual_bound(model)
            if value is not None:
                bound = value if bound is None else max(bound, value)
            diagnostic = f'{model.getProblemStatus()}; dual={model.getDualSolutionStatus()}'
            if model.getPrimalSolutionStatus() not in (SolutionStatus.Optimal, SolutionStatus.Feasible):
                break
            model.acceptedSolutionStatus(AccSolutionStatus.Feasible)
            levels = U @ np.asarray(S.level()).reshape(g, g) @ U.T
            if round_index == cut_rounds:
                break
            new = violated_triangles(levels, triangles)
            if not new:
                break
            for key in new:
                add_triangle(key)
                triangles.add(key)
            added += len(new)
        return dict(infeasible=False, bound=bound, P=levels, cuts=added, diagnostic=diagnostic)


def spectral_labels(P, data):
    """Deterministic farthest-first spectral rounding; only used for incumbents."""
    _, vectors = np.linalg.eigh((P+P.T)/2)
    scores = vectors[:, -data.facilities:]
    chosen = [int(np.argmax(np.sum(scores**2, axis=1)))]
    for _ in range(1, data.facilities):
        d = np.min([np.sum((scores-scores[i])**2, axis=1) for i in chosen], axis=0)
        for i in chosen:
            d[i] = -1
        chosen.append(int(np.argmax(d)))
    centers = scores[chosen].copy()
    labels = np.zeros(data.n, dtype=int)
    for _ in range(15):
        labels = np.argmin(np.sum((scores[:, None]-centers[None, :])**2, axis=2), axis=1)
        labels = _repair_nonempty(labels, data)
        updated = np.array([scores[labels == j].mean(axis=0) for j in range(data.facilities)])
        if np.allclose(updated, centers):
            break
        centers = updated
    return labels


def solve_projector_sdp(data, method, time_limit=3600., warm_start=None,
                        node_time_limit=60., root_time_limit=300., mip_gap=1e-4,
                        absolute_gap=1e-6, log_output=False, maximum_nodes=None):
    """Best-bound search with valid must-link/cannot-link disjunctions.

    Node bounds are inherited when an SDP times out or has unknown status.
    Exact partition SOCPs provide both incumbents and terminal dual bounds.
    A failed terminal certificate is retained as an open node, never pruned.
    """
    if data.tau not in (1.5, 2., 3.):
        raise ValueError('The SDP search supports tau=1.5, 2, 3')
    if method not in ('PW', 'LR-D', 'LR-A', 'LR-SA'):
        raise ValueError(f'Unsupported SDP method: {method}')
    if time_limit <= 0 or node_time_limit <= 0 or root_time_limit <= 0:
        raise ValueError('Time limits must be positive')
    started = perf_counter()
    initial = warm_start
    if initial is None:
        initial = build_warm_start(data, WarmStartOptions(strategy='multistart', random_starts=1, alternating_iterations=3))
    best = math.inf if initial is None else initial.objective
    labels = None if initial is None or initial.labels is None else initial.labels.copy()
    centers = None if initial is None or initial.centers is None else initial.centers.copy()
    baseline = 0. if initial is None else max(0., float(initial.metadata.get('sdp_lower_bound', 0.)))
    if not math.isfinite(baseline):
        baseline = 0.
    heap = [(baseline, 0, (), ())]
    serial, nodes, infeasible, unknown, terminal_count = 0, 0, 0, 0, 0
    triangles = set()
    history = []
    diagnostics = []
    closed_floor = math.inf
    root_bound, root_runtime = baseline, 0.
    status = 'TIME_LIMIT'
    def tolerance():
        return max(absolute_gap, mip_gap*abs(best)) if math.isfinite(best) else absolute_gap
    def remaining():
        return time_limit-(perf_counter()-started)
    def update(candidate):
        nonlocal best, labels, centers
        if candidate is not None and candidate['objective'] < best and candidate['centers'] is not None:
            best, labels, centers = candidate['objective'], candidate['labels'].copy(), candidate['centers'].copy()
    while heap and remaining() > 0 and (maximum_nodes is None or nodes < maximum_nodes):
        inherited, _, must, cannot = heapq.heappop(heap)
        if inherited >= best-tolerance():
            closed_floor = min(closed_floor, inherited)
            continue
        groups = components(data.n, must, cannot, data.facilities)
        if groups is None:
            infeasible += 1
            continue
        nodes += 1
        node_bound, P = inherited, None
        if len(groups) == data.facilities:
            terminal_count += 1
            fixed = np.empty(data.n, dtype=int)
            for j, group in enumerate(groups):
                fixed[list(group)] = j
            candidate = partition_certificate(data, fixed, min(node_time_limit, max(0., remaining())), log_output)
            update(candidate)
            if candidate is not None and candidate['lower_bound'] is not None:
                node_bound = max(node_bound, candidate['lower_bound'])
            if node_bound >= best-tolerance():
                closed_floor = min(closed_floor, node_bound)
                continue
            # No unsupported declaration of exactness on an unknown/loose SOCP.
            serial += 1
            heapq.heappush(heap, (node_bound, serial, must, cannot))
            status = 'SUBPROBLEM_LIMIT'
            break
        budget = min(root_time_limit if nodes == 1 else node_time_limit, max(0., remaining()))
        node_started = perf_counter()
        result = solve_node(data, method, must, cannot, triangles, budget, log_output=log_output)
        if nodes == 1:
            root_runtime = perf_counter()-node_started
        if result['infeasible']:
            infeasible += 1
            continue
        if result['bound'] is not None:
            node_bound = max(node_bound, result['bound'])
        else:
            unknown += 1
            diagnostics.append(result['diagnostic'])
        P = result['P']
        if nodes == 1:
            root_bound = node_bound
        if P is not None and remaining() > 0 and (nodes == 1 or nodes % 10 == 0):
            candidate = partition_certificate(data, spectral_labels(P, data), min(5., max(0., remaining())), log_output)
            update(candidate)
        if node_bound >= best-tolerance():
            closed_floor = min(closed_floor, node_bound)
        else:
            pair = branching_pair(P, groups, cannot)
            if pair is None:
                infeasible += 1
            else:
                i, h = pair
                serial += 1
                heapq.heappush(heap, (node_bound, serial, must+((i, h),), cannot))
                serial += 1
                heapq.heappush(heap, (node_bound, serial, must, cannot+(tuple(sorted((i, h))),)))
        global_bound = min([closed_floor, best] + ([heap[0][0]] if heap else []))
        history.append(dict(iteration=float(nodes), lower_bound=global_bound, upper_bound=best))
        if log_output:
            print(f'{method}: nodes={nodes}, open={len(heap)}, LB={global_bound:g}, UB={best:g}, triangles={len(triangles)}', flush=True)
    lower = min([closed_floor, best] + ([heap[0][0]] if heap else []))
    if not heap:
        status = 'OPTIMAL' if math.isfinite(best) else 'INFEASIBLE'
    elif maximum_nodes is not None and nodes >= maximum_nodes:
        status = 'NODE_LIMIT'
    return Results(method=method, status=status, objective=best, lower_bound=min(lower, best),
                   runtime=perf_counter()-started, labels=labels, centers=centers,
                   iterations=nodes, optimal=status == 'OPTIMAL', history=history,
                   metadata=dict(engine=ENGINE_VERSION, tau=data.tau, center_region='convex_hull' if data.tau==2. else 'coordinate_box', relaxation={'PW': 'pairwise', 'LR-D': 'disaggregated_matrix_perspective',
                                             'LR-A': 'block_diagonal_matrix_perspective',
                                             'LR-SA': 'sum_aggregated_matrix_perspective'}[method],
                                 sdp_root_bound=root_bound, sdp_bound_time=root_runtime,
                                 continuous_relaxation_bound=root_bound, continuous_relaxation_time=root_runtime,
                                 continuous_relaxation_gap=_relative_bound_gap(best, root_bound),
                                 sdp_root_gap=_relative_bound_gap(best, root_bound), symmetry_breaking=True,
                                 sdp_nodes=nodes, open_nodes=len(heap), infeasible_nodes=infeasible,
                                 unknown_sdp_nodes=unknown, terminal_partitions=terminal_count,
                                 partition_triangle_cuts=len(triangles), projection_logic_cuts=0,
                                 evaluated_partitions=terminal_count, evaluated_clusters=0,
                                 diagnostics=diagnostics[-10:]))
