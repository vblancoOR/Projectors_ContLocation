"""Optimization models for continuous multifacility location.

This module accompanies the paper *Projector-Based Conic Formulations and
Exact Decomposition for Continuous Multifacility Location*.  It contains
the compact, PW, LR-D, LR-A, and LR-SA solution methods:

``compact``
    A big-M mixed-integer second-order cone formulation solved by Gurobi.
``pairwise``
    An exact strengthened decomposition of the pairwise projector model. A
    Gurobi partition master calls MOSEK for the conic location subproblem.
``LR-D`` (``solve_lr_d``; legacy alias ``solve_mpco``)
    An exact disaggregated projector decomposition.  It stores one value cut
    for every evaluated facility/subset pair, which reuses information across
    partitions and facility labels.
``LR-A`` (``solve_lr_a``)
    The block-diagonal aggregated model; its reduced labeled master and
    conic value oracle coincide with PW.
``LR-SA`` (``solve_lr_sa``; legacy alias ``solve_mpco_sa``)
    The exact sum-aggregated MPCO outer approximation.  Its master uses one
    epigraph variable and an unlabeled co-assignment encoding of the rank-p
    partition projector.

The computational implementation represents a partition by binary incidence
vectors in the Gurobi masters.  These vectors and their rank-one projectors
are equivalent by the theorems in the paper.  Using incidence vectors avoids
asking a general-purpose solver to rediscover their binary structure from
nonconvex projector equations.

All models accept the norm order ``tau`` in ``{1, 3/2, 2, 3, infinity}``.
The fractional orders use exact lifted rotated second-order cones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Literal, Sequence

import math
import warnings

import numpy as np


WarmStartName = Literal["none", "multistart", "alternating", "sdp", "best"]
MethodName = Literal["compact", "PW", "LR-D", "LR-A", "LR-SA", "pairwise", "mpco", "mpco_sa"]
SUPPORTED_NORMS = (1.0, 1.5, 2.0, 3.0, math.inf)


def _normalize_norm_order(value: float | str) -> float:
    """Return a canonical norm order and reject unsupported values."""

    if isinstance(value, str):
        key = value.strip().lower().replace("∞", "inf")
        if key in {"inf", "infinity", "+inf", "+infinity"}:
            return math.inf
        try:
            value = float(key)
        except ValueError as exc:
            raise ValueError("tau must be one of 1, 3/2, 2, 3, or infinity") from exc
    order = float(value)
    for supported in SUPPORTED_NORMS:
        if (math.isinf(order) and math.isinf(supported)) or math.isclose(order, supported):
            return supported
    raise ValueError("tau must be one of 1, 3/2, 2, 3, or infinity")


def _norm(values: np.ndarray, tau: float, axis=None) -> np.ndarray:
    return np.linalg.norm(values, ord=tau, axis=axis)


@dataclass(slots=True)
class Data:
    """Input data for a continuous multifacility location instance.

    Parameters
    ----------
    points:
        Array with one demand point per row and one coordinate per column.
    facilities:
        Number of facilities.  Every facility must serve at least one point.
    weights:
        Optional nonnegative demand weights.  Unit weights are used by default.
    name:
        Human-readable identifier used in tables and figures.
    tau:
        Norm order.  Accepted values are 1, 3/2, 2, 3, and infinity.
    """

    points: np.ndarray
    facilities: int
    weights: np.ndarray | None = None
    name: str = "instance"
    tau: float | str = 2.0

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[0] == 0:
            raise ValueError("points must be a nonempty two-dimensional array")
        if not np.all(np.isfinite(self.points)):
            raise ValueError("points must contain only finite values")
        if not 1 <= int(self.facilities) <= self.points.shape[0]:
            raise ValueError("facilities must be between 1 and the number of points")
        self.facilities = int(self.facilities)
        if self.weights is None:
            self.weights = np.ones(self.points.shape[0], dtype=float)
        else:
            self.weights = np.asarray(self.weights, dtype=float)
        if self.weights.shape != (self.points.shape[0],):
            raise ValueError("weights must have one entry per demand point")
        if np.any(self.weights < 0) or not np.all(np.isfinite(self.weights)):
            raise ValueError("weights must be finite and nonnegative")
        if float(self.weights.sum()) <= 0:
            raise ValueError("at least one demand weight must be positive")
        self.tau = _normalize_norm_order(self.tau)

    @property
    def n(self) -> int:
        return self.points.shape[0]

    @property
    def dimension(self) -> int:
        return self.points.shape[1]

    @classmethod
    def blobs(
        cls,
        n: int = 30,
        facilities: int = 3,
        dimension: int = 2,
        spread: float = 0.65,
        seed: int = 0,
        name: str | None = None,
        tau: float | str = 2.0,
    ) -> "Data":
        """Generate a reproducible Gaussian-mixture test instance."""

        if n < facilities:
            raise ValueError("n must be at least facilities")
        rng = np.random.default_rng(seed)
        centers = rng.uniform(-5.0, 5.0, size=(facilities, dimension))
        labels = np.arange(n) % facilities
        rng.shuffle(labels)
        points = centers[labels] + rng.normal(scale=spread, size=(n, dimension))
        return cls(points, facilities, name=name or f"blobs_n{n}_p{facilities}_s{seed}", tau=tau)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        facilities: int,
        tau: float | str = 2.0,
        name: str | None = None,
    ) -> "Data":
        """Load a standardized ``id,x,y,weight`` benchmark CSV file."""

        import csv

        coordinates: list[list[float]] = []
        weights: list[float] = []
        source = Path(path)
        with source.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not {"x", "y"}.issubset(reader.fieldnames):
                raise ValueError("instance CSV must contain x and y columns")
            for row in reader:
                coordinates.append([float(row["x"]), float(row["y"])])
                weights.append(float(row.get("weight", 1.0) or 1.0))
        return cls(
            np.asarray(coordinates),
            facilities,
            np.asarray(weights),
            name or source.stem,
            tau,
        )


@dataclass(slots=True)
class Results:
    """Common result returned by every exact or heuristic method."""

    method: str
    status: str
    objective: float = math.inf
    lower_bound: float = -math.inf
    runtime: float = 0.0
    labels: np.ndarray | None = None
    centers: np.ndarray | None = None
    iterations: int = 0
    optimal: bool = False
    history: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def gap(self) -> float:
        if not np.isfinite(self.objective) or not np.isfinite(self.lower_bound):
            return math.inf
        return max(0.0, (self.objective - self.lower_bound) / max(1.0, abs(self.objective)))

    def summary(self) -> dict[str, Any]:
        """Return a flat record convenient for pandas or CSV output."""

        return {
            "method": self.method,
            "status": self.status,
            "objective": self.objective,
            "lower_bound": self.lower_bound,
            "gap": self.gap,
            "runtime": self.runtime,
            "iterations": self.iterations,
            "optimal": self.optimal,
        }


@dataclass(slots=True)
class WarmStartOptions:
    """Parameters shared by all warm-start generators."""

    strategy: WarmStartName = "best"
    random_starts: int = 10
    alternating_iterations: int = 50
    tolerance: float = 1e-7
    seed: int = 0
    sdp_time_limit: float | None = 60.0
    log_output: bool = False


def assignment_projectors(labels: Sequence[int], facilities: int) -> list[np.ndarray]:
    """Return the rank-one projectors associated with a labeled partition."""

    labels = np.asarray(labels, dtype=int)
    projectors: list[np.ndarray] = []
    for j in range(facilities):
        z = (labels == j).astype(float)
        cardinality = float(z.sum())
        if cardinality == 0:
            raise ValueError("every facility must serve at least one demand point")
        projectors.append(np.outer(z, z) / cardinality)
    return projectors


def objective_value(data: Data, labels: Sequence[int], centers: np.ndarray) -> float:
    """Evaluate a feasible location-allocation solution."""

    labels = np.asarray(labels, dtype=int)
    centers = np.asarray(centers, dtype=float)
    distances = _norm(data.points - centers[labels], data.tau, axis=1)
    return float(data.weights @ distances)


def _require_gurobi():
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError("Gurobi and gurobipy are required for this method") from exc
    return gp, GRB


def _require_mosek():
    try:
        from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError("MOSEK Fusion is required for this method") from exc
    return Domain, Expr, Matrix, Model, ObjectiveSense


def _repair_nonempty(labels: np.ndarray, data: Data) -> np.ndarray:
    """Repair empty labels by moving far points from clusters of size > 1."""

    labels = np.asarray(labels, dtype=int).copy()
    counts = np.bincount(labels, minlength=data.facilities)
    for empty in np.flatnonzero(counts == 0):
        donors = np.flatnonzero(counts[labels] > 1)
        if donors.size == 0:
            raise RuntimeError("cannot repair the partition")
        donor_centers = np.vstack(
            [data.points[labels == j].mean(axis=0) for j in range(data.facilities) if counts[j] > 0]
        )
        active = np.flatnonzero(counts > 0)
        center_map = {j: donor_centers[k] for k, j in enumerate(active)}
        scores = np.array(
            [_norm(data.points[i] - center_map[labels[i]], data.tau) for i in donors]
        )
        chosen = int(donors[int(np.argmax(scores))])
        old = int(labels[chosen])
        labels[chosen] = empty
        counts[old] -= 1
        counts[empty] += 1
    return labels


def _seed_partition(data: Data, rng: np.random.Generator) -> np.ndarray:
    """Farthest-first seed selection followed by closest-center assignment."""

    first = int(rng.integers(data.n))
    seeds = [first]
    nearest = _norm(data.points - data.points[first], data.tau, axis=1)
    while len(seeds) < data.facilities:
        remaining = np.setdiff1d(np.arange(data.n), np.asarray(seeds), assume_unique=True)
        next_seed = int(remaining[np.argmax(nearest[remaining])])
        seeds.append(next_seed)
        nearest = np.minimum(nearest, _norm(data.points - data.points[next_seed], data.tau, axis=1))
    distances = _norm(data.points[:, None, :] - data.points[seeds][None, :, :], data.tau, axis=2)
    return _repair_nonempty(np.argmin(distances, axis=1), data)


class SubproblemTimeLimit(RuntimeError):
    """A conic value could not be certified within the remaining budget."""


def solve_single_facility(
    points: np.ndarray,
    weights: np.ndarray,
    tau: float | str = 2.0,
    time_limit: float | None = None,
    log_output: bool = False,
) -> tuple[np.ndarray, float]:
    """Solve a weighted ell-p single-facility problem with MOSEK."""

    if time_limit is not None and time_limit <= 0:
        raise SubproblemTimeLimit("No time remains for the conic location subproblem")
    from mosek.fusion import SolutionError
    import mosek
    Domain, Expr, _Matrix, Model, ObjectiveSense = _require_mosek()
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    tau = _normalize_norm_order(tau)
    if points.shape[0] == 1:
        return points[0].copy(), 0.0
    with Model("single_facility") as model:
        if log_output:
            model.setLogHandler(None)
        if time_limit is not None:
            model.setSolverParam("optimizerMaxTime", max(float(time_limit), 0.01))
        x = model.variable("x", points.shape[1], Domain.unbounded())
        t = model.variable("t", points.shape[0], Domain.greaterThan(0.0))
        for i, point in enumerate(points):
            residual = Expr.sub(x, point.tolist())
            if tau == 2.0:
                model.constraint(Expr.vstack(t.index(i), residual), Domain.inQCone())
            elif tau == 1.0:
                absolute = model.variable(f"abs_{i}", points.shape[1], Domain.greaterThan(0.0))
                model.constraint(Expr.sub(absolute, residual), Domain.greaterThan(0.0))
                model.constraint(Expr.add(absolute, residual), Domain.greaterThan(0.0))
                model.constraint(Expr.sub(t.index(i), Expr.sum(absolute)), Domain.greaterThan(0.0))
            elif math.isinf(tau):
                for k in range(points.shape[1]):
                    model.constraint(Expr.sub(t.index(i), residual.index(k)), Domain.greaterThan(0.0))
                    model.constraint(Expr.add(t.index(i), residual.index(k)), Domain.greaterThan(0.0))
            else:
                auxiliary = model.variable(f"soc_scale_{i}", points.shape[1], Domain.greaterThan(0.0))
                model.constraint(Expr.sub(t.index(i), Expr.sum(auxiliary)), Domain.greaterThan(0.0))
                for k in range(points.shape[1]):
                    absolute = model.variable(f"soc_abs_{i}_{k}", 1, Domain.greaterThan(0.0))
                    mediator = model.variable(f"soc_med_{i}_{k}", 1, Domain.greaterThan(0.0))
                    model.constraint(Expr.sub(absolute, residual.index(k)), Domain.greaterThan(0.0))
                    model.constraint(Expr.add(absolute, residual.index(k)), Domain.greaterThan(0.0))
                    # A rotated cone (a,b,sqrt(2)c) imposes c^2 <= ab.
                    # tau=3:   u^2 <= t v, v^2 <= s u  => u^3 <= s t^2.
                    # tau=3/2: u^2 <= s v, v^2 <= t u  => u^3 <= s^2 t.
                    first = t.index(i) if tau == 3.0 else auxiliary.index(k)
                    second = auxiliary.index(k) if tau == 3.0 else t.index(i)
                    model.constraint(
                        Expr.vstack(first, mediator, Expr.mul(math.sqrt(2.0), absolute)),
                        Domain.inRotatedQCone(),
                    )
                    model.constraint(
                        Expr.vstack(second, absolute, Expr.mul(math.sqrt(2.0), mediator)),
                        Domain.inRotatedQCone(),
                    )
        model.objective(ObjectiveSense.Minimize, Expr.dot(weights.tolist(), t))
        model.solve()
        try:
            center = np.asarray(x.level(), dtype=float)
            return center, float(model.primalObjValue())
        except SolutionError as error:
            response = int(model.getSolverIntInfo("optimizeResponse"))
            diagnostic = (f"MOSEK location subproblem: problem={model.getProblemStatus()}, "
                          f"primal={model.getPrimalSolutionStatus()}, "
                          f"termination={response}, budget={time_limit}s")
            if response == int(mosek.rescode.trm_max_time):
                raise SubproblemTimeLimit(diagnostic) from error
            # Unknown can also indicate numerical trouble. Keep it as an
            # error, with the termination code, rather than call it a timeout.
            raise RuntimeError(diagnostic) from error


def solve_fixed_partition(
    data: Data,
    labels: Sequence[int],
    time_limit: float | None = None,
    log_output: bool = False,
) -> tuple[np.ndarray, float, list[float]]:
    """Optimize all facility locations for a fixed nonempty partition."""

    labels = _repair_nonempty(np.asarray(labels, dtype=int), data)
    centers = np.empty((data.facilities, data.dimension), dtype=float)
    costs: list[float] = []
    started = perf_counter()
    for j in range(data.facilities):
        remaining = None if time_limit is None else time_limit - (perf_counter() - started)
        if remaining is not None and remaining <= 0:
            raise SubproblemTimeLimit("Partition evaluation exhausted its time budget")
        mask = labels == j
        centers[j], cost = solve_single_facility(
            data.points[mask], data.weights[mask], data.tau, remaining, log_output
        )
        costs.append(cost)
    return centers, float(sum(costs)), costs


def alternating_warm_start(
    data: Data,
    labels: Sequence[int],
    maximum_iterations: int = 50,
    tolerance: float = 1e-7,
    time_limit: float | None = None,
    log_output: bool = False,
) -> Results:
    """Alternate exact location updates and closest-facility assignments."""

    started = perf_counter()
    labels = _repair_nonempty(np.asarray(labels, dtype=int), data)
    best_objective = math.inf
    best_labels = labels.copy()
    best_centers: np.ndarray | None = None
    history: list[dict[str, float]] = []
    for iteration in range(1, maximum_iterations + 1):
        remaining = None if time_limit is None else max(0.01, time_limit - (perf_counter() - started))
        centers, value, _ = solve_fixed_partition(data, labels, remaining, log_output)
        history.append({"iteration": float(iteration), "upper_bound": value})
        if value + tolerance < best_objective:
            best_objective = value
            best_labels = labels.copy()
            best_centers = centers.copy()
        distances = _norm(data.points[:, None, :] - centers[None, :, :], data.tau, axis=2)
        new_labels = _repair_nonempty(np.argmin(distances, axis=1), data)
        if np.array_equal(new_labels, labels):
            break
        if iteration > 1 and abs(history[-2]["upper_bound"] - value) <= tolerance:
            break
        labels = new_labels
        if time_limit is not None and perf_counter() - started >= time_limit:
            break
    return Results(
        method="alternating",
        status="FEASIBLE",
        objective=best_objective,
        runtime=perf_counter() - started,
        labels=best_labels,
        centers=best_centers,
        iterations=len(history),
        history=history,
    )


def sdp_warm_start(
    data: Data,
    time_limit: float | None = 60.0,
    log_output: bool = False,
) -> Results:
    """Solve the paper's doubly-nonnegative bound, round, and refine it."""

    started = perf_counter()
    lower_bound, scores, relaxation_time = solve_sdp_relaxation(data, time_limit, log_output)
    labels = _round_scores_nonempty(scores, data)
    remaining = None if time_limit is None else max(0.01, time_limit - (perf_counter() - started))
    refined = alternating_warm_start(data, labels, time_limit=remaining, log_output=log_output)
    refined.method = "sdp_rounding"
    refined.lower_bound = lower_bound
    refined.runtime = perf_counter() - started
    refined.metadata["fractional_scores"] = scores
    refined.metadata["sdp_relaxation_time"] = relaxation_time
    refined.metadata["sdp_lower_bound"] = lower_bound
    return refined


def solve_sdp_relaxation(
    data: Data,
    time_limit: float | None = 60.0,
    log_output: bool = False,
) -> tuple[float, np.ndarray, float]:
    """Return a certified DNN lower bound and fractional assignment scores.

    The dual objective is used as the certified lower bound.  This distinction
    matters when MOSEK stops at its time limit because the primal relaxation
    value need not be a valid lower bound on the original minimization model.
    """

    Domain, Expr, Matrix, Model, ObjectiveSense = _require_mosek()
    started = perf_counter()
    pairwise = _norm(data.points[:, None, :] - data.points[None, :, :], data.tau, axis=2)
    weighted = np.minimum(data.weights[:, None], data.weights[None, :]) * pairwise
    with Model("sdp_warm_start") as model:
        if log_output:
            model.setLogHandler(None)
        if time_limit is not None:
            model.setSolverParam("optimizerMaxTime", max(float(time_limit), 0.01))
        projectors = [model.variable(f"P{j}", Domain.inPSDCone(data.n)) for j in range(data.facilities)]
        for matrix in projectors:
            model.constraint(matrix, Domain.greaterThan(0.0))
            model.constraint(Expr.sum(matrix.diag()), Domain.equalsTo(1.0))
        rows = [Expr.mul(matrix, np.ones(data.n).tolist()) for matrix in projectors]
        model.constraint(Expr.add(rows), Domain.equalsTo(np.ones(data.n).tolist()))
        # Partial symmetry breaking in projector form. For i < facilities,
        # demand point i may use only facilities 0,...,i. At an integral
        # projector, (P_j e)_i is exactly the assignment indicator z[i,j].
        for i in range(min(data.n, data.facilities)):
            permitted = [rows[j].index(i) for j in range(i + 1)]
            model.constraint(Expr.add(permitted), Domain.equalsTo(1.0))
        terms = [Expr.dot(Matrix.dense(weighted), matrix) for matrix in projectors]
        model.objective(ObjectiveSense.Minimize, Expr.mul(0.5, Expr.add(terms)))
        model.solve()
        lower_bound = float(model.dualObjValue())
        levels = [np.asarray(matrix.level(), dtype=float).reshape(data.n, data.n) for matrix in projectors]
    scores = np.column_stack([matrix @ np.ones(data.n) for matrix in levels])
    return lower_bound, scores, perf_counter() - started


def _round_scores_nonempty(scores: np.ndarray, data: Data) -> np.ndarray:
    """Maximize agreement with fractional scores subject to nonempty clusters."""

    gp, GRB = _require_gurobi()
    model = gp.Model("round_sdp")
    model.Params.OutputFlag = 0
    z = model.addVars(data.n, data.facilities, vtype=GRB.BINARY, name="z")
    model.addConstrs((gp.quicksum(z[i, j] for j in range(data.facilities)) == 1 for i in range(data.n)))
    model.addConstrs((gp.quicksum(z[i, j] for i in range(data.n)) >= 1 for j in range(data.facilities)))
    model.setObjective(
        gp.quicksum(float(scores[i, j]) * z[i, j] for i in range(data.n) for j in range(data.facilities)),
        GRB.MAXIMIZE,
    )
    model.optimize()
    labels = np.array([max(range(data.facilities), key=lambda j: z[i, j].X) for i in range(data.n)])
    model.dispose()
    return labels


def build_warm_start(data: Data, options: WarmStartOptions | None = None) -> Results | None:
    """Generate the requested warm start, or the best of all available starts."""

    options = options or WarmStartOptions()
    if options.strategy == "none":
        return None
    rng = np.random.default_rng(options.seed)
    candidates: list[Results] = []
    if options.strategy in {"multistart", "alternating", "best"}:
        starts = max(1, options.random_starts if options.strategy != "alternating" else 1)
        for _ in range(starts):
            labels = _seed_partition(data, rng)
            candidates.append(
                alternating_warm_start(
                    data,
                    labels,
                    options.alternating_iterations,
                    options.tolerance,
                    log_output=options.log_output,
                )
            )
    if options.strategy in {"sdp", "best"}:
        try:
            candidates.append(sdp_warm_start(data, options.sdp_time_limit, options.log_output))
        except Exception as exc:
            if options.strategy == "sdp":
                raise
            warnings.warn(f"SDP warm start unavailable: {exc}", RuntimeWarning)
    if not candidates:
        return None
    best = min(candidates, key=lambda result: result.objective)
    certified_bounds = [candidate.lower_bound for candidate in candidates if np.isfinite(candidate.lower_bound)]
    if certified_bounds:
        best.lower_bound = max(certified_bounds)
    sdp_bounds = [candidate.metadata["sdp_lower_bound"] for candidate in candidates
                  if np.isfinite(candidate.metadata.get("sdp_lower_bound", math.nan))]
    if sdp_bounds:
        best.metadata["sdp_lower_bound"] = max(sdp_bounds)
    best.metadata["warm_start_candidates"] = [candidate.summary() for candidate in candidates]
    return best


def solve_compact(
    data: Data,
    time_limit: float = 300.0,
    mip_gap: float = 1e-4,
    warm_start: WarmStartOptions | Results | None = None,
    compute_relaxation: bool = True,
    relaxation_time_limit: float = 60.0,
    log_output: bool = False,
    use_sdp_bound: bool = False,
    sdp_bound_time_limit: float = 60.0,
    use_symmetry_breaking: bool = False,
) -> Results:
    """Solve the compact big-M conic/MINLP formulation with Gurobi.

    If the time limit is reached, the best incumbent is returned together
    with Gurobi's best lower bound.
    """

    gp, GRB = _require_gurobi()
    started = perf_counter()
    initial = warm_start if isinstance(warm_start, Results) else build_warm_start(data, warm_start)
    model = gp.Model("compact_big_m")
    model.Params.OutputFlag = int(log_output)
    model.Params.TimeLimit = max(float(time_limit), 0.01)
    model.Params.MIPGap = float(mip_gap)
    lower = data.points.min(axis=0)
    upper = data.points.max(axis=0)
    x = model.addVars(data.facilities, data.dimension, lb={(j, k): float(lower[k]) for j in range(data.facilities) for k in range(data.dimension)}, ub={(j, k): float(upper[k]) for j in range(data.facilities) for k in range(data.dimension)}, name="x")
    # Hull restriction for the existing norms; tau=1.5/3 instead use
    # coordinate-box bounds and their corresponding farthest-corner big-M.
    convex_weight = (model.addVars(data.facilities, data.n, lb=0.0, name="lambda")
                     if data.tau not in {1.5, 3.0} else None)
    z = model.addVars(data.n, data.facilities, vtype=GRB.BINARY, name="z")
    distance = model.addVars(data.n, data.facilities, lb=0.0, name="distance")
    active = model.addVars(data.n, data.facilities, lb=0.0, name="active")
    residual = model.addVars(data.n, data.facilities, data.dimension, lb=-GRB.INFINITY, name="residual")
    model.addConstrs((gp.quicksum(z[i, j] for j in range(data.facilities)) == 1 for i in range(data.n)))
    model.addConstrs((gp.quicksum(z[i, j] for i in range(data.n)) >= 1 for j in range(data.facilities)))
    if convex_weight is not None:
        model.addConstrs((gp.quicksum(convex_weight[j, h] for h in range(data.n)) == 1 for j in range(data.facilities)))
        model.addConstrs((x[j, k] == gp.quicksum(float(data.points[h, k]) * convex_weight[j, h] for h in range(data.n)) for j in range(data.facilities) for k in range(data.dimension)))
    for i in range(data.n):
        for j in range(data.facilities):
            for k in range(data.dimension):
                model.addConstr(residual[i, j, k] == x[j, k] - float(data.points[i, k]))
            components = [residual[i, j, k] for k in range(data.dimension)]
            if data.tau in {1.0, 2.0} or math.isinf(data.tau):
                order = GRB.INFINITY if math.isinf(data.tau) else data.tau
                model.addGenConstrNorm(distance[i, j], components, order)
            else:
                # Exact rotated-SOC lifting of the rational norm epigraph.
                s = model.addVars(data.dimension, lb=0.0, name=f"soc_scale_{i}_{j}")
                model.addConstr(gp.quicksum(s[k] for k in range(data.dimension)) <= distance[i, j])
                for k in range(data.dimension):
                    absolute = model.addVar(lb=0.0, name=f"soc_abs_{i}_{j}_{k}")
                    mediator = model.addVar(lb=0.0, name=f"soc_med_{i}_{j}_{k}")
                    model.addConstr(absolute >= residual[i, j, k])
                    model.addConstr(absolute >= -residual[i, j, k])
                    first = distance[i, j] if data.tau == 3.0 else s[k]
                    second = s[k] if data.tau == 3.0 else distance[i, j]
                    model.addQConstr(absolute * absolute <= first * mediator)
                    model.addQConstr(mediator * mediator <= second * absolute)
    if data.tau in {1.5, 3.0}:
        # Farthest coordinate-box corner: valid without a Euclidean hull assumption.
        big_m = _norm(np.maximum(np.abs(data.points-lower), np.abs(data.points-upper)), data.tau, axis=1)
    else:
        big_m = _norm(data.points[:, None, :] - data.points[None, :, :], data.tau, axis=2).max(axis=1)
    model.addConstrs((active[i, j] >= distance[i, j] - float(big_m[i]) * (1 - z[i, j]) for i in range(data.n) for j in range(data.facilities)))
    model.setObjective(gp.quicksum(float(data.weights[i]) * active[i, j] for i in range(data.n) for j in range(data.facilities)), GRB.MINIMIZE)
    if use_symmetry_breaking:
        for i in range(min(data.n, data.facilities)):
            model.addConstr(gp.quicksum(z[i, j] for j in range(i + 1)) == 1,
                            name=f"symmetry_prefix_{i}")
    sdp_bound, sdp_time = _obtain_sdp_bound(
        data, initial, use_sdp_bound,
        min(sdp_bound_time_limit, max(0.01, _remaining(started, time_limit))), log_output)
    if use_sdp_bound:
        model.addConstr(gp.quicksum(float(data.weights[i]) * active[i, j]
                                   for i in range(data.n) for j in range(data.facilities))
                        >= sdp_bound, name="certified_sdp_bound")
    if initial is not None and initial.labels is not None:
        order = list(dict.fromkeys(int(label) for label in initial.labels))
        _set_partition_start(z, initial.labels, data)
        if initial.centers is not None:
            for j in range(data.facilities):
                for k in range(data.dimension):
                    x[j, k].Start = float(initial.centers[order[j], k])
    relaxation_bound = -math.inf
    relaxation_time = 0.0
    if compute_relaxation and _remaining(started, time_limit) > 0:
        relaxation_bound, relaxation_time = _solve_gurobi_relaxation(
            model,
            min(float(relaxation_time_limit), _remaining(started, time_limit)),
            log_output,
        )
    model.Params.TimeLimit = max(0.01, _remaining(started, time_limit))
    model.optimize()
    runtime = perf_counter() - started
    status = _gurobi_status(model.Status, GRB)
    if model.SolCount:
        labels = np.array([max(range(data.facilities), key=lambda j: z[i, j].X) for i in range(data.n)])
        centers = np.array([[x[j, k].X for k in range(data.dimension)] for j in range(data.facilities)])
        # Re-evaluate the incumbent in the original norm to remove any small
        # discrepancy caused by conic feasibility tolerances.
        objective = objective_value(data, labels, centers)
        if initial is not None and initial.objective < objective:
            labels, centers, objective = initial.labels, initial.centers, initial.objective
    elif initial is not None:
        labels, centers, objective = initial.labels, initial.centers, initial.objective
    else:
        labels, centers, objective = None, None, math.inf
    bound = float(model.ObjBound) if model.SolCount or model.Status in {GRB.TIME_LIMIT, GRB.OPTIMAL} else -math.inf
    result = Results(
        method="compact_big_m",
        status=status,
        objective=objective,
        lower_bound=bound,
        runtime=runtime,
        labels=labels,
        centers=centers,
        optimal=model.Status == GRB.OPTIMAL,
        metadata={
            "big_m": big_m,
            "sdp_root_bound": sdp_bound,
            "sdp_bound_time": sdp_time,
            "symmetry_breaking": use_symmetry_breaking,
            "node_count": float(model.NodeCount),
            "tau": data.tau,
            "solver_objective": float(model.ObjVal) if model.SolCount else math.inf,
            "continuous_relaxation_bound": relaxation_bound,
            "continuous_relaxation_time": relaxation_time,
            "continuous_relaxation_gap": _relative_bound_gap(objective, relaxation_bound),
        },
    )
    model.dispose()
    return result


def _gurobi_status(status: int, GRB: Any) -> str:
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return names.get(status, f"GUROBI_STATUS_{status}")


def _build_partition_master(
    data: Data,
    name: str,
    log_output: bool,
    use_symmetry_breaking: bool = True,
):
    gp, GRB = _require_gurobi()
    model = gp.Model(name)
    model.Params.OutputFlag = int(log_output)
    z = model.addVars(data.n, data.facilities, vtype=GRB.BINARY, name="z")
    model.addConstrs((gp.quicksum(z[i, j] for j in range(data.facilities)) == 1 for i in range(data.n)))
    model.addConstrs((gp.quicksum(z[i, j] for i in range(data.n)) >= 1 for j in range(data.facilities)))
    if use_symmetry_breaking:
        # If point i is not assigned to an earlier facility, assign it to
        # facility i. Together with row assignment, this is equivalently
        # sum_{j=0}^i z[i,j] = 1 for i=0,...,facilities-1.
        for i in range(min(data.n, data.facilities)):
            model.addConstr(
                gp.quicksum(z[i, j] for j in range(i + 1)) == 1,
                name=f"symmetry_prefix_{i}",
            )
    return gp, GRB, model, z


def _build_aggregate_partition_master(
    data: Data,
    name: str,
    log_output: bool,
    use_symmetry_breaking: bool = True,
):
    """Build an unlabeled finite master for the aggregate projector.

    The binary variable ``same[i,h]`` is one exactly when demand points i and
    h belong to the same block.  Therefore, the complete ``same`` vector
    identifies the aggregate partition projector independently of the labels
    used by the incidence variables ``z``.  This is the computational
    counterpart of the rank-p projector P in MPCO-SA.
    """

    gp, GRB, model, z = _build_partition_master(
        data, name, log_output, use_symmetry_breaking
    )
    same = model.addVars(
        ((i, h) for i in range(data.n) for h in range(i + 1, data.n)),
        vtype=GRB.BINARY,
        name="same",
    )
    for i in range(data.n):
        for h in range(i + 1, data.n):
            # At most one j can be active in each row. These constraints
            # impose same[i,h]=1 iff the two active facility labels coincide.
            for j in range(data.facilities):
                model.addConstr(
                    same[i, h] >= z[i, j] + z[h, j] - 1,
                    name=f"same_lb_{i}_{h}_{j}",
                )
                model.addConstr(
                    same[i, h] <= 1 - z[i, j] + z[h, j],
                    name=f"same_ub_{i}_{h}_{j}",
                )
    return gp, GRB, model, z, same


def _solve_gurobi_relaxation(
    model: Any,
    time_limit: float,
    log_output: bool,
) -> tuple[float, float]:
    """Solve the continuous relaxation of a constructed Gurobi model."""

    started = perf_counter()
    model.update()
    relaxation = model.relax()
    relaxation.Params.OutputFlag = int(log_output)
    relaxation.Params.TimeLimit = max(0.01, float(time_limit))
    relaxation.optimize()
    bound = float(relaxation.ObjVal) if relaxation.SolCount else -math.inf
    relaxation.dispose()
    return bound, perf_counter() - started


def _relative_bound_gap(upper_bound: float, lower_bound: float) -> float:
    if not np.isfinite(upper_bound) or not np.isfinite(lower_bound):
        return math.inf
    return max(0.0, (upper_bound - lower_bound) / max(1.0, abs(upper_bound)))


def _obtain_sdp_bound(
    data: Data,
    initial: Results | None,
    use_sdp_bound: bool,
    time_limit: float,
    log_output: bool,
) -> tuple[float, float]:
    """Reuse a warm-start SDP bound or solve the relaxation once."""

    if not use_sdp_bound:
        return 0.0, 0.0
    if initial is not None and np.isfinite(initial.metadata.get("sdp_lower_bound", math.nan)):
        return max(0.0, float(initial.metadata["sdp_lower_bound"])), 0.0
    bound, _scores, runtime = solve_sdp_relaxation(data, time_limit, log_output)
    return max(0.0, float(bound)), runtime


def _set_partition_start(z: Any, labels: np.ndarray | None, data: Data) -> None:
    if labels is None:
        return
    labels = np.asarray(labels, dtype=int)
    relabeling: dict[int, int] = {}
    canonical = np.empty_like(labels)
    for i, label in enumerate(labels):
        if int(label) not in relabeling:
            relabeling[int(label)] = len(relabeling)
        canonical[i] = relabeling[int(label)]
    for i in range(data.n):
        for j in range(data.facilities):
            z[i, j].Start = float(canonical[i] == j)


def _extract_labels(z: Any, data: Data) -> np.ndarray:
    return np.array([max(range(data.facilities), key=lambda j: z[i, j].X) for i in range(data.n)])


def _remaining(started: float, time_limit: float) -> float:
    return max(0.0, float(time_limit) - (perf_counter() - started))


def _solve_labeled_projector(
    data: Data,
    time_limit: float = 300.0,
    tolerance: float = 1e-6,
    maximum_iterations: int | None = None,
    warm_start: WarmStartOptions | Results | None = None,
    use_sdp_bound: bool = True,
    sdp_bound_time_limit: float = 60.0,
    use_symmetry_breaking: bool = True,
    compute_relaxation: bool = True,
    relaxation_time_limit: float = 60.0,
    log_output: bool = False,
    mip_gap: float = 1e-4,
    formulation: str = "PW",
) -> Results:
    """Exact strengthened decomposition for the pairwise projector model."""

    started = perf_counter()
    initial = warm_start if isinstance(warm_start, Results) else build_warm_start(data, warm_start)
    gp, GRB, model, z = _build_partition_master(
        data, f"{formulation}_master", log_output, use_symmetry_breaking
    )
    model.Params.MIPGap = float(mip_gap)
    theta = model.addVar(lb=0.0, name="theta")
    sdp_budget = min(float(sdp_bound_time_limit), max(0.01, _remaining(started, time_limit)))
    sdp_bound, sdp_time = _obtain_sdp_bound(data, initial, use_sdp_bound, sdp_budget, log_output)
    if sdp_bound > 0.0:
        model.addConstr(theta >= sdp_bound, name="certified_sdp_bound")
    model.setObjective(theta, GRB.MINIMIZE)
    _set_partition_start(z, None if initial is None else initial.labels, data)
    relaxation_bound = -math.inf
    relaxation_time = 0.0
    if compute_relaxation and _remaining(started, time_limit) > 0:
        relaxation_bound, relaxation_time = _solve_gurobi_relaxation(
            model,
            min(float(relaxation_time_limit), _remaining(started, time_limit)),
            log_output,
        )
    best_objective = math.inf if initial is None else initial.objective
    best_labels = None if initial is None else initial.labels.copy()
    best_centers = None if initial is None or initial.centers is None else initial.centers.copy()
    lower_bound = sdp_bound
    history: list[dict[str, float]] = []
    seen: set[tuple[int, ...]] = set()
    # By default the wall-clock limit, rather than an arbitrary iteration
    # cap, controls termination.  Users may impose a cap for profiling.
    limit = maximum_iterations or 1_000_000
    subproblem_termination = None
    status = "TIME_LIMIT"
    for iteration in range(1, limit + 1):
        remaining = _remaining(started, time_limit)
        if remaining <= 0:
            break
        model.Params.TimeLimit = remaining
        model.optimize()
        if model.SolCount == 0:
            status = _gurobi_status(model.Status, GRB)
            break
        candidate_bound = float(model.ObjBound)
        if np.isfinite(candidate_bound):
            lower_bound = max(lower_bound, candidate_bound)
        labels = _extract_labels(z, data)
        key = tuple(int(value) for value in labels)
        if key in seen and best_objective - lower_bound <= tolerance:
            status = "OPTIMAL"
            break
        seen.add(key)
        try:
            centers, value, _cluster_costs = solve_fixed_partition(
                data, labels, _remaining(started, time_limit), log_output
            )
        except SubproblemTimeLimit as error:
            subproblem_termination = str(error)
            status = "TIME_LIMIT"
            break
        if value < best_objective:
            best_objective, best_labels, best_centers = value, labels.copy(), centers.copy()
        # Here z is only the computational representation of P_j e. Thus,
        # this is precisely the projector logic cut in the paper, rather
        # than a cut derived from the compact distance formulation.
        mismatch = gp.quicksum(1 - z[i, int(labels[i])] for i in range(data.n))
        model.addConstr(theta >= float(value) * (1 - mismatch), name=f"partition_cut_{iteration}")
        history.append({"iteration": float(iteration), "lower_bound": lower_bound, "upper_bound": best_objective})
        if best_objective - lower_bound <= tolerance:
            status = "OPTIMAL"
            break
    else:
        status = "ITERATION_LIMIT"
    result = Results(
        method=formulation,
        status=status,
        objective=best_objective,
        lower_bound=min(lower_bound, best_objective),
        runtime=perf_counter() - started,
        labels=best_labels,
        centers=best_centers,
        iterations=len(history),
        optimal=status == "OPTIMAL",
        history=history,
        metadata={
            "subproblem_termination": subproblem_termination,
            "formulation": formulation,
            "projector_encoding": "block_diagonal_rank_p" if formulation == "LR-A" else "rank_one_per_facility",
            "evaluated_partitions": len(seen),
            "tau": data.tau,
            "sdp_root_bound": sdp_bound,
            "sdp_bound_time": sdp_time,
            "projection_logic_cuts": len(history),
            "symmetry_breaking": use_symmetry_breaking,
            "continuous_relaxation_bound": relaxation_bound,
            "continuous_relaxation_time": relaxation_time,
            "continuous_relaxation_gap": _relative_bound_gap(best_objective, relaxation_bound),
            "sdp_root_gap": _relative_bound_gap(best_objective, sdp_bound),
        },
    )
    model.dispose()
    return result


def solve_pairwise(data: Data, *args, **kwargs) -> Results:
    """PW: exact labeled projector logic cuts and fixed-partition conic solves."""
    return _solve_labeled_projector(data, *args, **kwargs, formulation="PW")


def solve_lr_a(data: Data, *args, **kwargs) -> Results:
    """LR-A (MPCO-A): block-diagonal rank-p projector decomposition.

    Eliminating fixed block projectors reduces Qbar=Pbar Qbar to one
    center per block. The conic value and labeled-partition logic cut are
    identical to PW, so both intentionally share the same master algorithm.
    No dense pn-by-pn block-diagonal matrix needs to be materialized.
    """
    return _solve_labeled_projector(data, *args, **kwargs, formulation="LR-A")


def solve_lr_d(
    data: Data,
    time_limit: float = 300.0,
    tolerance: float = 1e-6,
    maximum_iterations: int | None = None,
    warm_start: WarmStartOptions | Results | None = None,
    use_sdp_bound: bool = True,
    sdp_bound_time_limit: float = 60.0,
    use_symmetry_breaking: bool = True,
    compute_relaxation: bool = True,
    relaxation_time_limit: float = 60.0,
    log_output: bool = False,
    mip_gap: float = 1e-4,
) -> Results:
    """Exact disaggregated projector decomposition with projector-value cuts.

    The master uses one epigraph variable per facility.  A cut generated for
    a subset is valid for every occurrence of that labeled subset and avoids
    resolving its MOSEK location problem.  Finite convergence follows because
    there are finitely many nonempty subsets of demand points.
    """

    started = perf_counter()
    initial = warm_start if isinstance(warm_start, Results) else build_warm_start(data, warm_start)
    gp, GRB, model, z = _build_partition_master(
        data, "LR-D_master", log_output, use_symmetry_breaking
    )
    model.Params.MIPGap = float(mip_gap)
    theta = model.addVars(data.facilities, lb=0.0, name="theta")
    total_theta = gp.quicksum(theta[j] for j in range(data.facilities))
    sdp_budget = min(float(sdp_bound_time_limit), max(0.01, _remaining(started, time_limit)))
    sdp_bound, sdp_time = _obtain_sdp_bound(data, initial, use_sdp_bound, sdp_budget, log_output)
    if sdp_bound > 0.0:
        model.addConstr(total_theta >= sdp_bound, name="certified_sdp_bound")
    model.setObjective(total_theta, GRB.MINIMIZE)
    _set_partition_start(z, None if initial is None else initial.labels, data)
    relaxation_bound = -math.inf
    relaxation_time = 0.0
    if compute_relaxation and _remaining(started, time_limit) > 0:
        relaxation_bound, relaxation_time = _solve_gurobi_relaxation(
            model,
            min(float(relaxation_time_limit), _remaining(started, time_limit)),
            log_output,
        )
    best_objective = math.inf if initial is None else initial.objective
    best_labels = None if initial is None else initial.labels.copy()
    best_centers = None if initial is None or initial.centers is None else initial.centers.copy()
    lower_bound = sdp_bound
    history: list[dict[str, float]] = []
    cache: dict[tuple[int, ...], tuple[np.ndarray, float]] = {}
    seen_partitions: set[tuple[int, ...]] = set()
    # By default the wall-clock limit, rather than an arbitrary iteration
    # cap, controls termination.  Users may impose a cap for profiling.
    limit = maximum_iterations or 1_000_000
    subproblem_termination = None
    status = "TIME_LIMIT"
    for iteration in range(1, limit + 1):
        remaining = _remaining(started, time_limit)
        if remaining <= 0:
            break
        model.Params.TimeLimit = remaining
        model.optimize()
        if model.SolCount == 0:
            status = _gurobi_status(model.Status, GRB)
            break
        candidate_bound = float(model.ObjBound)
        if np.isfinite(candidate_bound):
            lower_bound = max(lower_bound, candidate_bound)
        labels = _extract_labels(z, data)
        seen_partitions.add(tuple(int(value) for value in labels))
        centers = np.empty((data.facilities, data.dimension))
        total = 0.0
        new_cut = False
        try:
            for j in range(data.facilities):
                subset = tuple(int(i) for i in np.flatnonzero(labels == j))
                if subset not in cache:
                    center, cost = solve_single_facility(
                        data.points[list(subset)],
                        data.weights[list(subset)],
                        data.tau,
                        _remaining(started, time_limit),
                        log_output,
                    )
                    cache[subset] = (center, cost)
                center, cost = cache[subset]
                centers[j] = center
                total += cost
                complement = [i for i in range(data.n) if i not in subset]
                # z[:,j] represents Y_j e in the finite projector master. The
                # cut is therefore a fixed-projector value cut.
                mismatch = gp.quicksum(1 - z[i, j] for i in subset) + gp.quicksum(z[i, j] for i in complement)
                model.addConstr(theta[j] >= float(cost) * (1 - mismatch), name=f"cluster_cut_{iteration}_{j}")
                new_cut = True
        except SubproblemTimeLimit as error:
            subproblem_termination = str(error)
            status = "TIME_LIMIT"
            break
        if total < best_objective:
            best_objective, best_labels, best_centers = total, labels.copy(), centers.copy()
        history.append({"iteration": float(iteration), "lower_bound": lower_bound, "upper_bound": best_objective})
        if best_objective - lower_bound <= tolerance:
            status = "OPTIMAL"
            break
        if not new_cut:
            status = "STALLED"
            break
    else:
        status = "ITERATION_LIMIT"
    result = Results(
        method="LR-D",
        status=status,
        objective=best_objective,
        lower_bound=min(lower_bound, best_objective),
        runtime=perf_counter() - started,
        labels=best_labels,
        centers=best_centers,
        iterations=len(history),
        optimal=status == "OPTIMAL",
        history=history,
        metadata={
            "subproblem_termination": subproblem_termination,
            "evaluated_partitions": len(seen_partitions),
            "evaluated_clusters": len(cache),
            "tau": data.tau,
            "sdp_root_bound": sdp_bound,
            "sdp_bound_time": sdp_time,
            "projection_logic_cuts": len(cache),
            "symmetry_breaking": use_symmetry_breaking,
            "continuous_relaxation_bound": relaxation_bound,
            "continuous_relaxation_time": relaxation_time,
            "continuous_relaxation_gap": _relative_bound_gap(best_objective, relaxation_bound),
            "sdp_root_gap": _relative_bound_gap(best_objective, sdp_bound),
        },
    )
    model.dispose()
    return result


def solve_lr_sa(
    data: Data,
    time_limit: float = 300.0,
    tolerance: float = 1e-6,
    maximum_iterations: int | None = None,
    warm_start: WarmStartOptions | Results | None = None,
    use_sdp_bound: bool = True,
    sdp_bound_time_limit: float = 60.0,
    use_symmetry_breaking: bool = True,
    compute_relaxation: bool = True,
    relaxation_time_limit: float = 60.0,
    log_output: bool = False,
    mip_gap: float = 1e-4,
) -> Results:
    """Solve MPCO-SA by exact outer approximation over aggregate projectors.

    The Gurobi master contains one epigraph variable and selects an unlabeled
    partition through pairwise co-assignment variables.  For a fixed
    partition, MOSEK solves the conic location subproblem, which is the
    fixed-projector value function of MPCO-SA.  The resulting logic cut is
    exact at the evaluated aggregate projector and inactive at every other
    partition.  No big-M constraints are used.
    """

    started = perf_counter()
    initial = warm_start if isinstance(warm_start, Results) else build_warm_start(data, warm_start)
    gp, GRB, model, z, same = _build_aggregate_partition_master(
        data, "LR-SA_master", log_output, use_symmetry_breaking
    )
    model.Params.MIPGap = float(mip_gap)
    theta = model.addVar(lb=0.0, name="theta")
    sdp_budget = min(float(sdp_bound_time_limit), max(0.01, _remaining(started, time_limit)))
    sdp_bound, sdp_time = _obtain_sdp_bound(
        data, initial, use_sdp_bound, sdp_budget, log_output
    )
    if sdp_bound > 0.0:
        model.addConstr(theta >= sdp_bound, name="certified_sdp_bound")
    model.setObjective(theta, GRB.MINIMIZE)
    _set_partition_start(z, None if initial is None else initial.labels, data)

    relaxation_bound = -math.inf
    relaxation_time = 0.0
    if compute_relaxation and _remaining(started, time_limit) > 0:
        relaxation_bound, relaxation_time = _solve_gurobi_relaxation(
            model,
            min(float(relaxation_time_limit), _remaining(started, time_limit)),
            log_output,
        )

    best_objective = math.inf if initial is None else initial.objective
    best_labels = None if initial is None else initial.labels.copy()
    best_centers = None if initial is None or initial.centers is None else initial.centers.copy()
    lower_bound = sdp_bound
    history: list[dict[str, float]] = []
    seen: set[tuple[int, ...]] = set()
    limit = maximum_iterations or 1_000_000
    subproblem_termination = None
    status = "TIME_LIMIT"

    for iteration in range(1, limit + 1):
        remaining = _remaining(started, time_limit)
        if remaining <= 0:
            break
        model.Params.TimeLimit = remaining
        model.optimize()
        if model.SolCount == 0:
            status = _gurobi_status(model.Status, GRB)
            break

        candidate_bound = float(model.ObjBound)
        if np.isfinite(candidate_bound):
            lower_bound = max(lower_bound, candidate_bound)
        labels = _extract_labels(z, data)
        pair_values = {
            (i, h): int(labels[i] == labels[h])
            for i in range(data.n)
            for h in range(i + 1, data.n)
        }
        key = tuple(pair_values[i, h] for i in range(data.n) for h in range(i + 1, data.n))

        try:
            centers, value, _cluster_costs = solve_fixed_partition(
                data, labels, _remaining(started, time_limit), log_output
            )
        except SubproblemTimeLimit as error:
            subproblem_termination = str(error)
            status = "TIME_LIMIT"
            break
        if value < best_objective:
            best_objective, best_labels, best_centers = value, labels.copy(), centers.copy()

        # The Hamming distance between co-assignment vectors is zero exactly
        # for the evaluated unlabeled partition and at least one for every
        # other aggregate projector. This is the incidence implementation of
        # the normalized projector-separation cut in the manuscript.
        mismatch = gp.quicksum(
            (1 - same[i, h]) if pair_values[i, h] else same[i, h]
            for i in range(data.n)
            for h in range(i + 1, data.n)
        )
        model.addConstr(
            theta >= float(value) * (1 - mismatch),
            name=f"aggregate_partition_cut_{iteration}",
        )
        seen.add(key)
        history.append(
            {
                "iteration": float(iteration),
                "lower_bound": lower_bound,
                "upper_bound": best_objective,
            }
        )
        if best_objective - lower_bound <= tolerance:
            status = "OPTIMAL"
            break
    else:
        status = "ITERATION_LIMIT"

    result = Results(
        method="LR-SA",
        status=status,
        objective=best_objective,
        lower_bound=min(lower_bound, best_objective),
        runtime=perf_counter() - started,
        labels=best_labels,
        centers=best_centers,
        iterations=len(history),
        optimal=status == "OPTIMAL",
        history=history,
        metadata={
            "subproblem_termination": subproblem_termination,
            "evaluated_partitions": len(seen),
            "evaluated_clusters": 0,
            "tau": data.tau,
            "sdp_root_bound": sdp_bound,
            "sdp_bound_time": sdp_time,
            "projection_logic_cuts": len(history),
            "symmetry_breaking": use_symmetry_breaking,
            "aggregate_projector_encoding": "pairwise_coassignment",
            "coassignment_variables": data.n * (data.n - 1) // 2,
            "continuous_relaxation_bound": relaxation_bound,
            "continuous_relaxation_time": relaxation_time,
            "continuous_relaxation_gap": _relative_bound_gap(best_objective, relaxation_bound),
            "sdp_root_gap": _relative_bound_gap(best_objective, sdp_bound),
        },
    )
    model.dispose()
    return result


# Backwards-compatible names for existing notebooks and campaigns.
solve_mpco = solve_lr_d
solve_mpco_sa = solve_lr_sa


def solve_all(
    data: Data,
    methods: Iterable[MethodName] = ("compact", "PW", "LR-D", "LR-A", "LR-SA"),
    time_limit: float = 300.0,
    warm_start: WarmStartOptions | None = None,
    keep_best: bool = True,
    log_output: bool = False,
) -> tuple[dict[str, Results], Results | None]:
    """Run selected methods with a shared warm start and return the best."""

    initial = build_warm_start(data, warm_start or WarmStartOptions())
    results: dict[str, Results] = {}
    for method in methods:
        if method == "compact":
            results[method] = solve_compact(data, time_limit, warm_start=initial, log_output=log_output)
        elif method in {"pairwise", "PW"}:
            results[method] = solve_pairwise(data, time_limit, warm_start=initial, log_output=log_output)
        elif method in {"mpco", "LR-D"}:
            results[method] = solve_mpco(data, time_limit, warm_start=initial, log_output=log_output)
        elif method in {"mpco_sa", "LR-SA"}:
            results[method] = solve_mpco_sa(data, time_limit, warm_start=initial, log_output=log_output)
        elif method == "LR-A":
            results[method] = solve_lr_a(data, time_limit=time_limit, warm_start=initial, log_output=log_output)
        else:
            raise ValueError(f"unknown method: {method}")
    feasible = [result for result in results.values() if np.isfinite(result.objective)]
    best = min(feasible, key=lambda result: result.objective) if keep_best and feasible else None
    return results, best


def plot_solution(data: Data, result: Results, ax: Any = None, title: str | None = None) -> Any:
    """Plot a two-dimensional instance and a feasible solution."""

    if data.dimension != 2:
        raise ValueError("plot_solution is available only for two-dimensional instances")
    if result.labels is None or result.centers is None:
        raise ValueError("the result does not contain a feasible solution")
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    colors = plt.get_cmap("tab10")
    for j in range(data.facilities):
        mask = result.labels == j
        ax.scatter(data.points[mask, 0], data.points[mask, 1], s=38, color=colors(j), alpha=0.8, label=f"Facility {j + 1}")
        ax.scatter(result.centers[j, 0], result.centers[j, 1], s=180, marker="*", color=colors(j), edgecolor="black", linewidth=1.0)
        for point in data.points[mask]:
            ax.plot([point[0], result.centers[j, 0]], [point[1], result.centers[j, 1]], color=colors(j), alpha=0.16, linewidth=0.8)
    ax.set_title(title or f"{result.method}: objective={result.objective:.4f}")
    ax.set_xlabel("Coordinate 1")
    ax.set_ylabel("Coordinate 2")
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def plot_comparison(data: Data, results: dict[str, Results], figsize: tuple[float, float] | None = None) -> Any:
    """Plot all feasible solutions side by side."""

    import matplotlib.pyplot as plt

    feasible = [(name, result) for name, result in results.items() if result.labels is not None]
    if not feasible:
        raise ValueError("no feasible results to plot")
    figure, axes = plt.subplots(1, len(feasible), figsize=figsize or (5 * len(feasible), 4.5), squeeze=False)
    for ax, (name, result) in zip(axes[0], feasible):
        plot_solution(data, result, ax=ax, title=f"{name}\nobj={result.objective:.3f}, time={result.runtime:.2f}s")
    figure.tight_layout()
    return figure


def plot_history(result: Results, ax: Any = None) -> Any:
    """Plot lower and upper bounds recorded by a decomposition method."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    if not result.history:
        ax.text(0.5, 0.5, "No iteration history", ha="center", va="center")
        return ax
    iterations = [row["iteration"] for row in result.history]
    if "lower_bound" in result.history[0]:
        ax.plot(iterations, [row["lower_bound"] for row in result.history], marker="o", label="Lower bound")
    ax.plot(iterations, [row["upper_bound"] for row in result.history], marker="s", label="Upper bound")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective bound")
    ax.set_title(result.method)
    ax.legend()
    return ax


def save_results_csv(results: dict[str, Results], path: str | Path) -> None:
    """Save a comparison table without requiring pandas."""

    import csv

    records = [result.summary() for result in results.values()]
    if not records:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
