"""Small tests for data handling and licensed solver integrations."""

import math
import csv

import numpy as np

from sdp_clusters import (
    Data,
    WarmStartOptions,
    alternating_warm_start,
    assignment_projectors,
    objective_value,
    sdp_warm_start,
    solve_compact,
    solve_mpco,
    solve_pairwise,
    solve_single_facility,
)


def small_instance() -> Data:
    return Data(
        points=np.array([[0.0, 0.0], [0.2, 0.0], [4.8, 5.0], [5.0, 5.0]]),
        facilities=2,
        name="four_points",
    )


def test_data_and_projectors() -> None:
    data = small_instance()
    labels = np.array([0, 0, 1, 1])
    projectors = assignment_projectors(labels, data.facilities)
    assert all(np.allclose(P @ P, P) for P in projectors)
    assert np.allclose(sum(P @ np.ones(data.n) for P in projectors), np.ones(data.n))


def test_all_supported_norm_epigraphs() -> None:
    points = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
    weights = np.ones(3)
    for tau in (1.0, 1.5, 2.0, 3.0, math.inf):
        center, value = solve_single_facility(points, weights, tau=tau, time_limit=20)
        direct = sum(np.linalg.norm(point - center, ord=tau) for point in points)
        assert math.isclose(value, direct, rel_tol=2e-5, abs_tol=2e-5)


def test_norm_validation() -> None:
    assert Data(np.zeros((2, 2)), 1, tau="infinity").tau == math.inf
    try:
        Data(np.zeros((2, 2)), 1, tau=4)
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported norm order was accepted")


def test_load_standardized_instance(tmp_path) -> None:
    path = tmp_path / "sample.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "x", "y", "weight"))
        writer.writerow((1, 0.0, 1.0, 2.0))
        writer.writerow((2, 3.0, 4.0, 5.0))
    data = Data.from_csv(path, facilities=1, tau=3)
    assert data.n == 2
    assert data.tau == 3.0
    assert np.allclose(data.weights, [2.0, 5.0])


def test_alternating_solution_is_feasible() -> None:
    data = small_instance()
    result = alternating_warm_start(data, [0, 0, 1, 1], maximum_iterations=5)
    assert np.isfinite(result.objective)
    assert set(result.labels) == {0, 1}
    assert math.isclose(result.objective, objective_value(data, result.labels, result.centers), rel_tol=1e-5)


def test_sdp_warm_start() -> None:
    result = sdp_warm_start(small_instance(), time_limit=20)
    assert np.isfinite(result.objective)
    assert np.isfinite(result.lower_bound)
    assert result.lower_bound <= result.objective + 1e-5


def test_exact_methods_agree_on_small_instance() -> None:
    data = small_instance()
    warm = WarmStartOptions(strategy="multistart", random_starts=2, seed=1)
    compact = solve_compact(data, time_limit=30, warm_start=warm)
    pairwise = solve_pairwise(data, time_limit=30, warm_start=warm)
    mpco = solve_mpco(data, time_limit=30, warm_start=warm)
    values = np.array([compact.objective, pairwise.objective, mpco.objective])
    assert np.all(np.isfinite(values))
    assert values.max() - values.min() <= 1e-4
    for result in (compact, pairwise, mpco):
        relaxation = result.metadata["continuous_relaxation_bound"]
        assert np.isfinite(relaxation)
        assert relaxation <= result.objective + 1e-5
        assert result.metadata["continuous_relaxation_gap"] >= -1e-7
    for result in (pairwise, mpco):
        assert result.metadata["symmetry_breaking"] is True
        assert result.metadata["projection_logic_cuts"] > 0

