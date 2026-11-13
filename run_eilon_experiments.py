#!/usr/bin/env python3
"""Run reproducible Eilon experiments for the projector formulations.

The script is intentionally separate from the demonstration notebooks.  It
checkpoints every solver run, so an interrupted campaign can be resumed, and
stores both a flat table for the paper and detailed JSON records for studying
how the SDP bound, symmetry breaking, warm starts, and decomposition cuts
affect the algorithms.

Download the instances first with::

    python fetch_instances.py --families eilon

An initial small campaign can then be run with::

    python run_eilon_experiments.py --profile quick

Use ``--profile full`` for the SDP and symmetry-breaking ablations.  Existing
runs are skipped unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from sdp_clusters import (
    Data,
    Results,
    WarmStartOptions,
    build_warm_start,
    solve_compact,
    solve_mpco_sa,
    solve_lr_d,
    solve_lr_a,
    solve_lr_sa,
    solve_pairwise,
)


SUPPORTED_TAUS = (1.0, 1.5, 2.0, 3.0, math.inf)


@dataclass(frozen=True)
class Configuration:
    """One algorithmic configuration in the ablation study."""

    name: str
    method: str
    use_sdp_bound: bool = True
    use_symmetry_breaking: bool = True
    paper_strengthening: bool = False


def tau_label(tau: float) -> str:
    return "inf" if math.isinf(tau) else f"{tau:g}"


def finite_or_none(value: Any) -> Any:
    """Convert NumPy values and nonfinite floats to JSON-safe values."""

    if isinstance(value, np.ndarray):
        return [finite_or_none(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def relative_gap(upper: float, lower: float) -> float:
    if not np.isfinite(upper) or not np.isfinite(lower):
        return math.inf
    return max(0.0, (upper - lower) / max(1.0, abs(upper)))


def discover_eilon(instance_root: Path, selected_sizes: set[int] | None) -> list[Path]:
    files = sorted((instance_root / "processed" / "eilon").glob("eilon*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No Eilon files were found under {instance_root}. "
            "Run 'python fetch_instances.py --families eilon' first."
        )
    if selected_sizes is None:
        return files
    retained = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as stream:
            n = sum(1 for _ in csv.DictReader(stream))
        if n in selected_sizes:
            retained.append(path)
    return retained


def instance_size(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def configurations(profile: str, include_compact: bool) -> list[Configuration]:
    if profile == "paper":
        return ([Configuration("compact", "compact", paper_strengthening=True)] if include_compact else []) + [
            Configuration("PW", "PW"), Configuration("LR-D", "LR-D"),
            Configuration("LR-A", "LR-A"), Configuration("LR-SA", "LR-SA")]
    result: list[Configuration] = []
    if include_compact:
        result.append(Configuration("compact", "compact"))
    result.extend(
        [
            Configuration("pairwise_sdp_sym", "pairwise"),
            Configuration("mpco_sa_sdp_sym", "mpco_sa"),
        ]
    )
    if profile == "full":
        for method in ("pairwise", "mpco_sa"):
            result.extend(
                [
                    Configuration(f"{method}_no_sdp", method, False, True),
                    Configuration(f"{method}_no_sym", method, True, False),
                    Configuration(f"{method}_plain", method, False, False),
                ]
            )
    return result


def warm_start_record(warm: Results | None, elapsed: float) -> dict[str, Any]:
    if warm is None:
        return {
            "warm_method": "none",
            "warm_objective": math.inf,
            "warm_lower_bound": -math.inf,
            "warm_gap": math.inf,
            "warm_time": elapsed,
            "warm_candidates": [],
        }
    return {
        "warm_method": warm.method,
        "warm_objective": warm.objective,
        "warm_lower_bound": warm.lower_bound,
        "warm_gap": warm.gap,
        "warm_time": elapsed,
        "warm_candidates": warm.metadata.get("warm_start_candidates", []),
    }


def solve_configuration(
    data: Data,
    config: Configuration,
    warm: Results | None,
    time_limit: float,
    relaxation_time_limit: float,
    sdp_time_limit: float,
    mip_gap: float,
    log_output: bool,
    projector_engine: str = "legacy",
    sdp_node_time_limit: float = 60.0,
    sdp_root_time_limit: float = 300.0,
) -> Results:
    common = dict(
        time_limit=time_limit,
        warm_start=warm,
        compute_relaxation=True,
        relaxation_time_limit=relaxation_time_limit,
        log_output=log_output,
    )
    decomposition = dict(
        use_sdp_bound=config.use_sdp_bound,
        sdp_bound_time_limit=sdp_time_limit,
        use_symmetry_breaking=config.use_symmetry_breaking,
    )
    if config.method == "compact":
        if config.name == "compact" and getattr(config, "paper_strengthening", False):
            return solve_compact(data, mip_gap=mip_gap, **decomposition, **common)
        return solve_compact(data, mip_gap=mip_gap, **common)
    if projector_engine == "master" and config.method in {"PW", "LR-D", "LR-A", "LR-SA"}:
        from projector_master import solve_projector_master
        return solve_projector_master(data, config.method, time_limit=time_limit,
                                      warm_start=warm, mip_gap=mip_gap,
                                      sdp_time_limit=sdp_root_time_limit,
                                      relaxation_time_limit=relaxation_time_limit,
                                      log_output=log_output)
    if projector_engine == "sdp" and config.method in {"PW", "LR-D", "LR-A", "LR-SA"}:
        from sdp_projector_search import solve_projector_sdp
        return solve_projector_sdp(data, config.method, time_limit=time_limit,
                                   warm_start=warm, mip_gap=mip_gap,
                                   node_time_limit=sdp_node_time_limit,
                                   root_time_limit=sdp_root_time_limit,
                                   log_output=log_output)
    paper_solvers = {"PW": solve_pairwise, "LR-D": solve_lr_d,
                     "LR-A": solve_lr_a, "LR-SA": solve_lr_sa}
    if config.method in paper_solvers:
        return paper_solvers[config.method](data, mip_gap=mip_gap, **decomposition, **common)
    if config.method == "pairwise":
        return solve_pairwise(data, **decomposition, **common)
    if config.method == "mpco_sa":
        return solve_mpco_sa(data, **decomposition, **common)
    raise ValueError(f"Unknown method {config.method}")


def result_record(
    data: Data,
    source: Path,
    config: Configuration,
    result: Results,
    warm_info: dict[str, Any],
) -> dict[str, Any]:
    metadata = result.metadata
    history = result.history
    first_lb = history[0].get("lower_bound", math.nan) if history else math.nan
    last_lb = history[-1].get("lower_bound", math.nan) if history else math.nan
    first_ub = history[0].get("upper_bound", math.nan) if history else math.nan
    last_ub = history[-1].get("upper_bound", math.nan) if history else math.nan
    sdp_bound = metadata.get("sdp_root_bound", math.nan)
    relaxation_bound = metadata.get("continuous_relaxation_bound", math.nan)
    pairwise_distances = np.linalg.norm(
        data.points[:, None, :] - data.points[None, :, :], ord=data.tau, axis=2
    )
    positive_distances = pairwise_distances[pairwise_distances > 0]
    return {
        "run_id": f"{source.stem}__p{data.facilities}__tau{tau_label(data.tau)}__{config.name}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "instance": source.stem,
        "instance_file": str(source),
        "n": data.n,
        "dimension": data.dimension,
        "weight_sum": float(data.weights.sum()),
        "weight_min": float(data.weights.min()),
        "weight_max": float(data.weights.max()),
        "distance_diameter": float(pairwise_distances.max()),
        "positive_distance_min": float(positive_distances.min()) if positive_distances.size else 0.0,
        "positive_distance_mean": float(positive_distances.mean()) if positive_distances.size else 0.0,
        "p": data.facilities,
        "tau": tau_label(data.tau),
        "configuration": config.name,
        "method": result.method,
        "use_sdp_bound": config.use_sdp_bound if config.method != "compact" or config.paper_strengthening else False,
        "use_symmetry_breaking": config.use_symmetry_breaking if config.method != "compact" or config.paper_strengthening else False,
        "status": result.status,
        "optimal": result.optimal,
        "objective": result.objective,
        "lower_bound": result.lower_bound,
        "gap": result.gap,
        "runtime": result.runtime,
        "iterations": result.iterations,
        "first_iteration_lb": first_lb,
        "last_iteration_lb": last_lb,
        "first_iteration_ub": first_ub,
        "last_iteration_ub": last_ub,
        "sdp_root_bound": sdp_bound,
        "sdp_root_gap": relative_gap(result.objective, sdp_bound),
        "sdp_bound_time": metadata.get("sdp_bound_time", math.nan),
        "continuous_relaxation_bound": relaxation_bound,
        "continuous_relaxation_gap": relative_gap(result.objective, relaxation_bound),
        "continuous_relaxation_time": metadata.get("continuous_relaxation_time", math.nan),
        "sdp_improvement_over_master": (
            sdp_bound - relaxation_bound
            if np.isfinite(sdp_bound) and np.isfinite(relaxation_bound)
            else math.nan
        ),
        "projection_logic_cuts": metadata.get("projection_logic_cuts", 0),
        "evaluated_partitions": metadata.get("evaluated_partitions", 0),
        "evaluated_clusters": metadata.get("evaluated_clusters", 0),
        "node_count": metadata.get("node_count", math.nan),
        "warm_method": warm_info["warm_method"],
        "warm_objective": warm_info["warm_objective"],
        "warm_lower_bound": warm_info["warm_lower_bound"],
        "warm_gap": warm_info["warm_gap"],
        "warm_time": warm_info["warm_time"],
    }


def detailed_record(
    flat: dict[str, Any],
    result: Results,
    warm_info: dict[str, Any],
) -> dict[str, Any]:
    return finite_or_none(
        {
            "summary": flat,
            "history": result.history,
            "metadata": result.metadata,
            "warm_start_candidates": warm_info["warm_candidates"],
            "labels": result.labels,
            "centers": result.centers,
        }
    )


def read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["run_id"] for row in csv.DictReader(stream) if row.get("run_id")}


def append_csv(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(record))
        if not exists:
            writer.writeheader()
        writer.writerow(finite_or_none(record))


def write_detail(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_error(path: Path, run_id: str, error: BaseException) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("run_id", "error", "traceback"))
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(
            {
                "run_id": run_id,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )


def create_figures(results_csv: Path, figure_dir: Path) -> None:
    """Create diagnostic figures when pandas and Matplotlib are available."""

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("Skipping figures because pandas or Matplotlib is unavailable.")
        return
    table = pd.read_csv(results_csv)
    if table.empty:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    table["instance_key"] = table["instance"].astype(str) + ", p=" + table["p"].astype(str) + ", tau=" + table["tau"].astype(str)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for name, group in table.groupby("configuration"):
        by_n = group.groupby("n", as_index=True).median(numeric_only=True)
        axes[0].plot(by_n.index, by_n["runtime"], marker="o", label=name)
        axes[1].plot(by_n.index, by_n["gap"], marker="o", label=name)
        axes[2].plot(by_n.index, by_n["iterations"], marker="o", label=name)
    axes[0].set(xlabel="Number of demand points", ylabel="Median time (s)", title="Runtime")
    axes[1].set(xlabel="Number of demand points", ylabel="Median relative gap", title="Final proof gap")
    axes[2].set(xlabel="Number of demand points", ylabel="Median iterations", title="Decomposition effort")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "performance_by_size.pdf", bbox_inches="tight")
    plt.close(fig)

    projector = table[table["configuration"].isin(["pairwise_sdp_sym", "mpco_sa_sdp_sym"])]
    if not projector.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for name, group in projector.groupby("configuration"):
            axes[0].scatter(group["continuous_relaxation_gap"], group["sdp_root_gap"], alpha=0.7, label=name)
            axes[1].scatter(group["sdp_root_gap"], group["iterations"], alpha=0.7, label=name)
        axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        axes[0].set(xlabel="Initial master-relaxation gap", ylabel="SDP root gap", title="Strength of the SDP bound")
        axes[1].set(xlabel="SDP root gap", ylabel="Decomposition iterations", title="Bound quality and effort")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / "sdp_diagnostics.pdf", bbox_inches="tight")
        plt.close(fig)


def parse_tau(values: Iterable[str]) -> list[float]:
    parsed = []
    for value in values:
        parsed.append(math.inf if value.lower() in {"inf", "infinity"} else float(value))
    invalid = [value for value in parsed if not any((math.isinf(value) and math.isinf(tau)) or math.isclose(value, tau) for tau in SUPPORTED_TAUS)]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported norm orders: {invalid}")
    return parsed


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, default=root / "instances")
    parser.add_argument("--output", type=Path, default=root / "results" / "eilon")
    parser.add_argument("--resume-from", type=Path, action="append", default=[],
                        help="Additional results CSV whose recorded run IDs should be skipped (repeatable).")
    parser.add_argument("--profile", choices=("quick", "full", "paper"), default="quick")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=(20, 30, 40, 50),
        help="Instance sizes to retain (default: 20 30 40 50).",
    )
    parser.add_argument("--facilities", nargs="+", type=int, default=(2, 3, 5))
    parser.add_argument("--taus", nargs="+", default=("1", "1.5", "2", "3", "inf"))
    parser.add_argument("--time-limit", type=float, default=600.0, help="Seconds per exact solve.")
    parser.add_argument("--relaxation-time-limit", type=float, default=60.0)
    parser.add_argument("--sdp-time-limit", type=float, default=120.0)
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument("--warm-random-starts", type=int, default=10)
    parser.add_argument("--warm-iterations", type=int, default=50)
    parser.add_argument("--warm-sdp-time-limit", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-compact", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-output", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--projector-engine", choices=("legacy", "sdp", "master"), default="legacy")
    parser.add_argument("--sdp-node-time-limit", type=float, default=60.0)
    parser.add_argument("--sdp-root-time-limit", type=float, default=300.0)
    arguments = parser.parse_args()
    if arguments.projector_engine in {"sdp", "master"} and (arguments.profile != "paper" or any(tau not in (1.5, 2.0, 3.0) for tau in parse_tau(arguments.taus))):
        parser.error("The SDP search requires --profile paper and taus in {1.5, 2, 3}")
    if arguments.sdp_node_time_limit <= 0 or arguments.sdp_root_time_limit <= 0:
        parser.error("SDP node/root time limits must be positive")

    if arguments.overwrite and arguments.output.exists():
        # Do not delete user files. Start a timestamped campaign instead.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        arguments.output = arguments.output.with_name(f"{arguments.output.name}_{stamp}")

    taus = parse_tau(arguments.taus)
    sources = discover_eilon(arguments.instances, None if arguments.sizes is None else set(arguments.sizes))
    configs = configurations(arguments.profile, not arguments.skip_compact)
    results_csv = arguments.output / "runs_large.csv"
    completed = read_completed(results_csv)
    for checkpoint in arguments.resume_from:
        if not checkpoint.is_file():
            parser.error(f"Resume CSV does not exist: {checkpoint}")
        completed.update(read_completed(checkpoint))
    details_dir = arguments.output / "details"
    errors_csv = arguments.output / "errors.csv"

    settings = vars(arguments).copy()
    settings.update(
        instances=str(arguments.instances),
        output=str(arguments.output),
        resume_from=[str(path) for path in arguments.resume_from],
        taus=[tau_label(tau) for tau in taus],
        configurations=[asdict(config) for config in configs],
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    sizes = {source: instance_size(source) for source in sources}
    total = sum(
        1
        for source in sources
        for p in arguments.facilities
        for _tau in taus
        for _config in configs
        if p < sizes[source]
    )
    count = 0
    for source in sources:
        n = sizes[source]
        for p in arguments.facilities:
            if p >= n:
                continue
            for tau in taus:
                if all(
                    f"{source.stem}__p{p}__tau{tau_label(tau)}__{config.name}" in completed
                    for config in configs
                ):
                    count += len(configs)
                    print(f"[{count}/{total}] skip completed {source.stem}, p={p}, tau={tau_label(tau)}")
                    continue
                data = Data.from_csv(source, facilities=p, tau=tau)
                warm_started = perf_counter()
                warm = build_warm_start(
                    data,
                    WarmStartOptions(
                        strategy="best",
                        random_starts=arguments.warm_random_starts,
                        alternating_iterations=arguments.warm_iterations,
                        sdp_time_limit=arguments.warm_sdp_time_limit,
                        seed=arguments.seed,
                        log_output=arguments.log_output,
                    ),
                )
                warm_info = warm_start_record(warm, perf_counter() - warm_started)
                for config in configs:
                    count += 1
                    run_id = f"{source.stem}__p{p}__tau{tau_label(tau)}__{config.name}"
                    if run_id in completed:
                        print(f"[{count}/{total}] skip {run_id}")
                        continue
                    print(f"[{count}/{total}] run  {run_id}", flush=True)
                    try:
                        result = solve_configuration(
                            data,
                            config,
                            warm,
                            arguments.time_limit,
                            arguments.relaxation_time_limit,
                            arguments.sdp_time_limit,
                            arguments.mip_gap,
                            arguments.log_output,
                            arguments.projector_engine,
                            arguments.sdp_node_time_limit,
                            arguments.sdp_root_time_limit,
                        )
                        flat = result_record(data, source, config, result, warm_info)
                        if arguments.projector_engine in {"sdp", "master"}:
                            flat.update(engine=result.metadata.get("engine", "gurobi_compact"),
                                        sdp_relaxation=result.metadata.get("relaxation", "legacy_shared_sdp"),
                                        sdp_nodes=result.metadata.get("sdp_nodes", 0),
                                        open_nodes=result.metadata.get("open_nodes", 0),
                                        partition_triangle_cuts=result.metadata.get("partition_triangle_cuts", 0),
                                        unknown_sdp_nodes=result.metadata.get("unknown_sdp_nodes", 0),
                                        terminal_partitions=result.metadata.get("terminal_partitions", 0))
                        append_csv(results_csv, flat)
                        write_detail(details_dir / f"{run_id}.json", detailed_record(flat, result, warm_info))
                        completed.add(run_id)
                        print(
                            f"    {result.status}: obj={result.objective:.6g}, "
                            f"lb={result.lower_bound:.6g}, gap={result.gap:.3g}, "
                            f"time={result.runtime:.2f}s"
                        )
                    except Exception as error:  # preserve the rest of a long campaign
                        write_error(errors_csv, run_id, error)
                        print(f"    ERROR: {error}", file=sys.stderr)

    if results_csv.exists() and not arguments.no_figures:
        create_figures(results_csv, arguments.output / "figures")
    print(f"Results written to {arguments.output}")


if __name__ == "__main__":
    main()
