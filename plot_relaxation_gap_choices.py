#!/usr/bin/env python3
"""Create SIAM-ready alternatives for the continuous-relaxation comparison.

The script keeps only (instance, p) runs available for every model within a
given (n, tau) group.  This is the same paired comparison used in the paper.

Examples
--------
Generate every alternative::

    python3 plot_relaxation_gap_choices.py all_v1.xlsx --style all

Generate only the compact dot plot with a fully specified output name::

    python3 plot_relaxation_gap_choices.py all_v1.xlsx \
        --style dot --output figures/relaxation_gap_deviation.pdf
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
# Keep Jupyter's inline backend. Terminal executions use a non-interactive
# backend so figure generation does not require a display server.
if "ipykernel" not in sys.modules:
    mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


MODELS = ("compact", "PW", "LR-D", "LR-A", "LR-SA")
TIME_LIMIT = 3600.0
PROJECTOR_MODELS = MODELS[1:]
DISPLAY = {"compact": "MICO", "PW": "PW", "LR-D": "LR-D",
           "LR-A": "LR-A", "LR-SA": "LR-SA"}
COLORS = {"PW": "#0072B2", "LR-D": "#D55E00",
          "LR-A": "#009E73", "LR-SA": "#CC79A7"}
MARKERS = {"PW": "o", "LR-D": "s", "LR-A": "^", "LR-SA": "D"}


def configure_fonts(tex: str) -> None:
    """Use LaTeX when requested and available, with a CM fallback."""
    use_tex = tex == "on" or (tex == "auto" and shutil.which("latex") is not None)
    mpl.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def paired_summary(workbook: Path, sheet: str) -> pd.DataFrame:
    """Return mean gaps on common runs and deviations from MICO."""
    data = pd.read_excel(workbook, sheet_name=sheet)
    required = {"n", "tau", "configuration", "instance", "p",
                "continuous_relaxation_gap"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing workbook columns: {sorted(missing)}")

    records: list[dict[str, float | int | str]] = []
    for (n, tau), group in data.groupby(["n", "tau"], sort=True):
        available = {
            model: set(zip(part["instance"], part["p"]))
            for model in MODELS
            for part in [group[group["configuration"].eq(model)]]
        }
        common = set.intersection(*(available[model] for model in MODELS))
        if not common:
            continue
        paired = group[
            group.apply(lambda row: (row["instance"], row["p"]) in common, axis=1)
        ]
        means = paired.groupby("configuration")["continuous_relaxation_gap"].mean()
        baseline = float(means["compact"])
        for model in MODELS:
            gap = float(means[model])
            records.append({
                "n": int(n), "tau": float(tau), "model": model,
                "runs": len(common), "gap_percent": 100.0 * gap,
                "relative_deviation": -100.0 * (gap - baseline) / baseline,
                "gap_index": 100.0 * gap / baseline,
            })
    return pd.DataFrame.from_records(records)


def paired_main_results(workbook: Path, sheet: str) -> pd.DataFrame:
    """Average runtime, final gap, and unsolved rate on matched runs."""
    data = pd.read_excel(workbook, sheet_name=sheet)
    required = {"n", "tau", "configuration", "instance", "p", "runtime",
                "gap", "Unsolved"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing workbook columns: {sorted(missing)}")
    # Solver overhead may make a recorded run slightly larger than the nominal
    # limit. Cap each individual observation before computing group averages.
    data["runtime"] = pd.to_numeric(data["runtime"], errors="coerce").clip(
        upper=TIME_LIMIT
    )
    records: list[dict[str, float | int | str]] = []
    for (n, tau), group in data.groupby(["n", "tau"], sort=True):
        available = {
            model: set(zip(part["instance"], part["p"]))
            for model in MODELS
            for part in [group[group["configuration"].eq(model)]]
        }
        common = set.intersection(*(available[model] for model in MODELS))
        if not common:
            continue
        paired = group[
            group.apply(lambda row: (row["instance"], row["p"]) in common, axis=1)
        ]
        for model in MODELS:
            part = paired[paired["configuration"].eq(model)]
            records.append({
                "n": int(n), "tau": float(tau), "model": model,
                "runs": len(part), "runtime": float(part["runtime"].mean()),
                "gap_percent": 100.0 * float(part["gap"].mean()),
                "unsolved_percent": 100.0 * float(part["Unsolved"].mean()),
            })
    return pd.DataFrame.from_records(records)


def latex_main_results_table(results: pd.DataFrame, destination: Path | None = None) -> str:
    """Create the three-block table used as ``tab:eilon-main-results``.

    Values are printed with one decimal. Boldface is decided from the
    unrounded averages, and exact ties are retained.
    """
    metrics = ("runtime", "gap_percent", "unsolved_percent")
    block_names = ("Time (s)", r"Opt Gap (\%)", r"Unsolved (\%)")
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\tiny",
        r"\setlength{\tabcolsep}{2.2pt}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{cc||rrrrr||rrrrr||rrrrr}", r"\hline\hline",
        "& & " + " & ".join(
            rf"\multicolumn{{5}}{{c{'||' if k < 2 else ''}}}{{{name}}}"
            for k, name in enumerate(block_names)
        ) + r" \\",
        r"$n$ & $\tau$ & " + " & ".join([
            header for _ in metrics for header in (
                r"\ref{eq:weber_original}", r"\ref{mod:pairwise}",
                r"\ref{mod:mpco-disaggregated}",
                r"\ref{mod:mpco-aggregated}",
                r"\ref{mod:mpco-sum-aggregated}",
            )
        ]) + r" \\",
        r"\hline\hline",
    ]
    previous_n: int | None = None
    for (n, tau), group in results.groupby(["n", "tau"], sort=True):
        indexed = group.set_index("model").reindex(MODELS)
        values: list[str] = []
        for metric in metrics:
            best = float(indexed[metric].min())
            for model in MODELS:
                value = float(indexed.loc[model, metric])
                shown = ("3600.0" if metric == "runtime" and
                         value >= TIME_LIMIT - 1e-10 else f"{value:.1f}")
                if np.isclose(value, best, rtol=1e-10, atol=1e-10) and ((metric == "runtime" and value < 3599.0) or (metric == "unsolved_percent" and value < 99.9) or metric == "gap_percent"):
                    shown = rf"\textbf{{{shown}}}"
                values.append(shown)
        tau_text = r"$3/2$" if tau == 1.5 else rf"${int(tau)}$"
        n_text = str(int(n)) if int(n) != previous_n else ""
        lines.append(f"{n_text} & {tau_text} & " + " & ".join(values) + r" \\")
        previous_n = int(n)
    lines.extend([
        r"\hline\hline", r"\end{tabular}", r"}",
        r"\caption{Average runtime, final MIP gap, and percentage of unsolved instances. Boldfaced values indicate the best performance within each $(n, \tau)$ group (except those reaching timelimit for all instances or 100\% final GAP).}",
        r"\label{tab:eilon-main-results}", r"\end{table}", "",
    ])
    latex = "\n".join(lines)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(latex, encoding="utf-8")
    return latex


def matched_run_data(workbook: Path, sheet: str) -> pd.DataFrame:
    """Return individual cases observed for all five formulations."""
    data = pd.read_excel(workbook, sheet_name=sheet)
    required = {"instance", "p", "tau", "n", "configuration", "runtime"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing workbook columns: {sorted(missing)}")
    data = data[data["configuration"].isin(MODELS)].copy()
    data["runtime"] = pd.to_numeric(data["runtime"], errors="coerce").clip(
        upper=TIME_LIMIT
    )
    key = ["instance", "p", "tau"]
    coverage = data.groupby(key)["configuration"].nunique()
    complete = coverage[coverage.eq(len(MODELS))].index
    matched = data.set_index(key).loc[lambda x: x.index.isin(complete)].reset_index()
    if "optimal" in matched:
        matched["certified"] = matched["optimal"].fillna(False).astype(bool)
    else:
        matched["certified"] = matched["status"].astype(str).str.upper().eq("OPTIMAL")
    return matched


def plot_performance_profile(
    matched: pd.DataFrame,
    sizes: tuple[int, ...] = (10, 20, 30, 40, 50),
    log_scale: bool = True,
) -> plt.Figure:
    """Dolan--More style runtime profile based on certified solutions."""
    key = ["instance", "p", "tau"]
    selected = matched[matched["n"].isin(sizes)]
    times = selected.pivot(index=key, columns="configuration", values="runtime").reindex(columns=MODELS)
    success = selected.pivot(index=key, columns="configuration", values="certified").reindex(columns=MODELS)
    cost = np.where(success.to_numpy(dtype=bool), times.to_numpy(), np.inf)
    best = np.min(cost, axis=1)
    ratios = np.full(cost.shape, np.inf)
    valid = np.isfinite(best)
    ratios[valid] = cost[valid] / best[valid, None]
    finite = ratios[np.isfinite(ratios)]
    xmax = max(2.0, float(np.ceil(finite.max()))) if finite.size else 2.0
    profile_colors = {"compact": "#555555", **COLORS}
    styles = ("--", "-", "-.", ":", (0, (3, 1, 1, 1)))
    fig, ax = plt.subplots(figsize=(6.2, 3.25))
    for j, (model, linestyle) in enumerate(zip(MODELS, styles)):
        finite_j = ratios[np.isfinite(ratios[:, j]), j]
        x = np.unique(np.r_[1.0, finite_j, xmax])
        y = (ratios[:, j, None] <= x[None, :]).mean(axis=0)
        ax.step(x, y, where="post", color=profile_colors[model], linewidth=1.45,
                linestyle=linestyle,
                label=DISPLAY[model])
    if log_scale:
        ax.set_xscale("log")
        readable_ticks = [tick for tick in (1, 2, 5, 10, 20, 30, 50, 100)
                          if tick <= xmax]
        ax.set_xticks(readable_ticks)
        ax.set_xticklabels([str(tick) for tick in readable_ticks])
    ax.set_xlim(1, xmax)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel(r"Runtime factor $q$ relative to the fastest proof of optimality")
    ax.set_ylabel("Share of instances solved to proven optimality")
    ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout(pad=0.3)
    return fig


def plot_cumulative_solved_profile(
    matched: pd.DataFrame,
    sizes: tuple[int, ...] = (10, 20, 30, 40, 50),
    time_limit: float = TIME_LIMIT,
    log_scale: bool = True,
) -> plt.Figure:
    """Plot the number of certified instances as a function of CPU time."""
    selected = matched[matched["n"].isin(sizes)].copy()
    if selected.empty:
        raise ValueError(f"No matched cases are available for sizes {sizes}")
    selected["runtime"] = pd.to_numeric(selected["runtime"], errors="coerce").clip(
        upper=time_limit
    )
    positive = selected.loc[selected["certified"] & selected["runtime"].gt(0), "runtime"]
    xmin = max(1e-3, 0.8 * float(positive.min())) if not positive.empty else 1e-3
    certified_times = selected.loc[selected["certified"], "runtime"].to_numpy(float)
    grid = np.unique(np.r_[xmin, certified_times, time_limit])
    grid = grid[(grid >= xmin) & (grid <= time_limit)]

    profile_colors = {"compact": "#555555", **COLORS}
    styles = ("--", "-", "-.", ":", (0, (3, 1, 1, 1)))
    fig, ax = plt.subplots(figsize=(6.2, 3.25))
    for model, linestyle in zip(MODELS, styles):
        method = selected[selected["configuration"].eq(model)]
        times = method.loc[method["certified"], "runtime"].to_numpy(float)
        solved = np.array([(times <= limit).sum() for limit in grid])
        ax.step(grid, solved, where="post", color=profile_colors[model],
                linewidth=1.45, linestyle=linestyle, label=DISPLAY[model])
    if log_scale:
        ax.set_xscale("log")
    ax.set_xlim(xmin, time_limit)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("CPU time (s)")
    ax.set_ylabel("Number of instances solved to optimality")
    ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout(pad=0.3)
    return fig


def group_order(summary: pd.DataFrame) -> list[tuple[int, float]]:
    return list(dict.fromkeys(zip(summary["n"], summary["tau"])))


def group_labels(groups: list[tuple[int, float]]) -> list[str]:
    return [rf"$n={n},\ \tau=3/2$" if tau == 1.5
            else rf"$n={n},\ \tau={int(tau)}$" for n, tau in groups]


def finish_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="0.88", linewidth=0.55, zorder=0)


def plot_dot(summary: pd.DataFrame) -> plt.Figure:
    """Horizontal dot plot. Best option when vertical space is limited."""
    groups = group_order(summary)
    y = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(6.7, 2.65))
    for model in PROJECTOR_MODELS:
        part = summary[summary["model"].eq(model)].set_index(["n", "tau"])
        values = [part.loc[group, "relative_deviation"] for group in groups]
        ax.scatter(values, y, s=27, marker=MARKERS[model], color=COLORS[model],
                   edgecolor="white", linewidth=0.35, label=DISPLAY[model], zorder=3)
    ax.axvline(0, color="0.25", linewidth=0.75)
    ax.set_yticks(y, group_labels(groups))
    ax.invert_yaxis()
    ax.set_xlabel(r"Relative deviation from the MICO LB gap (\%)")
    ax.legend(ncol=4, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), handletextpad=0.3, columnspacing=1.0)
    finish_axes(ax)
    fig.tight_layout(pad=0.25)
    return fig


def plot_lollipop(summary: pd.DataFrame) -> plt.Figure:
    """Small multiples emphasizing the gain obtained by each formulation."""
    groups = group_order(summary)
    fig, axes = plt.subplots(3, 3, figsize=(6.7, 3.65), sharex=True)
    for ax, group in zip(axes.flat, groups):
        part = summary[(summary["n"].eq(group[0])) &
                       (summary["tau"].eq(group[1]))].set_index("model")
        values = [part.loc[m, "relative_deviation"] for m in PROJECTOR_MODELS]
        positions = np.arange(4)
        for pos, model, value in zip(positions, PROJECTOR_MODELS, values):
            ax.plot([value, 0], [pos, pos], color="0.78", linewidth=1.0)
            ax.scatter(value, pos, color=COLORS[model], marker=MARKERS[model], s=19,
                       zorder=3)
        tau = "3/2" if group[1] == 1.5 else str(int(group[1]))
        ax.set_title(rf"$n={group[0]},\ \tau={tau}$", pad=2)
        ax.axvline(0, color="0.25", linewidth=0.65)
        ax.set_yticks(positions, [DISPLAY[m] for m in PROJECTOR_MODELS])
        ax.invert_yaxis()
        finish_axes(ax)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Deviation (\%)")
    fig.tight_layout(pad=0.35, h_pad=0.55, w_pad=0.55)
    return fig


def plot_heatmap(summary: pd.DataFrame) -> plt.Figure:
    """Very compact overview. Darker blue means a stronger improvement."""
    groups = group_order(summary)
    matrix = np.array([
        [summary[(summary["n"].eq(n)) & (summary["tau"].eq(tau)) &
                 (summary["model"].eq(model))]["relative_deviation"].iloc[0]
         for model in PROJECTOR_MODELS]
        for n, tau in groups
    ])
    limit = max(1.0, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(5.1, 2.75))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(4), [DISPLAY[m] for m in PROJECTOR_MODELS])
    ax.set_yticks(np.arange(len(groups)), group_labels(groups))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if abs(matrix[i, j]) > 0.55 * limit else "black"
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                    color=color, fontsize=7)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    bar.set_label(r"Deviation from MICO (\%)")
    ax.tick_params(length=0)
    fig.tight_layout(pad=0.25)
    return fig


def plot_index(summary: pd.DataFrame) -> plt.Figure:
    """Show the relaxation gap as an index, with MICO fixed at 100."""
    groups = group_order(summary)
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(6.7, 2.65))
    ax.axhline(100, color="0.25", linewidth=0.75, label="MICO")
    for model in PROJECTOR_MODELS:
        part = summary[summary["model"].eq(model)].set_index(["n", "tau"])
        values = [part.loc[group, "gap_index"] for group in groups]
        ax.plot(x, values, marker=MARKERS[model], markersize=4, linewidth=1.0,
                color=COLORS[model], label=DISPLAY[model])
    ax.set_xticks(x, group_labels(groups), rotation=35, ha="right")
    ax.set_ylabel("LB-gap index (MICO = 100)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.55)
    ax.legend(ncol=5, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), handletextpad=0.3, columnspacing=0.9)
    fig.tight_layout(pad=0.25)
    return fig


def plot_bar(summary: pd.DataFrame, n: int | None = None) -> plt.Figure:
    """Three tau blocks, each with one deviation bar per projector model.

    If ``n`` is given, the plot represents that instance size.  Otherwise,
    deviations are averaged over the available sizes for each value of tau.
    Missing tau blocks are explicitly labeled as unavailable.
    """
    data = summary if n is None else summary[summary["n"].eq(n)]
    taus = (1.5, 2.0, 3.0)
    centers = np.arange(len(taus), dtype=float)
    width = 0.18
    offsets = (np.arange(len(PROJECTOR_MODELS)) - 1.5) * width
    fig, ax = plt.subplots(figsize=(5.7, 2.65))

    for offset, model in zip(offsets, PROJECTOR_MODELS):
        values = []
        for tau in taus:
            selected = data[(data["tau"].eq(tau)) & (data["model"].eq(model))]
            values.append(float(selected["relative_deviation"].mean())
                          if not selected.empty else np.nan)
        ax.bar(centers + offset, values, width=width * 0.88,
               color=COLORS[model], label=DISPLAY[model],
               edgecolor="white", linewidth=0.35, zorder=3)

    for center, tau in zip(centers, taus):
        if data[data["tau"].eq(tau)].empty:
            ax.text(center, -0.15, "N/A", ha="center", va="top",
                    fontsize=7, color="0.35")
    ax.axhline(0, color="0.25", linewidth=0.75)
    ax.set_xticks(centers, [r"$\tau=3/2$", r"$\tau=2$", r"$\tau=3$"])
    ax.tick_params(axis="x", labelsize=6.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: rf"${value:g}\%$"))
    ax.set_ylabel(r"Deviation from the MICO LB gap")
    if n is not None:
        ax.text(0.985, 0.96, rf"$n={n}$", transform=ax.transAxes,
                ha="right", va="top")
    ax.legend(ncol=4, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), handlelength=1.1,
              handletextpad=0.35, columnspacing=1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.55, zorder=0)
    ax.margins(x=0.08)
    fig.tight_layout(pad=0.25)
    return fig


def plot_bar_all_sizes(
    summary: pd.DataFrame,
    sizes: tuple[int, ...] = (10, 20, 30, 40, 50),
) -> plt.Figure:
    """Hierarchical bars with an outer block for n and inner blocks for tau."""
    taus = (1.5, 2.0, 3.0)
    width = 0.16
    method_offsets = (np.arange(4) - 1.5) * width
    subgroup_gap = 0.88
    size_gap = 0.62
    positions: dict[tuple[int, float], float] = {}
    size_centers: dict[int, float] = {}
    cursor = 0.0
    for size in sizes:
        local = []
        for tau in taus:
            positions[(size, tau)] = cursor
            local.append(cursor)
            cursor += subgroup_gap
        size_centers[size] = float(np.mean(local))
        cursor += size_gap

    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    for offset, model in zip(method_offsets, PROJECTOR_MODELS):
        xs, values = [], []
        for size in sizes:
            for tau in taus:
                selected = summary[(summary["n"].eq(size)) &
                                   (summary["tau"].eq(tau)) &
                                   (summary["model"].eq(model))]
                xs.append(positions[(size, tau)] + offset)
                values.append(float(selected["relative_deviation"].iloc[0])
                              if not selected.empty else np.nan)
        ax.bar(xs, values, width=width * 0.9, color=COLORS[model],
               edgecolor="white", linewidth=0.3, label=DISPLAY[model], zorder=3)

    for size in sizes:
        for tau in taus:
            center = positions[(size, tau)]
            if summary[(summary["n"].eq(size)) & summary["tau"].eq(tau)].empty:
                ax.text(center, -0.12, "N/A", rotation=90, ha="center", va="top",
                        fontsize=5.5, color="0.45")
    boundaries = []
    for left, right in zip(sizes[:-1], sizes[1:]):
        boundaries.append((positions[(left, 3.0)] + positions[(right, 1.5)]) / 2)
    for boundary in boundaries:
        ax.axvline(boundary, color="0.72", linewidth=0.65)

    subgroup_centers = [positions[(size, tau)] for size in sizes for tau in taus]
    subgroup_labels = [r"$3/2$", r"$2$", r"$3$"] * len(sizes)
    ax.set_xticks(subgroup_centers, subgroup_labels)
    ax.tick_params(axis="x", pad=1.5, labelsize=6.0)
    for size, center in size_centers.items():
        ax.text(center, -0.16, rf"$n={size}$", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.5)
    ax.text(positions[(sizes[0], 1.5)] - 0.48, -0.048, r"$\tau=$",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=6.0)
    ax.axhline(0, color="0.25", linewidth=0.75)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: rf"${value:g}\%$"))
    #ax.set_ylabel(r"Deviation from the MICO LB gap")
    ax.set_ylabel("Improvement w.r.t. MICO Relax Gap")
    ax.legend(ncol=4, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), handlelength=1.1,
              handletextpad=0.35, columnspacing=1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.55, zorder=0)
    ax.margins(x=0.015)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.86, bottom=0.23)
    return fig


def final_mip_gap_summary(
    workbook: Path,
    sheet: str,
    selection: str = "any_unsolved",
) -> pd.DataFrame:
    """Average final MIP gaps on a common set of difficult matched cases.

    ``any_unsolved`` retains a case when at least one of the five methods did
    not prove optimality. ``compact_unsolved`` retains a case when MISOCO did
    not prove optimality. In either case, every method is averaged over the
    same selected cases within each ``(n, tau)`` group.
    """
    matched = matched_run_data(workbook, sheet)
    if "gap" not in matched:
        raise ValueError("Missing workbook column: gap")
    key = ["instance", "p", "tau"]
    certified = matched.pivot_table(
        index=key, columns="configuration", values="certified", aggfunc="all"
    ).reindex(columns=MODELS)
    if selection == "any_unsolved":
        selected_keys = certified.index[~certified.all(axis=1)]
    elif selection == "compact_unsolved":
        selected_keys = certified.index[~certified["compact"].astype(bool)]
    else:
        raise ValueError("selection must be 'any_unsolved' or 'compact_unsolved'")

    difficult = matched.set_index(key).loc[
        lambda frame: frame.index.isin(selected_keys)
    ].reset_index()
    difficult["gap_percent"] = (
        100.0 * pd.to_numeric(difficult["gap"], errors="coerce").clip(lower=0)
    )
    summary = (
        difficult.groupby(["n", "tau", "configuration"], as_index=False)
        .agg(mean_gap_percent=("gap_percent", "mean"), cases=("gap_percent", "count"))
        .rename(columns={"configuration": "model"})
    )
    summary["model"] = pd.Categorical(summary["model"], MODELS, ordered=True)
    return summary.sort_values(["n", "tau", "model"]).reset_index(drop=True)


def _hierarchical_bar_positions(
    sizes: tuple[int, ...],
    taus: tuple[float, ...] = (1.5, 2.0, 3.0),
) -> tuple[dict[tuple[int, float], float], dict[int, float]]:
    positions: dict[tuple[int, float], float] = {}
    size_centers: dict[int, float] = {}
    cursor = 0.0
    for size in sizes:
        local = []
        for tau in taus:
            positions[(size, tau)] = cursor
            local.append(cursor)
            cursor += 0.88
        size_centers[size] = float(np.mean(local))
        cursor += 0.62
    return positions, size_centers


def _finish_hierarchical_gap_axis(
    ax: plt.Axes,
    positions: dict[tuple[int, float], float],
    size_centers: dict[int, float],
    sizes: tuple[int, ...],
    ylabel: str,
) -> None:
    taus = (1.5, 2.0, 3.0)
    for left, right in zip(sizes[:-1], sizes[1:]):
        boundary = (positions[(left, 3.0)] + positions[(right, 1.5)]) / 2
        ax.axvline(boundary, color="0.72", linewidth=0.65)
    centers = [positions[(size, tau)] for size in sizes for tau in taus]
    ax.set_xticks(centers, [r"$3/2$", r"$2$", r"$3$"] * len(sizes))
    ax.tick_params(axis="x", pad=1.5, labelsize=6.0)
    for size, center in size_centers.items():
        ax.text(center, -0.16, rf"$n={size}$", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.5)
    ax.text(positions[(sizes[0], 1.5)] - 0.48, -0.048, r"$\tau=$",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=6.0)
    ax.axhline(0, color="0.25", linewidth=0.75)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: rf"${value:g}\%$"))
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.55, zorder=0)
    ax.margins(x=0.015)


def plot_average_unsolved_mip_gaps(
    summary: pd.DataFrame,
    sizes: tuple[int, ...] = (10, 20, 30, 40, 50),
) -> plt.Figure:
    """Plot mean final gaps on cases not solved by at least one method."""
    taus = (1.5, 2.0, 3.0)
    positions, size_centers = _hierarchical_bar_positions(sizes, taus)
    width = 0.13
    offsets = (np.arange(len(MODELS)) - 2.0) * width
    colors = {"compact": "#555555", **COLORS}
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    for offset, model in zip(offsets, MODELS):
        xs, values = [], []
        for size in sizes:
            for tau in taus:
                row = summary[(summary["n"].eq(size)) &
                              (summary["tau"].eq(tau)) &
                              (summary["model"].eq(model))]
                xs.append(positions[(size, tau)] + offset)
                values.append(float(row["mean_gap_percent"].iloc[0])
                              if not row.empty else np.nan)
        ax.bar(xs, values, width=width * 0.9, color=colors[model],
               edgecolor="white", linewidth=0.3, label=DISPLAY[model], zorder=3)
    for size in sizes:
        for tau in taus:
            if summary[(summary["n"].eq(size)) & summary["tau"].eq(tau)].empty:
                ax.text(positions[(size, tau)], -0.12, "N/A", rotation=90,
                        ha="center", va="top", fontsize=5.5, color="0.45")
    _finish_hierarchical_gap_axis(
        ax, positions, size_centers, sizes, "Average final MIP gap"
    )
    ax.legend(ncol=5, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), handlelength=1.1,
              handletextpad=0.35, columnspacing=0.9)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.86, bottom=0.23)
    return fig


def plot_mip_gap_improvement(
    summary: pd.DataFrame,
    sizes: tuple[int, ...] = (10, 20, 30, 40, 50),
) -> plt.Figure:
    """Plot percentage reduction in final MIP gap relative to MISOCO."""
    taus = (1.5, 2.0, 3.0)
    positions, size_centers = _hierarchical_bar_positions(sizes, taus)
    width = 0.16
    offsets = (np.arange(len(PROJECTOR_MODELS)) - 1.5) * width
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    for offset, model in zip(offsets, PROJECTOR_MODELS):
        xs, values = [], []
        for size in sizes:
            for tau in taus:
                group = summary[(summary["n"].eq(size)) & summary["tau"].eq(tau)]
                base = group[group["model"].eq("compact")]["mean_gap_percent"]
                method = group[group["model"].eq(model)]["mean_gap_percent"]
                value = np.nan
                if not base.empty and not method.empty and float(base.iloc[0]) > 1e-12:
                    value =  (float(base.iloc[0]) - float(method.iloc[0])) #/ float(base.iloc[0])
                xs.append(positions[(size, tau)] + offset)
                values.append(value)
        ax.bar(xs, values, width=width * 0.9, color=COLORS[model],
               edgecolor="white", linewidth=0.3, label=DISPLAY[model], zorder=3)
    for size in sizes:
        for tau in taus:
            if summary[(summary["n"].eq(size)) & summary["tau"].eq(tau)].empty:
                ax.text(positions[(size, tau)], -0.12, "N/A", rotation=90,
                        ha="center", va="top", fontsize=5.5, color="0.45")
    _finish_hierarchical_gap_axis(
        ax, positions, size_centers, sizes,
        "Final Opt Gap improvement over MICO",
    )
    ax.legend(ncol=4, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), handlelength=1.1,
              handletextpad=0.35, columnspacing=1.0)
    fig.subplots_adjust(left=0.10, right=0.995, top=0.86, bottom=0.23)
    return fig


PLOTS = {"dot": plot_dot, "bar": plot_bar, "bar-all": plot_bar_all_sizes,
         "lollipop": plot_lollipop,
         "heatmap": plot_heatmap, "index": plot_index}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path, nargs="?", default=Path("all_v1.xlsx"))
    parser.add_argument("--sheet", default="all_v1")
    parser.add_argument("--style", choices=(*PLOTS, "all"), default="all")
    parser.add_argument("--n", type=int,
                        help="Instance size for the three-block bar plot.")
    parser.add_argument("--output", type=Path,
                        help="Output PDF for one style. With --style all, this is a directory.")
    parser.add_argument("--tex", choices=("auto", "on", "off"), default="auto",
                        help="Use LaTeX text rendering when available (default: auto).")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--table-output", type=Path,
                        help="Also write the three-block LaTeX results table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_fonts(args.tex)
    summary = paired_summary(args.workbook, args.sheet)
    if args.table_output is not None:
        main_results = paired_main_results(args.workbook, args.sheet)
        latex_main_results_table(main_results, args.table_output)
        print(args.table_output)
    styles = list(PLOTS) if args.style == "all" else [args.style]
    output_dir = args.output if args.style == "all" and args.output else Path("figures")
    if args.style == "all":
        output_dir.mkdir(parents=True, exist_ok=True)
    for style in styles:
        if args.style == "all":
            destination = output_dir / f"relaxation_gap_{style}.pdf"
        else:
            destination = args.output or Path(f"relaxation_gap_{style}.pdf")
            destination.parent.mkdir(parents=True, exist_ok=True)
        figure = plot_bar(summary, args.n) if style == "bar" else PLOTS[style](summary)
        figure.savefig(destination, dpi=args.dpi)
        plt.close(figure)
        print(destination)


if __name__ == "__main__":
    main()
