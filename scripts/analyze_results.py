"""Recompute final-manuscript statistics and figures from measured runs only."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
ALGORITHMS = ["qega", "qsa", "qpso", "sga", "sweep"]
SORTERS = ["mks", "rds", "dps"]
LEVELS = {
    "N": [200, 500, 1000, 2000, 5000],
    "K": [10, 15, 20, 25, 30],
    "M": [8, 18, 35, 65, 150],
    "type": ["Type-R", "Type-U"],
}
COLORS = dict(zip(ALGORITHMS, ["#0072B2", "#D55E00", "#009E73", "#A51E36", "#6B6B6B"]))


def ci95(values) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))


def rank_biserial(x, y) -> float:
    differences = np.asarray(x) - np.asarray(y)
    differences = differences[differences != 0]
    if not len(differences):
        return 0.0
    ranks = stats.rankdata(np.abs(differences))
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    return float((positive - negative) / (positive + negative))


def load_frame(results: Path, name: str) -> pd.DataFrame:
    path = results / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; complete this experiment before analysis")
    frame = pd.read_csv(path)
    if not frame["feasible"].all():
        raise ValueError(f"infeasible observations in {path}")
    return frame


def add_rpd(frame: pd.DataFrame, bks: pd.Series) -> pd.DataFrame:
    result = frame.copy()
    reference = result["instance_id"].map(bks)
    result["rpd"] = (result["makespan_min"] - reference) / reference * 100.0
    if result["rpd"].isna().any() or (result["rpd"] < -1e-8).any():
        raise ValueError("invalid RPD against the fixed BKS pool")
    result["rpd"] = result["rpd"].clip(lower=0.0)
    return result


def paired_statistics(frame: pd.DataFrame, column: str, order: list[str]) -> pd.DataFrame:
    pivot = frame.pivot(index="instance_id", columns=column, values="rpd").dropna()
    pivot = pivot[order]
    test = stats.friedmanchisquare(*(pivot[name] for name in order))
    n, k = pivot.shape
    ranks = pivot.rank(axis=1, method="average")
    mean_ranks = ranks.mean()
    kendall_w = test.statistic / (n * (k - 1))
    rows = [{
        "panel": "Omnibus", "method1": "Friedman", "method2": "",
        "statistic": test.statistic, "p_value": test.pvalue,
        "kendall_w": kendall_w, "rank1": np.nan, "rank2": np.nan,
        "rank_biserial": np.nan, "sample_size": n,
    }]
    for name in order:
        rows.append({
            "panel": "Mean rank", "method1": name, "method2": "",
            "statistic": mean_ranks[name], "p_value": np.nan,
            "kendall_w": kendall_w, "rank1": mean_ranks[name], "rank2": np.nan,
            "rank_biserial": np.nan, "sample_size": n,
        })
    denominator = math.sqrt(k * (k + 1) / (12 * n))
    for first, second in itertools.combinations(order, 2):
        q = abs(mean_ranks[first] - mean_ranks[second]) / denominator
        rows.append({
            "panel": "Nemenyi", "method1": first, "method2": second,
            "statistic": mean_ranks[first] - mean_ranks[second],
            "p_value": stats.studentized_range.sf(q, k, np.inf),
            "kendall_w": kendall_w, "rank1": mean_ranks[first],
            "rank2": mean_ranks[second],
            "rank_biserial": rank_biserial(pivot[first], pivot[second]),
            "sample_size": n,
        })
    return pd.DataFrame(rows)


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "SimSun", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    })


def save_table(frame: pd.DataFrame, out: Path, name: str) -> None:
    frame.to_csv(out / f"{name}.csv", index=False, float_format="%.10g")
    (out / f"{name}.tex").write_text(
        frame.to_latex(index=False, float_format=lambda value: f"{value:.4f}"),
        encoding="utf-8",
    )


def save_figure(fig: plt.Figure, figures: Path, number: int) -> None:
    fig.savefig(figures / f"Figure{number}.pdf")
    fig.savefig(figures / f"Figure{number}.png", dpi=300)
    plt.close(fig)


def taguchi_outputs(frame: pd.DataFrame, out: Path, figures: Path) -> None:
    params = frame["params_json"].map(json.loads)
    data = frame.copy()
    for factor in ["n", "N_p", "p_c", "p_m", "alpha"]:
        data[factor] = params.map(lambda p, key=factor: p[key])
    rows = []
    for label, group in data.groupby("label"):
        rpd = group["rpd"].to_numpy()
        rows.append({
            "configuration": label, **{factor: group[factor].iloc[0]
                                        for factor in ["n", "N_p", "p_c", "p_m", "alpha"]},
            "mean_rpd": rpd.mean(), "sd_rpd": rpd.std(ddof=1),
            "runtime_s": group["runtime_s"].mean(),
            "sn_ratio": -10 * np.log10(np.mean(rpd**2)), "runs": len(group),
        })
    table = pd.DataFrame(rows).sort_values("configuration")
    save_table(table, out, "Table7_TaguchiL16")
    response = []
    for factor in ["n", "N_p", "p_c", "p_m", "alpha"]:
        means = table.groupby(factor)["mean_rpd"].mean()
        delta = means.max() - means.min()
        for level, value in means.items():
            response.append({"factor": factor, "level": level, "mean_rpd": value, "delta": delta})
    response = pd.DataFrame(response)
    ranked = response.drop_duplicates("factor").sort_values("delta", ascending=False)
    rank_map = {factor: i + 1 for i, factor in enumerate(ranked["factor"])}
    response["rank"] = response["factor"].map(rank_map)
    save_table(response, out, "Table8_ParameterResponse")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5))
    for ax, factor in zip(axes.flat, ["n", "N_p", "p_c", "p_m", "alpha"]):
        group = response[response["factor"].eq(factor)].sort_values("level")
        ax.plot(group["level"], group["mean_rpd"], marker="o", color="#0072B2")
        ax.set(title=factor, xlabel="Parameter level", ylabel="Mean RPD (%)")
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    save_figure(fig, figures, 5)


def n_outputs(frame: pd.DataFrame, out: Path, figures: Path) -> None:
    data = frame.copy()
    data["n"] = data["params_json"].map(lambda value: json.loads(value)["n"])
    rows = []
    for (N, n), group in data.groupby(["N", "n"]):
        rows.append({"N": N, "n": n, "mean_rpd": group["rpd"].mean(),
                     "ci95": ci95(group["rpd"]), "instances": len(group)})
    trend = pd.DataFrame(rows).sort_values(["N", "n"])
    save_table(trend, out, "Figure6_NSensitivity")
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    for N, group in trend.groupby("N"):
        ax.errorbar(group["n"], group["mean_rpd"], yerr=group["ci95"],
                    marker="o", capsize=3, label=f"N={N}")
    ax.set(xlabel="Number of clusters, n", ylabel="Mean RPD (%)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, 6)


def sorting_outputs(frame: pd.DataFrame, out: Path, figures: Path) -> None:
    data = frame.copy()
    data["sorter"] = data["params_json"].map(lambda value: json.loads(value)["sorter"])
    summary = data.groupby("sorter").agg(
        mean_rpd=("rpd", "mean"), ci95=("rpd", ci95),
        instances=("instance_id", "nunique")).reindex(SORTERS).reset_index()
    save_table(summary, out, "Figure7_SortingSummary")
    save_table(paired_statistics(data, "sorter", SORTERS), out, "SortingStatistics")
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    ax.bar(summary["sorter"].str.upper(), summary["mean_rpd"], yerr=summary["ci95"],
           color=["#0072B2", "#6B6B6B", "#A51E36"], alpha=0.7, capsize=3)
    ax.set(xlabel="Sorting method", ylabel="Mean RPD (%)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, 7)


def algorithm_outputs(frame: pd.DataFrame, out: Path, figures: Path) -> None:
    rows = []
    for factor, levels in LEVELS.items():
        for level in levels:
            for algorithm in ALGORITHMS:
                group = frame[(frame["algorithm"].eq(algorithm)) & (frame[factor].eq(level))]
                rows.append({"factor": factor, "level": level, "algorithm": algorithm,
                             "arpd": group["rpd"].mean(), "ci95": ci95(group["rpd"]),
                             "instances": group["instance_id"].nunique()})
    for algorithm in ALGORITHMS:
        group = frame[frame["algorithm"].eq(algorithm)]
        rows.append({"factor": "Overall", "level": "Average", "algorithm": algorithm,
                     "arpd": group["rpd"].mean(), "ci95": ci95(group["rpd"]),
                     "instances": group["instance_id"].nunique()})
    table = pd.DataFrame(rows)
    save_table(table, out, "Table10_AlgorithmComparison")
    save_table(paired_statistics(frame, "algorithm", ALGORITHMS), out, "AlgorithmStatistics")
    overall = table[table["factor"].eq("Overall")].set_index("algorithm").reindex(ALGORITHMS)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar([a.upper() for a in ALGORITHMS], overall["arpd"], yerr=overall["ci95"],
           color=[COLORS[a] for a in ALGORITHMS], alpha=0.7, capsize=3)
    ax.set(ylabel="ARPD (%)", xlabel="Algorithm")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, 8)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, factor in zip(axes, ["N", "K", "M"]):
        for algorithm in ALGORITHMS:
            group = table[(table["factor"].eq(factor)) & table["algorithm"].eq(algorithm)].copy()
            group["level"] = pd.to_numeric(group["level"])
            group = group.sort_values("level")
            ax.errorbar(group["level"], group["arpd"], yerr=group["ci95"],
                        marker="o", capsize=2, color=COLORS[algorithm], label=algorithm.upper())
        ax.set(xscale="log", xlabel=factor, ylabel="ARPD (%)")
        ax.set_xticks(LEVELS[factor], [str(level) for level in LEVELS[factor]])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    save_figure(fig, figures, 9)
    runtime = frame.groupby(["algorithm", "N"]).agg(
        mean_runtime_s=("runtime_s", "mean"), ci95=("runtime_s", ci95),
        instances=("instance_id", "nunique")).reset_index()
    save_table(runtime, out, "Figure10_Runtime")
    fig, ax = plt.subplots(figsize=(4.5, 3.1))
    for algorithm in ALGORITHMS:
        group = runtime[runtime["algorithm"].eq(algorithm)].sort_values("N")
        ax.errorbar(group["N"], group["mean_runtime_s"], yerr=group["ci95"],
                    marker="o", capsize=2, color=COLORS[algorithm], label=algorithm.upper())
    ax.set(xscale="log", yscale="log", xlabel="Number of orders, N",
           ylabel="Wall-clock runtime (s)")
    ax.set_xticks(LEVELS["N"], [str(level) for level in LEVELS["N"]])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    save_figure(fig, figures, 10)


def uav_outputs(frame: pd.DataFrame, out: Path, figures: Path) -> None:
    data = frame.copy()
    data["base_instance"] = data["instance_id"].str.replace(r"_dr[0-9.]+$", "", regex=True)
    data["uav_ratio"] = data["instance_id"].str.extract(r"_dr([0-9.]+)$").astype(float)
    baseline = data[np.isclose(data["uav_ratio"], 0.0)][
        ["base_instance", "run", "seed", "makespan_min"]].rename(
        columns={"makespan_min": "baseline_makespan"})
    paired = data.merge(baseline, on=["base_instance", "run", "seed"], validate="many_to_one")
    paired["change_pct"] = ((paired["makespan_min"] - paired["baseline_makespan"])
                            / paired["baseline_makespan"] * 100.0)
    rows = []
    for (N, ratio), group in paired.groupby(["N", "uav_ratio"]):
        rows.append({"N": N, "uav_ratio": ratio, "change_pct": group["change_pct"].mean(),
                     "ci95": ci95(group["change_pct"]), "instances": len(group)})
    table = pd.DataFrame(rows).sort_values(["N", "uav_ratio"])
    save_table(table, out, "UAVAblation")
    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    for N, group in table.groupby("N"):
        ax.errorbar(group["uav_ratio"] * 100, group["change_pct"], yerr=group["ci95"],
                    marker="o", capsize=3, label=f"N={N}")
    ax.axhline(0, color="0.5", linewidth=0.7)
    ax.set(xlabel="UAV proportion (%)", ylabel="Makespan change (%)",
           xticks=[0, 10, 20, 30, 40])
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, 11)


def clustering_outputs(frame: pd.DataFrame, bounds: pd.DataFrame,
                       out: Path, figures: Path) -> None:
    merged = frame.merge(bounds[["instance_id", "lower_bound_min", "solver_time_s"]],
                         on="instance_id", validate="many_to_one")
    merged["gap_pct"] = ((merged["makespan_min"] - merged["lower_bound_min"])
                         / merged["lower_bound_min"] * 100.0)
    table = merged.groupby(["N", "algorithm"]).agg(
        lower_bound_min=("lower_bound_min", "mean"), solver_time_s=("solver_time_s", "mean"),
        gap_pct=("gap_pct", "mean"), instances=("instance_id", "nunique")).reset_index()
    save_table(table, out, "LowerBoundGaps")
    search = pd.DataFrame({"N": LEVELS["N"]})
    search["n"] = search["N"].map(lambda N: min(25, max(15, round(N / 50))))
    search["log10_order_space"] = search["N"].map(lambda N: math.lgamma(N + 1) / math.log(10))
    search["log10_cluster_space"] = search["n"].map(lambda n: math.lgamma(n + 1) / math.log(10))
    save_table(search, out, "SearchSpaceCompression")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axes[0].plot(search["N"], search["log10_order_space"], "o-", label=r"$\log_{10}(N!)$")
    axes[0].plot(search["N"], search["log10_cluster_space"], "s--", label=r"$\log_{10}(n!)$")
    axes[0].set(xscale="log", xlabel="Orders, N", ylabel="Search-space size")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    width = 0.35
    Ns = sorted(table["N"].unique())
    for offset, algorithm in [(-width / 2, "qega"), (width / 2, "order_search")]:
        group = table[table["algorithm"].eq(algorithm)].set_index("N").reindex(Ns)
        axes[1].bar(np.arange(len(Ns)) + offset, group["gap_pct"], width,
                    label=algorithm, color=COLORS.get(algorithm, "#6B6B6B"))
    axes[1].set(xticks=np.arange(len(Ns)), xticklabels=Ns,
                xlabel="Orders, N", ylabel="Gap to LB (%)")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figures, 12)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    results = args.results
    out = args.output or results / "analysis"
    figures = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    names = ["main", "sorting", "taguchi_qega", "n_sensitivity", "uav_ablation", "clustering"]
    frames = {name: load_frame(results, name) for name in names}
    pool = pd.concat(frames.values(), ignore_index=True)
    bks = pool.groupby("instance_id")["makespan_min"].min()
    for name in ["main", "sorting", "taguchi_qega", "n_sensitivity"]:
        frames[name] = add_rpd(frames[name], bks)
    bks.rename("best_known_makespan_min").reset_index().to_csv(
        out / "fixed_bks.csv", index=False)
    taguchi_outputs(frames["taguchi_qega"], out, figures)
    n_outputs(frames["n_sensitivity"], out, figures)
    sorting_outputs(frames["sorting"], out, figures)
    algorithm_outputs(frames["main"], out, figures)
    uav_outputs(frames["uav_ablation"], out, figures)
    bounds = pd.read_csv(results / "milp_bounds.csv")
    clustering_outputs(frames["clustering"], bounds, out, figures)
    print(f"Measured analysis written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
