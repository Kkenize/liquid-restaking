from __future__ import annotations

from pathlib import Path
from math import ceil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def mean_band(ax, data: pd.DataFrame, x: str, y: str, label: str, marker: str = "o") -> None:
    g = data.groupby(x)[y].agg(["mean", "sem"]).reset_index()
    line, = ax.plot(g[x], g["mean"], marker=marker, markersize=3.5, linewidth=1.4, label=label)
    ax.fill_between(g[x], g["mean"] - 1.96 * g["sem"], g["mean"] + 1.96 * g["sem"], alpha=0.12, color=line.get_color())


def architecture_frontier() -> None:
    df = pd.read_csv(DATA / "experiment_a_architecture.csv")
    metrics = {
        "total_reward": "Reward",
        "aggregate_expected_loss": "Expected claims",
        "aggregate_std": "Standard deviation",
        "aggregate_cvar99": "99% CVaR",
    }
    base = df[df.kappa == 1][["regime", "seed", *metrics.keys()]].copy()
    base = base.rename(columns={k: f"base_{k}" for k in metrics})
    df = df.merge(base, on=["regime", "seed"], how="left")
    for metric in metrics:
        df[f"norm_{metric}"] = df[metric] / df[f"base_{metric}"]
    regimes = ["near_independent", "clustered", "global_clustered"]
    titles = ["Near independence", "Clustered factors", "Global + clustered factors"]
    markers = ["o", "s", "^", "D"]
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.45), sharey=True)
    for ax, regime, title in zip(axes, regimes, titles):
        sub = df[df.regime == regime]
        for (metric, label), marker in zip(metrics.items(), markers):
            mean_band(ax, sub, "kappa", f"norm_{metric}", label, marker)
        ax.axhline(1.0, linewidth=0.7, linestyle=":")
        ax.set_title(title)
        ax.set_xlabel(r"Reuse cap $\kappa$")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r"Outcome / value at $\kappa=1$")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "architecture_frontier")

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.35))
    for regime, title, marker in zip(regimes, titles, ["o", "s", "^"]):
        mean_band(axes[0], df[df.regime == regime], "kappa", "mean_reuse_degree", title, marker)
        mean_band(axes[1], df[df.regime == regime], "kappa", "mean_effective_breadth", title, marker)
    axes[0].plot([1, 12], [1, 12], linestyle=":", linewidth=0.8, label="cap")
    axes[0].set_ylabel("Gross exposure / capital")
    axes[1].set_ylabel("Effective breadth")
    for ax in axes:
        ax.set_xlabel(r"Reuse cap $\kappa$")
        ax.grid(alpha=0.2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "architecture_exposure_breadth")


def misspecification() -> None:
    df = pd.read_csv(DATA / "experiment_b_misspecification.csv")
    main = df[(df.experiment == "B") & (df.model == "diagonal")].copy()
    regimes = ["near_independent", "clustered", "global_clustered"]
    labels = ["Near independent", "Clustered", "Global + clustered"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.55))
    means = main.groupby("regime")[["variance_ratio_vs_aware", "cvar99_ratio_vs_aware"]].mean().reindex(regimes)
    x = np.arange(len(regimes))
    width = 0.34
    axes[0].bar(x - width/2, means["variance_ratio_vs_aware"], width, label="Variance ratio")
    axes[0].bar(x + width/2, means["cvar99_ratio_vs_aware"], width, label="99% CVaR ratio")
    axes[0].axhline(1.0, linewidth=0.8, linestyle=":")
    axes[0].set_xticks(x, labels, rotation=12, ha="right")
    axes[0].set_ylabel("Diagonal / aware ratio")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    all_main = df[df.experiment == "B"].copy()
    order = ["diagonal", "factor_rank3_n250", "factor_rank3_n1000", "factor_rank3_n5000"]
    display = ["Diagonal", "Rank 3\n$n=250$", "Rank 3\n$n=1000$", "Rank 3\n$n=5000$"]
    for regime, label, marker in zip(regimes, labels, ["o", "s", "^"]):
        vals = all_main[all_main.regime == regime].groupby("model")["regret"].mean().reindex(order)
        axes[1].plot(range(len(order)), vals, marker=marker, linewidth=1.4, label=label)
    axes[1].set_xticks(range(len(order)), display)
    axes[1].set_ylabel("True utility regret")
    axes[1].set_yscale("symlog", linthresh=1e-5)
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    fig.tight_layout(pad=0.8, w_pad=1.5)
    save(fig, "independence_misspecification")


def factor_topology() -> None:
    df = pd.read_csv(DATA / "experiment_c_topology.csv")
    methods = ["reward_yield", "equal_weight", "lowest_p", "one_per_operator_cluster"]
    names = ["Reward yield", "Equal weight", "Lowest marginal $p$", "One per operator"]
    markers = ["o", "s", "^", "D"]
    fig, axes = plt.subplots(1, 2, figsize=(5.7, 2.35), sharey=True)
    for ax, kappa in zip(axes, [1.0, 4.0]):
        sub = df[df.kappa == kappa]
        for method, name, marker in zip(methods, names, markers):
            mean_band(ax, sub[sub.method == method], "global_share", "utility_gap_to_aware", name, marker)
        ax.set_title(rf"$\kappa={int(kappa)}$")
        ax.set_xlabel("Latent variance in global factor")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Utility loss relative to aware optimum")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, "factor_topology_heuristics")

    alloc = pd.read_csv(DATA / "experiment_c_allocations.csv")
    sub = alloc[(alloc.kappa == 4.0) & (alloc.global_share == 0.55) & (alloc.seed == 0)].copy()
    method_order = ["aware", "reward_yield", "equal_weight", "lowest_p", "one_per_operator_cluster"]
    matrix = sub.pivot(index="method", columns="position", values="exposure").reindex(method_order)
    fig, ax = plt.subplots(figsize=(6.2, 2.2))
    im = ax.imshow(matrix.values, aspect="auto")
    ax.set_yticks(range(len(method_order)), ["Aware", "Reward yield", "Equal", "Lowest $p$", "One/operator"])
    ax.set_xticks(range(matrix.shape[1]), [str(i + 1) for i in matrix.columns], rotation=0)
    ax.set_xlabel("Slashable position")
    ax.set_title(r"Representative allocations: strong global factor, $\kappa=4$")
    fig.colorbar(im, ax=ax, label="Exposure")
    fig.tight_layout()
    save(fig, "factor_topology_allocations")


def competition() -> None:
    df = pd.read_csv(DATA / "experiment_d_competition.csv")
    bvals = [0.5, 1.5, 5.0]
    fig, axes = plt.subplots(2, 3, figsize=(7.35, 4.0), sharex=True)
    for col, b in enumerate(bvals):
        sub = df[df.b_scale == b]
        for prefix, label, marker in [("ne", "Equilibrium", "o"), ("planner", "Planner", "s")]:
            mean_band(axes[0, col], sub, "n", f"{prefix}_total_gross_exposure", label, marker)
            mean_band(axes[1, col], sub, "n", f"{prefix}_aggregate_cvar99", label, marker)
        axes[0, col].set_title(rf"Background scale $={b}$")
        axes[1, col].set_xlabel("Number of allocators")
        for row in range(2):
            axes[row, col].grid(alpha=0.2)
    axes[0, 0].set_ylabel("Total gross exposure")
    axes[1, 0].set_ylabel("Aggregate 99% CVaR")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "competition_equilibrium_planner")

    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for b, marker in zip(bvals, ["o", "s", "^"]):
        mean_band(ax, df[df.b_scale == b], "n", "welfare_gap", rf"Background scale {b}", marker)
    ax.set_xlabel("Number of allocators")
    ax.set_ylabel("Planner welfare minus equilibrium welfare")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "competition_welfare_gap")


def solver_study() -> None:
    df = pd.read_csv(DATA / "experiment_e_solvers.csv")
    df["size"] = df.n * df.m
    solvers = ["br_cyclic", "br_random", "pg_tuned", "pg_conservative", "pg_sqrt_decay", "natural_residual"]
    names = ["Cyclic BR", "Random BR", "Adaptive PG", "Fixed-step PG", r"Square-root-decay PG", "Residual minimizer"]
    markers = ["o", "s", "^", "D", "P", "v"]
    conditions = ["easy", "clustered", "stress"]
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.45), sharey=True)
    for ax, condition in zip(axes, conditions):
        sub = df[df.condition == condition]
        for solver, name, marker in zip(solvers, names, markers):
            d = sub[sub.solver == solver]
            if d.empty:
                continue
            g = d.groupby("size")["runtime"].median().reset_index()
            ax.plot(g["size"], g["runtime"], marker=marker, markersize=3.5, linewidth=1.3, label=name)
        ax.set_title(condition.capitalize())
        ax.set_xlabel(r"Problem size $nm$")
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Median runtime (seconds)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.11))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, "solver_scaling")

    hist = pd.read_csv(DATA / "experiment_e_histories.csv")
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for solver, name, marker in [
        ("br_cyclic", "Cyclic BR", "o"),
        ("pg_tuned", "Adaptive PG", "s"),
        ("pg_conservative", "Fixed-step PG", "^"),
        ("pg_sqrt_decay", r"Square-root-decay PG", "P"),
    ]:
        d = hist[hist.solver == solver].copy()
        # Downsample long traces for a legible vector figure.
        if len(d) > 1000:
            idx = np.unique(np.geomspace(1, len(d), 500).astype(int) - 1)
            d = d.iloc[idx]
        ax.plot(d.iteration, np.maximum(d.step_norm, 1e-12), marker=marker if len(d) < 50 else None, linewidth=1.2, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sweep or projected iteration")
    ax.set_ylabel("Successive-step norm")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "solver_convergence_trace")


def risk_robustness() -> None:
    df = pd.read_csv(DATA / "experiment_f_risk_robustness.csv")
    theta = df[df.family == "theta"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.35))
    dist = theta[theta.method == "entropic"].groupby("risk_parameter")["distance_to_matched_other"].mean().reset_index()
    axes[0].plot(dist.risk_parameter, dist.distance_to_matched_other, marker="o")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Risk sensitivity $\theta=\eta$")
    axes[0].set_ylabel(r"$\ell_1$ distance: entropic vs. MV")
    axes[0].grid(alpha=0.2)

    for method, label, marker in [("mean_variance", "Mean--variance", "o"), ("entropic", "Entropic", "s")]:
        mean_band(axes[1], theta[theta.method == method], "risk_parameter", "gross_exposure", label, marker)
        mean_band(axes[2], theta[theta.method == method], "risk_parameter", "cvar99", label, marker)
    for ax in axes[1:]:
        ax.set_xscale("log")
        ax.set_xlabel(r"Risk sensitivity $\theta=\eta$")
        ax.grid(alpha=0.2)
    axes[1].set_ylabel("Gross exposure")
    axes[2].set_ylabel("99% CVaR")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    save(fig, "risk_measure_entropic")

    zeta = df[df.family == "zeta"].copy()
    fig, ax1 = plt.subplots(figsize=(4.2, 2.6))
    g = zeta.groupby("risk_parameter")[["gross_exposure", "cvar95"]].mean().reset_index()
    line1, = ax1.plot(g.risk_parameter, g.gross_exposure, marker="o", label="Gross exposure")
    ax1.set_xlabel(r"CVaR-premium weight $\zeta$")
    ax1.set_ylabel("Gross exposure")
    ax1.grid(alpha=0.2)
    ax2 = ax1.twinx()
    line2, = ax2.plot(g.risk_parameter, g.cvar95, marker="s", linestyle="--", label="95% CVaR")
    ax2.set_ylabel("95% CVaR")
    ax1.legend([line1, line2], [line1.get_label(), line2.get_label()], frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "risk_measure_cvar")


def validation_figure() -> None:
    df = pd.read_csv(DATA / "experiment_b_misspecification.csv")
    val = df[df.experiment == "B_validation"]
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    ax.plot(val.m, val.observed_multiplier, marker="o", label="Computed")
    ax.plot(val.m, val.theoretical_multiplier, linestyle="--", label=r"$1+(m-1)\rho$")
    ax.set_xlabel("Number of positions")
    ax.set_ylabel("True / diagonal variance")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "independence_multiplier_validation")



def contact_sheet() -> None:
    """Create a review sheet from all generated PNG figures."""
    image_paths = sorted(
        path for path in FIG.glob("*.png") if path.name != "contact_sheet.png"
    )
    if not image_paths:
        return
    columns = 3
    rows = ceil(len(image_paths) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3.4 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    for ax, path in zip(axes_array, image_paths):
        image = plt.imread(path)
        ax.imshow(image)
        ax.set_title(path.stem.replace("_", " "), fontsize=10)
        ax.axis("off")
    for ax in axes_array[len(image_paths):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "contact_sheet.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    architecture_frontier()
    misspecification()
    factor_topology()
    competition()
    solver_study()
    risk_robustness()
    validation_figure()
    contact_sheet()
    print(f"Wrote figures to {FIG}")


if __name__ == "__main__":
    main()
