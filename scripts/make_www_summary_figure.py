from __future__ import annotations

"""Build the three-panel empirical summary figure used in the WWW paper.

The figure is generated only from processed CSV outputs produced by
``scripts/run_all_experiments.py``.  No solver is run in this script.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(ROOT)}. Run the corresponding experiments "
            "or `python scripts/reproduce_all.py` first."
        )
    return path


def build_summary_figure() -> None:
    """Generate the WWW three-panel empirical summary in PDF and PNG."""
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Panel (a): ratios within each seed relative to unique stake (kappa = 1).
    architecture = pd.read_csv(_require(DATA / "experiment_a_architecture.csv"))
    architecture = architecture[architecture["regime"] == "global_clustered"].copy()
    baseline = architecture[architecture["kappa"] == 1][
        ["seed", "total_reward", "aggregate_cvar99"]
    ].rename(
        columns={
            "total_reward": "base_reward",
            "aggregate_cvar99": "base_cvar",
        }
    )
    architecture = architecture.merge(baseline, on="seed", validate="many_to_one")
    architecture["reward_ratio"] = architecture["total_reward"] / architecture["base_reward"]
    architecture["cvar_ratio"] = architecture["aggregate_cvar99"] / architecture["base_cvar"]
    architecture_summary = (
        architecture.groupby("kappa", as_index=False)
        .agg(
            reward_mean=("reward_ratio", "mean"),
            reward_se=("reward_ratio", "sem"),
            cvar_mean=("cvar_ratio", "mean"),
            cvar_se=("cvar_ratio", "sem"),
        )
        .sort_values("kappa")
    )

    # Panel (b): equilibrium and planner exposure at the intermediate background scale.
    competition = pd.read_csv(_require(DATA / "experiment_d_competition.csv"))
    competition = competition[competition["b_scale"] == 1.5].copy()
    competition_summary = (
        competition.groupby("n", as_index=False)
        .agg(
            ne_mean=("ne_total_gross_exposure", "mean"),
            ne_se=("ne_total_gross_exposure", "sem"),
            planner_mean=("planner_total_gross_exposure", "mean"),
            planner_se=("planner_total_gross_exposure", "sem"),
        )
        .sort_values("n")
    )

    # Panel (c): portfolios optimized under diagonal covariance, evaluated under truth.
    misspecification = pd.read_csv(_require(DATA / "experiment_b_misspecification.csv"))
    misspecification = misspecification[
        (misspecification["experiment"] == "B")
        & (misspecification["model"] == "diagonal")
    ].copy()
    regime_order = ["near_independent", "clustered", "global_clustered"]
    regime_labels = ["Near\nindependent", "Clustered", "Global +\nclustered"]
    misspecification_summary = (
        misspecification.groupby("regime")
        .agg(
            var_mean=("variance_ratio_vs_aware", "mean"),
            cvar_mean=("cvar99_ratio_vs_aware", "mean"),
        )
        .reindex(regime_order)
        .reset_index()
    )

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.5,
            "lines.markersize": 3.5,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), constrained_layout=True)

    ax = axes[0]
    ax.plot(
        architecture_summary["kappa"],
        architecture_summary["reward_mean"],
        marker="o",
        label="Strategic reward",
    )
    ax.fill_between(
        architecture_summary["kappa"],
        architecture_summary["reward_mean"] - 1.96 * architecture_summary["reward_se"],
        architecture_summary["reward_mean"] + 1.96 * architecture_summary["reward_se"],
        alpha=0.18,
    )
    ax.plot(
        architecture_summary["kappa"],
        architecture_summary["cvar_mean"],
        marker="o",
        label="99% CVaR",
    )
    ax.fill_between(
        architecture_summary["kappa"],
        architecture_summary["cvar_mean"] - 1.96 * architecture_summary["cvar_se"],
        architecture_summary["cvar_mean"] + 1.96 * architecture_summary["cvar_se"],
        alpha=0.18,
    )
    ax.axhline(1.0, linestyle="--", linewidth=0.8)
    ax.set_title("(a) Collateral-reuse frontier")
    ax.set_xlabel(r"Reuse cap $\kappa$")
    ax.set_ylabel(r"Ratio to $\kappa=1$")
    ax.set_xticks([1, 4, 8, 12])
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(
        competition_summary["n"],
        competition_summary["ne_mean"],
        marker="o",
        label="Nash equilibrium",
    )
    ax.fill_between(
        competition_summary["n"],
        competition_summary["ne_mean"] - 1.96 * competition_summary["ne_se"],
        competition_summary["ne_mean"] + 1.96 * competition_summary["ne_se"],
        alpha=0.18,
    )
    ax.plot(
        competition_summary["n"],
        competition_summary["planner_mean"],
        marker="o",
        label="Planner",
    )
    ax.fill_between(
        competition_summary["n"],
        competition_summary["planner_mean"] - 1.96 * competition_summary["planner_se"],
        competition_summary["planner_mean"] + 1.96 * competition_summary["planner_se"],
        alpha=0.18,
    )
    ax.set_title("(b) Competition and overexposure")
    ax.set_xlabel("Number of allocators")
    ax.set_ylabel("Aggregate gross exposure")
    ax.set_xticks([1, 4, 8, 12])
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[2]
    positions = np.arange(len(regime_order))
    width = 0.36
    ax.bar(positions - width / 2, misspecification_summary["var_mean"], width, label="Variance")
    ax.bar(positions + width / 2, misspecification_summary["cvar_mean"], width, label="99% CVaR")
    ax.axhline(1.0, linestyle="--", linewidth=0.8)
    ax.set_title("(c) Ignoring dependence")
    ax.set_ylabel("Risk ratio to aware optimum")
    ax.set_xticks(positions, regime_labels)
    ax.set_ylim(0.99, 1.075)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    pdf_path = FIGURES / "www_empirical_summary.pdf"
    png_path = FIGURES / "www_empirical_summary.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Refresh the review contact sheet now that the WWW figure exists.
    from make_figures import contact_sheet

    contact_sheet()
    print(f"Wrote {pdf_path.relative_to(ROOT)} and {png_path.relative_to(ROOT)}")


if __name__ == "__main__":
    build_summary_figure()
