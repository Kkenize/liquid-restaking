from __future__ import annotations

"""Generate the human-readable results summary from current processed CSVs."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUTPUT = ROOT / "RESULTS_SUMMARY.md"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    a = pd.read_csv(DATA / "experiment_a_architecture.csv")
    b = pd.read_csv(DATA / "experiment_b_misspecification.csv")
    c = pd.read_csv(DATA / "experiment_c_topology.csv")
    d = pd.read_csv(DATA / "experiment_d_competition.csv")
    e = pd.read_csv(DATA / "experiment_e_solvers.csv")
    f = pd.read_csv(DATA / "experiment_f_risk_robustness.csv")
    validation = pd.read_csv(DATA / "validation_report.csv")

    lines: list[str] = [
        "# Results summary",
        "",
        "This file is generated directly from the current CSV outputs by",
        "`scripts/summarize_results.py`. The experiments are synthetic stress",
        "tests and should not be interpreted as estimates of current protocol",
        "returns or slash frequencies.",
        "",
        "## Validation",
        "",
    ]
    val_rows = []
    for _, row in validation.iterrows():
        val_rows.append([
            str(row["test"]),
            f"{float(row['value']):.3e}",
            f"{float(row['tolerance']):.3e}",
            "yes" if bool(row["passed"]) else "no",
        ])
    lines.extend([table(["Test", "Observed", "Tolerance", "Passed"], val_rows), ""])

    lines.extend(["## A. Collateral architecture frontier", ""])
    names = {
        "near_independent": "Near independent",
        "clustered": "Clustered",
        "global_clustered": "Global + clustered",
    }
    a_rows = []
    for regime, label in names.items():
        grouped = a[(a["regime"] == regime) & (a["kappa"].isin([1, 12]))].groupby("kappa").mean(numeric_only=True)
        one, full = grouped.loc[1], grouped.loc[12]
        a_rows.append([
            label,
            f"{full['mean_reuse_degree']:.2f}",
            f"{full['total_reward'] / one['total_reward']:.2f}",
            f"{full['aggregate_expected_loss'] / one['aggregate_expected_loss']:.2f}",
            f"{full['aggregate_cvar99'] / one['aggregate_cvar99']:.2f}",
        ])
    lines.extend([
        table(["Dependence", "Reuse at cap 12", "Reward multiplier", "Expected-claim multiplier", "99% CVaR multiplier"], a_rows),
        "",
        "The cap is not always exhausted: stronger common factors reduce the",
        "equilibrium demand for additional reused exposure.",
        "",
    ])

    lines.extend(["## B. Dependence misspecification", ""])
    b_main = b[(b["experiment"] == "B") & (b["model"] == "diagonal")]
    b_rows = []
    for regime, label in names.items():
        row = b_main[b_main["regime"] == regime].mean(numeric_only=True)
        b_rows.append([
            label,
            f"{row['regret']:.5f}",
            f"{100 * (row['variance_ratio_vs_aware'] - 1):.1f}%",
            f"{100 * (row['cvar99_ratio_vs_aware'] - 1):.1f}%",
            f"{row['allocation_turnover']:.3f}",
        ])
    lines.extend([table(["Dependence", "Utility regret", "Excess true variance", "Excess 99% CVaR", "Turnover"], b_rows), ""])

    lines.extend(["## C. Factor topology", ""])
    c_sub = c[(c["kappa"] == 4.0) & (c["global_share"] == 0.55) & (c["method"] != "aware")]
    c_rows = []
    display = {
        "reward_yield": "Reward yield",
        "equal_weight": "Equal weight",
        "lowest_p": "Lowest marginal probability",
        "one_per_operator_cluster": "One per operator cluster",
    }
    for method, label in display.items():
        row = c_sub[c_sub["method"] == method].mean(numeric_only=True)
        c_rows.append([label, f"{row['utility_gap_to_aware']:.5f}", f"{row['gross_exposure']:.3f}", f"{row['cvar99']:.3f}"])
    lines.extend([table(["Heuristic", "Utility loss", "Gross exposure", "99% CVaR"], c_rows), ""])

    lines.extend(["## D. Competition and welfare", ""])
    d_sub = d[(d["b_scale"] == 1.5) & (d["n"].isin([1, 4, 12]))].groupby("n").mean(numeric_only=True)
    d_rows = []
    for n in [1, 4, 12]:
        row = d_sub.loc[n]
        d_rows.append([
            str(n),
            f"{row['ne_total_gross_exposure']:.3f}",
            f"{row['planner_total_gross_exposure']:.3f}",
            f"{row['ne_aggregate_cvar99']:.3f}",
            f"{row['planner_aggregate_cvar99']:.3f}",
            f"{row['welfare_gap']:.4f}",
        ])
    lines.extend([table(["Allocators", "NE exposure", "Planner exposure", "NE 99% CVaR", "Planner 99% CVaR", "Welfare gap"], d_rows), ""])

    lines.extend(["## E. Solver behavior", ""])
    e_group = e.groupby("solver").agg(
        convergence_rate=("converged", "mean"),
        median_iterations=("iterations", "median"),
        median_runtime=("runtime", "median"),
        max_exploitability=("exploitability", "max"),
        max_vi_residual=("vi_residual", "max"),
    )
    solver_order = ["br_cyclic", "br_reverse", "br_random", "pg_tuned", "pg_conservative", "pg_sqrt_decay", "natural_residual"]
    solver_names = {
        "br_cyclic": "Cyclic best response",
        "br_reverse": "Reverse best response",
        "br_random": "Randomized best response",
        "pg_tuned": "Adaptive projected gradient",
        "pg_conservative": "Fixed-step projected gradient",
        "pg_sqrt_decay": "Square-root-decay projected gradient",
        "natural_residual": "Natural-residual minimizer",
    }
    e_rows = []
    for solver in solver_order:
        if solver not in e_group.index:
            continue
        row = e_group.loc[solver]
        e_rows.append([
            solver_names[solver],
            f"{100 * row['convergence_rate']:.0f}%",
            f"{row['median_iterations']:.0f}",
            f"{row['median_runtime']:.3f}",
            f"{row['max_exploitability']:.1e}",
            f"{row['max_vi_residual']:.1e}",
        ])
    lines.extend([
        table(["Solver", "Converged", "Median iterations", "Median runtime (s)", "Max exploitability", "Max VI residual"], e_rows),
        "",
        "Runtime is machine dependent; exploitability, VI residual, and",
        "cross-solver agreement are the substantive diagnostics.",
        "",
    ])

    lines.extend(["## F. Risk-objective robustness", ""])
    theta = f[(f["family"] == "theta") & (f["risk_parameter"] == 2.0)].groupby("method").mean(numeric_only=True)
    f_rows = []
    for method, label in [("mean_variance", "Mean-variance"), ("entropic", "Entropic")]:
        row = theta.loc[method]
        f_rows.append([label, f"{row['gross_exposure']:.3f}", f"{row['cvar99']:.3f}", f"{row['reward']:.3f}"])
    lines.extend([table(["Objective at sensitivity 2", "Gross exposure", "99% CVaR", "Reward"], f_rows), ""])

    zeta = f[f["family"] == "zeta"].groupby("risk_parameter").mean(numeric_only=True)
    z_rows = []
    for value in [0.05, 0.25, 0.50, 1.0]:
        row = zeta.loc[value]
        z_rows.append([f"{value:.2f}", f"{row['gross_exposure']:.3f}", f"{row['cvar95']:.3f}", f"{row['cvar99']:.3f}"])
    lines.extend([table(["CVaR premium zeta", "Gross exposure", "95% CVaR", "99% CVaR"], z_rows), ""])

    lines.extend([
        "## Output inventory",
        "",
        f"- Architecture equilibria: {len(a)}",
        f"- Misspecification rows: {len(b)}",
        f"- Factor-topology portfolios: {len(c)}",
        f"- Competition instances: {len(d)}",
        f"- Solver runs: {len(e)}",
        f"- Risk-objective runs: {len(f)}",
        "",
        "See `figures/contact_sheet.png` for a visual index and",
        "`results/reproduction_check.json` for machine-readable checks.",
    ])

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
