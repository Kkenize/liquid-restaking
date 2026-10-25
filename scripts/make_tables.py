from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def pct(x: float) -> str:
    return f"{100*x:.1f}\\%"


def architecture_table() -> None:
    df = pd.read_csv(DATA / "experiment_a_architecture.csv")
    rows = []
    names = {"near_independent": "Near independent", "clustered": "Clustered", "global_clustered": "Global + clustered"}
    for regime, name in names.items():
        g = df[(df.regime == regime) & (df.kappa.isin([1, 12]))].groupby("kappa").mean(numeric_only=True)
        one, full = g.loc[1], g.loc[12]
        rows.append({
            "Dependence": name,
            "Reuse at full": full.mean_reuse_degree,
            "Reward mult.": full.total_reward / one.total_reward,
            "Expected-claim mult.": full.aggregate_expected_loss / one.aggregate_expected_loss,
            "Std. mult.": full.aggregate_std / one.aggregate_std,
            "99\\% CVaR mult.": full.aggregate_cvar99 / one.aggregate_cvar99,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "architecture_endpoints.csv", index=False)
    tex = out.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.2f}", column_format="lrrrrr")
    (OUT / "architecture_endpoints.tex").write_text(tex, encoding="utf-8")


def misspec_table() -> None:
    df = pd.read_csv(DATA / "experiment_b_misspecification.csv")
    d = df[(df.experiment == "B") & (df.model == "diagonal")]
    out = d.groupby("regime")[["regret", "variance_ratio_vs_aware", "cvar99_ratio_vs_aware", "allocation_turnover"]].mean().reset_index()
    out["regime"] = out.regime.replace({"near_independent":"Near independent","clustered":"Clustered","global_clustered":"Global + clustered"})
    out.columns = ["Dependence", "Utility regret", "Variance ratio", "99\\% CVaR ratio", "Turnover"]
    out.to_csv(OUT / "misspecification_summary.csv", index=False)
    (OUT / "misspecification_summary.tex").write_text(out.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.4f}", column_format="lrrrr"), encoding="utf-8")


def competition_table() -> None:
    df = pd.read_csv(DATA / "experiment_d_competition.csv")
    d = df[(df.b_scale == 1.5) & (df.n.isin([1, 4, 12]))].groupby("n").mean(numeric_only=True).reset_index()
    out = pd.DataFrame({
        "Allocators": d.n.astype(int),
        "NE exposure": d.ne_total_gross_exposure,
        "Planner exposure": d.planner_total_gross_exposure,
        "Welfare gap": d.welfare_gap,
        "NE 99\\% CVaR": d.ne_aggregate_cvar99,
        "Planner 99\\% CVaR": d.planner_aggregate_cvar99,
    })
    out.to_csv(OUT / "competition_summary.csv", index=False)
    (OUT / "competition_summary.tex").write_text(out.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.3f}", column_format="rrrrrr"), encoding="utf-8")


def solver_table() -> None:
    df = pd.read_csv(DATA / "experiment_e_solvers.csv")
    out = df.groupby(["condition", "solver"]).agg(
        convergence_rate=("converged", "mean"),
        median_runtime=("runtime", "median"),
        median_iterations=("iterations", "median"),
        max_exploitability=("exploitability", "max"),
        max_vi_residual=("vi_residual", "max"),
    ).reset_index()
    out["condition"] = out.condition.str.capitalize()
    out["solver"] = out.solver.replace({
        "br_cyclic":"Cyclic BR", "br_random":"Random BR", "br_reverse":"Reverse BR",
        "pg_tuned":"Adaptive PG", "pg_conservative":"Fixed-step PG",
        "pg_sqrt_decay":r"Square-root-decay PG", "natural_residual":"Residual minimizer",
    })
    out.columns = ["Condition", "Solver", "Convergence rate", "Median runtime", "Median iterations", "Max exploitability", "Max VI residual"]
    out.to_csv(OUT / "solver_summary.csv", index=False)
    fmt = out.copy()
    fmt["Convergence rate"] = fmt["Convergence rate"].map(lambda z: f"{100*z:.0f}\\%")
    for col in ["Median runtime", "Median iterations"]:
        fmt[col] = fmt[col].map(lambda z: f"{z:.2f}")
    for col in ["Max exploitability", "Max VI residual"]:
        fmt[col] = fmt[col].map(lambda z: f"{z:.1e}")
    (OUT / "solver_summary.tex").write_text(fmt.to_latex(index=False, escape=False, column_format="llrrrrr"), encoding="utf-8")


def risk_table() -> None:
    df = pd.read_csv(DATA / "experiment_f_risk_robustness.csv")
    theta = df[(df.family == "theta") & (df.method == "entropic")]
    t = theta.groupby("risk_parameter")[["distance_to_matched_other", "gross_exposure", "cvar99"]].mean().reset_index()
    t.columns = ["Risk sensitivity", "$\\ell_1$ distance to MV", "Entropic exposure", "Entropic 99\\% CVaR"]
    t.to_csv(OUT / "entropic_summary.csv", index=False)
    (OUT / "entropic_summary.tex").write_text(t.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.3f}", column_format="rrrr"), encoding="utf-8")

    zeta = df[df.family == "zeta"].groupby("risk_parameter")[["gross_exposure", "cvar95", "cvar99"]].mean().reset_index()
    zeta.columns = ["$\\zeta$", "Gross exposure", "95\\% CVaR", "99\\% CVaR"]
    zeta.to_csv(OUT / "cvar_summary.csv", index=False)
    (OUT / "cvar_summary.tex").write_text(zeta.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.3f}", column_format="rrrr"), encoding="utf-8")


def validation_table() -> None:
    df = pd.read_csv(DATA / "validation_report.csv")
    df.to_csv(OUT / "validation_report.csv", index=False)
    names = {
        "factor_covariance_max_abs_error": "Factor covariance",
        "loss_mean_max_abs_error": "Loss mean",
        "loss_variance_max_abs_error": "Loss variance",
        "equilibrium_exploitability": "Equilibrium exploitability",
        "equilibrium_vi_residual": "Equilibrium VI residual",
        "cross_solver_l2_spread": r"Cross-solver $\ell_2$ spread",
    }

    def sci_tex(value: float) -> str:
        mantissa, exponent = f"{value:.2e}".split("e")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"

    lines = [
        r"\begin{tabular}{lrrc}",
        r"\toprule",
        r"Test & Observed error & Tolerance & Passed \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        label = names.get(row["test"], str(row["test"]).replace("_", r"\_"))
        passed = "yes" if bool(row["passed"]) else "no"
        lines.append(
            f"{label} & {sci_tex(float(row['value']))} & "
            f"{sci_tex(float(row['tolerance']))} & {passed} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "validation_report.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    architecture_table()
    misspec_table()
    competition_table()
    solver_table()
    risk_table()
    validation_table()
    print(f"Wrote tables to {OUT}")


if __name__ == "__main__":
    main()
