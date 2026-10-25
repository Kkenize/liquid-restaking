from __future__ import annotations

"""Check that a validation or full reproduction run produced credible outputs."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
TABLES = ROOT / "results" / "tables"
OUTPUT = ROOT / "results" / "reproduction_check.json"


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def read_csv(checks: list[dict[str, Any]], name: str) -> pd.DataFrame | None:
    path = PROCESSED / name
    if not path.exists():
        add(checks, f"file:{name}", False, "missing")
        return None
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - diagnostic path
        add(checks, f"file:{name}", False, f"unreadable: {exc}")
        return None
    add(checks, f"file:{name}", True, {"rows": len(frame), "columns": len(frame.columns)})
    return frame


def finite(frame: pd.DataFrame, columns: list[str]) -> bool:
    values = frame[columns].to_numpy(dtype=float)
    return bool(np.all(np.isfinite(values)))


def validation_checks(checks: list[dict[str, Any]]) -> None:
    validation = read_csv(checks, "validation_report.csv")
    if validation is None:
        return
    expected = 6
    add(checks, "validation:row_count", len(validation) == expected, {"observed": len(validation), "expected": expected})
    passed = validation["passed"].astype(str).str.lower().isin(["true", "1", "yes"]).all()
    add(checks, "validation:all_pass", bool(passed), validation.to_dict(orient="records"))


def full_checks(checks: list[dict[str, Any]]) -> None:
    expected_rows = {
        "experiment_a_architecture.csv": 180,
        "experiment_a_allocations.csv": 8640,
        "experiment_b_misspecification.csv": 124,
        "experiment_c_topology.csv": 400,
        "experiment_c_allocations.csv": 6400,
        "experiment_d_competition.csv": 75,
        "experiment_e_solvers.csv": 156,
        "experiment_f_risk_robustness.csv": 95,
        "experiment_f_allocations.csv": 1140,
    }
    frames: dict[str, pd.DataFrame] = {}
    for filename, expected in expected_rows.items():
        frame = read_csv(checks, filename)
        if frame is None:
            continue
        frames[filename] = frame
        add(
            checks,
            f"rows:{filename}",
            len(frame) == expected,
            {"observed": len(frame), "expected": expected},
        )

    a = frames.get("experiment_a_architecture.csv")
    if a is not None:
        add(checks, "A:all_converged", bool(a["converged"].all()), int((~a["converged"]).sum()))
        add(checks, "A:finite_core", finite(a, ["total_reward", "aggregate_expected_loss", "aggregate_std", "aggregate_cvar99", "exploitability", "vi_residual"]), "core columns")
        add(checks, "A:max_exploitability", float(a["exploitability"].max()) < 1e-6, float(a["exploitability"].max()))
        add(checks, "A:max_vi_residual", float(a["vi_residual"].max()) < 5e-5, float(a["vi_residual"].max()))

    b = frames.get("experiment_b_misspecification.csv")
    if b is not None:
        main = b[b["experiment"] == "B"]
        add(checks, "B:finite_core", finite(main, ["regret", "variance_ratio_vs_aware", "cvar99_ratio_vs_aware", "allocation_turnover"]), "main rows")
        validation = b[b["experiment"] == "B_validation"]
        if not validation.empty:
            error = np.max(np.abs(validation["observed_multiplier"] - validation["theoretical_multiplier"]))
            add(checks, "B:multiplier_identity", float(error) < 1e-8, float(error))

    c = frames.get("experiment_c_topology.csv")
    if c is not None:
        add(checks, "C:finite_core", finite(c, ["utility_gap_to_aware", "gross_exposure", "cvar99"]), "core columns")

    d = frames.get("experiment_d_competition.csv")
    if d is not None:
        add(checks, "D:all_ne_converged", bool(d["ne_converged"].all()), int((~d["ne_converged"]).sum()))
        add(checks, "D:all_planner_converged", bool(d["planner_converged"].all()), int((~d["planner_converged"]).sum()))
        add(checks, "D:max_exploitability", float(d["exploitability"].max()) < 1e-6, float(d["exploitability"].max()))
        add(checks, "D:max_vi_residual", float(d["vi_residual"].max()) < 5e-5, float(d["vi_residual"].max()))

    e = frames.get("experiment_e_solvers.csv")
    if e is not None:
        required = ["br_cyclic", "br_reverse", "br_random", "pg_tuned"]
        rates = e.groupby("solver")["converged"].mean()
        for solver in required:
            rate = float(rates.get(solver, 0.0))
            add(checks, f"E:{solver}_convergence", rate == 1.0, rate)
        certified = e[(e["exploitability"] < 1e-7) & (e["vi_residual"] < 1e-5)]
        add(checks, "E:certified_solutions_present", len(certified) > 0, len(certified))

    f = frames.get("experiment_f_risk_robustness.csv")
    if f is not None:
        add(checks, "F:all_optimizers_converged", bool(f["converged"].all()), int((~f["converged"]).sum()))
        add(checks, "F:finite_core", finite(f, ["gross_exposure", "cvar95", "cvar99"]), "core columns")

    figure_stems = [
        "architecture_frontier",
        "architecture_exposure_breadth",
        "independence_misspecification",
        "independence_multiplier_validation",
        "factor_topology_heuristics",
        "factor_topology_allocations",
        "competition_equilibrium_planner",
        "competition_welfare_gap",
        "solver_scaling",
        "solver_convergence_trace",
        "risk_measure_entropic",
        "risk_measure_cvar",
        "www_empirical_summary",
    ]
    for stem in figure_stems:
        for suffix in ["pdf", "png"]:
            path = FIGURES / f"{stem}.{suffix}"
            add(checks, f"figure:{path.name}", path.exists() and path.stat().st_size > 1000, path.stat().st_size if path.exists() else "missing")
    contact = FIGURES / "contact_sheet.png"
    add(checks, "figure:contact_sheet.png", contact.exists() and contact.stat().st_size > 1000, contact.stat().st_size if contact.exists() else "missing")

    table_stems = [
        "architecture_endpoints",
        "misspecification_summary",
        "competition_summary",
        "solver_summary",
        "entropic_summary",
        "cvar_summary",
        "validation_report",
    ]
    for stem in table_stems:
        for suffix in ["csv", "tex"]:
            path = TABLES / f"{stem}.{suffix}"
            add(checks, f"table:{path.name}", path.exists() and path.stat().st_size > 20, path.stat().st_size if path.exists() else "missing")

    for rel in ["RESULTS_SUMMARY.md", "results/run_metadata.json"]:
        path = ROOT / rel
        add(checks, f"artifact:{rel}", path.exists() and path.stat().st_size > 100, path.stat().st_size if path.exists() else "missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "validation"], default="full")
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    validation_checks(checks)
    if args.mode == "full":
        full_checks(checks)

    overall = all(check["passed"] for check in checks)
    payload = {"mode": args.mode, "passed": overall, "checks": checks}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    failures = [check for check in checks if not check["passed"]]
    print(f"Reproduction checks: {len(checks) - len(failures)}/{len(checks)} passed")
    for failure in failures:
        print(f"FAILED: {failure['name']}: {failure['detail']}")
    print(f"Wrote {OUTPUT}")
    if not overall:
        sys.exit(1)


if __name__ == "__main__":
    main()
