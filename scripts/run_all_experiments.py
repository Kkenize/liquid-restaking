from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restaking_risk.environment import Environment
from restaking_risk.experiment_utils import (
    clone_with_players,
    ensure_dirs,
    evaluate_single_allocation,
    low_rank_plus_diagonal_covariance,
    optimize_scale_for_weights,
    save_json,
    write_records,
)
from restaking_risk.generators import make_environment, make_homogeneous_equicorrelated
from restaking_risk.metrics import (
    allocation_turnover,
    covariance_decomposition,
    exploitability,
    portfolio_statistics,
    summarize_profile,
    var_cvar,
    vi_residual,
)
from restaking_risk.model import gross_claim_samples, social_welfare, total_reward, utility_vector
from restaking_risk.risk_objectives import (
    solve_single_cvar_premium,
    solve_single_entropic,
    solve_single_mean_variance,
)
from restaking_risk.solvers import (
    cyclic_best_response,
    feasible_start,
    natural_residual_solver,
    projected_pseudogradient,
    solve_planner,
)

DIRS = ensure_dirs(ROOT)


def _solve_equilibrium(env: Environment, V: np.ndarray, x0: np.ndarray | None = None):
    # The simultaneous projected pseudo-gradient method is the primary equilibrium solver.
    pg = projected_pseudogradient(
        env, V=V, x0=x0, step_size=0.15, adaptive=True,
        schedule="constant", tol=2e-8, maxiter=30_000, store_every=1,
    )
    if pg.converged and vi_residual(env, pg.x, V) < 2e-5:
        return pg, "pg_adaptive"

    # A strict-concavity best-response solver remains as a fallback and an
    # independent equilibrium certificate.
    br = cyclic_best_response(env, V=V, x0=pg.x, tol=2e-9, max_sweeps=1200)
    return br, "cyclic_br_fallback"


def experiment_a() -> pd.DataFrame:
    records: list[dict] = []
    allocation_records: list[dict] = []
    regimes = ["near_independent", "clustered", "global_clustered"]
    seeds = list(range(5))
    for regime in regimes:
        for seed in seeds:
            base = make_environment(n=4, m=12, kappa=1, regime=regime, seed=100 + seed,
                                    heterogeneous=True, eta_scale=1.25)
            F = base.sample_slashes(80_000, seed=10_000 + seed)
            warm = None
            for kappa in range(1, base.m + 1):
                env = base.with_updates(kappa=np.full(base.n, float(kappa)))
                C, d, V = env.risk_matrices()
                result, solver = _solve_equilibrium(env, V, warm)
                warm = result.x
                row = summarize_profile(env, result.x, F=F, V=V, cvar_level=0.99)
                row.update({
                    "experiment": "A",
                    "regime": regime,
                    "seed": seed,
                    "kappa": kappa,
                    "solver": solver,
                    "converged": result.converged,
                    "iterations": result.iterations,
                    "runtime": result.runtime,
                    "exploitability": exploitability(env, result.x, V),
                    "vi_residual": vi_residual(env, result.x, V),
                })
                records.append(row)
                stats = portfolio_statistics(result.x, env.W)
                for i in range(env.n):
                    for j in range(env.m):
                        allocation_records.append({
                            "regime": regime, "seed": seed, "kappa": kappa,
                            "allocator": i, "position": j, "exposure": result.x[i, j],
                            "marginal_variance_contribution": result.x[i, j] * (V @ result.x[i])[j],
                            "reuse_degree": stats["reuse_degree"][i],
                            "effective_breadth": stats["effective_breadth"][i],
                        })
            print(f"A complete: {regime}, seed {seed}", flush=True)
    df = write_records(records, DIRS["processed"] / "experiment_a_architecture.csv")
    write_records(allocation_records, DIRS["processed"] / "experiment_a_allocations.csv")
    return df


def experiment_b() -> pd.DataFrame:
    records: list[dict] = []
    regimes = ["near_independent", "clustered", "global_clustered"]
    sample_sizes = [250, 1_000, 5_000]
    for regime in regimes:
        for seed in range(10):
            base = make_environment(n=1, m=16, kappa=4, regime=regime, seed=200 + seed,
                                    heterogeneous=True, eta_scale=1.6)
            C, d, V = base.risk_matrices()
            Vdiag = np.diag(np.diag(V))
            F_eval = base.sample_slashes(120_000, seed=20_000 + seed)
            aware = solve_single_mean_variance(base, V=V)
            diag = solve_single_mean_variance(base, V=Vdiag)
            aware_eval = evaluate_single_allocation(base, aware.x, F_eval, V)
            diag_eval = evaluate_single_allocation(base, diag.x, F_eval, V)
            within, across, total = covariance_decomposition(diag.x, V, base.operator_cluster)
            base_row = {
                "experiment": "B",
                "regime": regime,
                "seed": seed,
                "kappa": base.kappa[0],
                "model": "diagonal",
                "regret": aware_eval["mean_variance_utility"] - diag_eval["mean_variance_utility"],
                "variance_ratio_vs_aware": diag_eval["variance"] / max(aware_eval["variance"], 1e-15),
                "cvar99_ratio_vs_aware": diag_eval["cvar99"] / max(aware_eval["cvar99"], 1e-15),
                "allocation_turnover": allocation_turnover(aware.x, diag.x),
                "within_cluster_variance": within,
                "across_cluster_variance": across,
                "true_variance": total,
                **{f"aware_{k}": v for k, v in aware_eval.items()},
                **{f"candidate_{k}": v for k, v in diag_eval.items()},
            }
            records.append(base_row)
            F_train_full = base.sample_slashes(max(sample_sizes), seed=30_000 + seed)
            for n_sample in sample_sizes:
                C_hat = low_rank_plus_diagonal_covariance(F_train_full[:n_sample], rank=3)
                V_hat = base.alpha[:, None] * C_hat * base.alpha[None, :]
                est = solve_single_mean_variance(base, V=V_hat)
                est_eval = evaluate_single_allocation(base, est.x, F_eval, V)
                within, across, total = covariance_decomposition(est.x, V, base.operator_cluster)
                records.append({
                    "experiment": "B",
                    "regime": regime,
                    "seed": seed,
                    "kappa": base.kappa[0],
                    "model": f"factor_rank3_n{n_sample}",
                    "sample_size": n_sample,
                    "regret": aware_eval["mean_variance_utility"] - est_eval["mean_variance_utility"],
                    "variance_ratio_vs_aware": est_eval["variance"] / max(aware_eval["variance"], 1e-15),
                    "cvar99_ratio_vs_aware": est_eval["cvar99"] / max(aware_eval["cvar99"], 1e-15),
                    "allocation_turnover": allocation_turnover(aware.x, est.x),
                    "within_cluster_variance": within,
                    "across_cluster_variance": across,
                    "true_variance": total,
                    **{f"aware_{k}": v for k, v in aware_eval.items()},
                    **{f"candidate_{k}": v for k, v in est_eval.items()},
                })
            print(f"B complete: {regime}, seed {seed}", flush=True)

    # Exact homogeneous benchmark used as a validation row in the paper.
    for m in [4, 8, 12, 16]:
        env = make_homogeneous_equicorrelated(n=1, m=m, kappa=1, p=0.06, alpha=0.6,
                                              latent_loading_sq=0.35, seed=m)
        C, _, V = env.risk_matrices()
        rho = C[0, 1] / C[0, 0]
        x = np.full(m, env.W[0] / m)
        true_var = float(x @ V @ x)
        diag_var = float(x @ np.diag(np.diag(V)) @ x)
        records.append({
            "experiment": "B_validation",
            "regime": "homogeneous",
            "seed": m,
            "kappa": 1,
            "model": "equal_weight_identity",
            "m": m,
            "rho": rho,
            "observed_multiplier": true_var / diag_var,
            "theoretical_multiplier": 1 + (m - 1) * rho,
        })
    return write_records(records, DIRS["processed"] / "experiment_b_misspecification.csv")


def _heuristic_weights(env: Environment, method: str) -> np.ndarray:
    if method == "equal_weight":
        return np.ones(env.m)
    if method == "reward_yield":
        return env.R / env.b
    if method == "lowest_p":
        w = np.zeros(env.m)
        idx = np.argsort(env.p)[: min(4, env.m)]
        w[idx] = 1.0
        return w
    if method == "one_per_operator_cluster":
        w = np.zeros(env.m)
        clusters = sorted(set(env.operator_cluster))
        score = env.R / env.b - env.alpha * env.p
        for c in clusters:
            idx = [j for j, cj in enumerate(env.operator_cluster) if cj == c]
            winner = idx[int(np.argmax(score[idx]))]
            w[winner] = 1.0
        return w
    raise ValueError(method)


def experiment_c() -> pd.DataFrame:
    records: list[dict] = []
    allocations: list[dict] = []
    global_shares = [0.0, 0.10, 0.25, 0.40, 0.55]
    methods = ["aware", "reward_yield", "equal_weight", "lowest_p", "one_per_operator_cluster"]
    for kappa in [1.0, 4.0]:
        for gshare in global_shares:
            for seed in range(8):
                env = make_environment(n=1, m=16, kappa=kappa, regime="topology", global_share=gshare,
                                       seed=300 + seed, heterogeneous=True, eta_scale=1.8)
                C, _, V = env.risk_matrices()
                F_eval = env.sample_slashes(100_000, seed=40_000 + seed)
                aware = solve_single_mean_variance(env, V=V).x
                for method in methods:
                    if method == "aware":
                        x = aware
                    else:
                        x = optimize_scale_for_weights(env, _heuristic_weights(env, method), V)
                    ev = evaluate_single_allocation(env, x, F_eval, V)
                    records.append({
                        "experiment": "C",
                        "kappa": kappa,
                        "global_share": gshare,
                        "seed": seed,
                        "method": method,
                        "utility_gap_to_aware": evaluate_single_allocation(env, aware, F_eval, V)["mean_variance_utility"] - ev["mean_variance_utility"],
                        **ev,
                    })
                    for j in range(env.m):
                        allocations.append({
                            "kappa": kappa, "global_share": gshare, "seed": seed,
                            "method": method, "position": j, "exposure": x[j],
                            "operator_cluster": env.operator_cluster[j],
                            "client_cluster": env.client_cluster[j],
                            "cloud_cluster": env.cloud_cluster[j],
                            "p": env.p[j], "reward_yield": env.R[j] / env.b[j],
                        })
            print(f"C complete: kappa {kappa}, global share {gshare}", flush=True)
    df = write_records(records, DIRS["processed"] / "experiment_c_topology.csv")
    write_records(allocations, DIRS["processed"] / "experiment_c_allocations.csv")
    return df


def experiment_d() -> pd.DataFrame:
    records: list[dict] = []
    for seed in range(5):
        base = make_environment(n=1, m=8, kappa=2, regime="clustered", seed=400 + seed,
                                heterogeneous=True, eta_scale=1.0)
        F = base.sample_slashes(80_000, seed=50_000 + seed)
        for b_scale in [0.5, 1.5, 5.0]:
            scaled_base = base.with_updates(b=base.b * b_scale)
            for n in [1, 2, 4, 8, 12]:
                env = clone_with_players(scaled_base, n=n, total_capital=4.0, kappa=2.0, eta=1.4)
                _, d, V = env.risk_matrices()
                ne, solver = _solve_equilibrium(env, V)
                planner = solve_planner(env, V=V, x0=ne.x)
                ne_metrics = summarize_profile(env, ne.x, F=F, V=V)
                planner_metrics = summarize_profile(env, planner.x, F=F, V=V)
                risk_cost_ne = float(sum(0.5 * env.eta[i] * ne.x[i] @ V @ ne.x[i] for i in range(n)))
                dissipation = (ne_metrics["aggregate_expected_loss"] + risk_cost_ne) / max(ne_metrics["total_reward"], 1e-15)
                records.append({
                    "experiment": "D",
                    "seed": seed,
                    "n": n,
                    "b_scale": b_scale,
                    "solver": solver,
                    "ne_converged": ne.converged,
                    "planner_converged": planner.converged,
                    "ne_iterations": ne.iterations,
                    "planner_iterations": planner.iterations,
                    "ne_runtime": ne.runtime,
                    "planner_runtime": planner.runtime,
                    "exploitability": exploitability(env, ne.x, V),
                    "vi_residual": vi_residual(env, ne.x, V),
                    "welfare_gap": planner_metrics["social_welfare"] - ne_metrics["social_welfare"],
                    "reward_dissipation_fraction": dissipation,
                    **{f"ne_{k}": v for k, v in ne_metrics.items()},
                    **{f"planner_{k}": v for k, v in planner_metrics.items()},
                })
            print(f"D complete: seed {seed}, b_scale {b_scale}", flush=True)
    return write_records(records, DIRS["processed"] / "experiment_d_competition.csv")


def _solver_env(condition: str, n: int, m: int, seed: int) -> Environment:
    if condition == "easy":
        return make_environment(n=n, m=m, kappa=1.0, regime="near_independent", seed=500 + seed,
                                heterogeneous=True, b_scale=3.0, eta_scale=1.8)
    if condition == "clustered":
        return make_environment(n=n, m=m, kappa=min(3.0, m), regime="clustered", seed=500 + seed,
                                heterogeneous=True, b_scale=1.0, eta_scale=1.0)
    if condition == "stress":
        return make_environment(n=n, m=m, kappa=float(m), regime="ill_conditioned", seed=500 + seed,
                                heterogeneous=True, reward_scale=1.4, b_scale=0.08, eta_scale=0.12)
    raise ValueError(condition)


def experiment_e() -> pd.DataFrame:
    records: list[dict] = []
    history_records: list[dict] = []
    sizes = [(2, 4), (4, 8), (8, 12), (12, 16)]
    conditions = ["easy", "clustered", "stress"]
    for condition in conditions:
        for n, m in sizes:
            for seed in range(2):
                env = _solver_env(condition, n, m, seed)
                _, _, V = env.risk_matrices()
                starts = {
                    "equal": feasible_start(env, "equal", seed),
                    "zero": feasible_start(env, "zero", seed),
                    "random": feasible_start(env, "random", seed),
                }
                solver_specs = [
                    ("br_cyclic", lambda: cyclic_best_response(env, V=V, x0=starts["equal"], update_order="cyclic", tol=2e-8, max_sweeps=1000)),
                    ("br_reverse", lambda: cyclic_best_response(env, V=V, x0=starts["random"], update_order="reverse", tol=2e-8, max_sweeps=1000)),
                    ("br_random", lambda: cyclic_best_response(env, V=V, x0=starts["zero"], update_order="random", tol=2e-8, max_sweeps=1000, seed=seed)),
                    ("pg_conservative", lambda: projected_pseudogradient(env, V=V, x0=starts["random"], step_size=0.015, adaptive=False, schedule="constant", tol=2e-8, maxiter=30000)),
                    ("pg_tuned", lambda: projected_pseudogradient(env, V=V, x0=starts["equal"], step_size=0.15, adaptive=True, schedule="constant", tol=2e-8, maxiter=30000)),
                    ("pg_sqrt_decay", lambda: projected_pseudogradient(env, V=V, x0=starts["equal"], step_size=2.0, adaptive=False, schedule="sqrt_decay", tol=2e-8, maxiter=30000)),
                ]
                if n * m <= 32:
                    solver_specs.append(("natural_residual", lambda: natural_residual_solver(env, V=V, x0=starts["equal"], tau=0.1, maxiter=5000)))
                solutions = {}
                certified_solutions = []
                for name, fn in solver_specs:
                    result = fn()
                    exploit = exploitability(env, result.x, V)
                    residual = vi_residual(env, result.x, V)
                    if exploit < 1e-7 and residual < 1e-5:
                        certified_solutions.append(result.x)
                    records.append({
                        "experiment": "E",
                        "condition": condition,
                        "n": n,
                        "m": m,
                        "seed": seed,
                        "solver": name,
                        "converged": result.converged,
                        "iterations": result.iterations,
                        "runtime": result.runtime,
                        "exploitability": exploit,
                        "vi_residual": residual,
                        "final_step": result.history.get("step", [np.nan])[-1] if result.history.get("step") else np.nan,
                    })
                    solutions[name] = result.x
                    # Preserve representative histories for the paper convergence plot.
                    if condition == "stress" and n == 8 and m == 12 and seed == 0 and name in {"br_cyclic", "pg_conservative", "pg_tuned", "pg_sqrt_decay"}:
                        for t, step in enumerate(result.history.get("step", []), start=1):
                            history_records.append({"solver": name, "iteration": t, "step_norm": step})
                # Cross-solver spread among profiles certified by both
                # exploitability and the natural VI residual.
                spread = max(
                    (np.linalg.norm(a - b) for a in certified_solutions for b in certified_solutions),
                    default=np.nan,
                )
                for rec in records[-len(solver_specs):]:
                    rec["cross_solver_spread"] = spread
                print(f"E complete: {condition}, n={n}, m={m}, seed={seed}", flush=True)
    df = write_records(records, DIRS["processed"] / "experiment_e_solvers.csv")
    write_records(history_records, DIRS["processed"] / "experiment_e_histories.csv")
    return df


def _entropic_value(env: Environment, x: np.ndarray, F: np.ndarray, theta: float) -> float:
    monetary = F * env.alpha[None, :]
    sample_mean = monetary.mean(axis=0)
    centered_losses = (monetary - sample_mean[None, :]) @ x
    risk = (env.alpha * env.p) @ x + (logsumexp(theta * centered_losses) - np.log(len(centered_losses))) / theta
    reward = np.sum(env.R * x / (env.b + x))
    return float(reward - risk)


def _cvar_value(env: Environment, x: np.ndarray, F: np.ndarray, zeta: float, level: float = 0.95) -> float:
    losses = (F * env.alpha[None, :]) @ x
    mean = losses.mean()
    _, cvar = var_cvar(losses, level)
    reward = np.sum(env.R * x / (env.b + x))
    return float(reward - (mean + zeta * (cvar - mean)))


def experiment_f() -> pd.DataFrame:
    records: list[dict] = []
    allocation_records: list[dict] = []
    theta_grid = [0.02, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0]
    zeta_grid = [0.05, 0.10, 0.25, 0.50, 1.0]
    for seed in range(5):
        raw = make_environment(n=1, m=12, kappa=4.0, regime="global_clustered", seed=600 + seed,
                               heterogeneous=True, reward_scale=1.80, eta_scale=1.0)
        # Increase event frequency and severity so tail-objective differences are visible.
        env = raw.with_updates(
            p=np.clip(raw.p * 1.6, 0.045, 0.11),
            alpha=np.clip(raw.alpha * 1.15, 0.45, 0.85),
            eta=np.array([1.0]),
        )
        _, _, V = env.risk_matrices()
        F_train = env.sample_slashes(30_000, seed=60_000 + seed)
        F_eval = env.sample_slashes(180_000, seed=70_000 + seed)
        for theta in theta_grid:
            mv = solve_single_mean_variance(env, V=V, eta=theta)
            ent = solve_single_entropic(env, F_train, theta=theta)
            mv_eval = evaluate_single_allocation(env, mv.x, F_eval, V, eta=theta)
            ent_eval = evaluate_single_allocation(env, ent.x, F_eval, V, eta=theta)
            for method, result, ev in [("mean_variance", mv, mv_eval), ("entropic", ent, ent_eval)]:
                records.append({
                    "experiment": "F",
                    "seed": seed,
                    "family": "theta",
                    "risk_parameter": theta,
                    "method": method,
                    "converged": result.converged,
                    "iterations": result.iterations,
                    "runtime": result.runtime,
                    "entropic_value_eval": _entropic_value(env, result.x, F_eval, theta),
                    "distance_to_matched_other": float(np.linalg.norm(mv.x - ent.x, ord=1)),
                    **ev,
                })
                for j in range(env.m):
                    allocation_records.append({
                        "seed": seed, "family": "theta", "risk_parameter": theta,
                        "method": method, "position": j, "exposure": result.x[j],
                    })
        for zeta in zeta_grid:
            cvar = solve_single_cvar_premium(env, F_train, zeta=zeta, level=0.95)
            ev = evaluate_single_allocation(env, cvar.x, F_eval, V, eta=1.0)
            records.append({
                "experiment": "F",
                "seed": seed,
                "family": "zeta",
                "risk_parameter": zeta,
                "method": "cvar_premium",
                "converged": cvar.converged,
                "iterations": cvar.iterations,
                "runtime": cvar.runtime,
                "cvar_objective_eval": _cvar_value(env, cvar.x, F_eval, zeta),
                **ev,
            })
            for j in range(env.m):
                allocation_records.append({
                    "seed": seed, "family": "zeta", "risk_parameter": zeta,
                    "method": "cvar_premium", "position": j, "exposure": cvar.x[j],
                })
        print(f"F complete: seed {seed}", flush=True)
    df = write_records(records, DIRS["processed"] / "experiment_f_risk_robustness.csv")
    write_records(allocation_records, DIRS["processed"] / "experiment_f_allocations.csv")
    return df


def validation_report() -> pd.DataFrame:
    records: list[dict] = []
    # Moment and factor tests.
    env = make_environment(n=2, m=10, kappa=2, regime="clustered", seed=777)
    C, d, V = env.risk_matrices()
    F = env.sample_slashes(400_000, seed=778)
    C_hat = np.cov(F, rowvar=False, ddof=0)
    records.append({"test": "factor_covariance_max_abs_error", "value": float(np.max(np.abs(C_hat - C))), "tolerance": 0.0015, "passed": float(np.max(np.abs(C_hat - C))) < 0.0015})
    x = feasible_start(env, "random", 4)
    indiv, agg = gross_claim_samples(env, x, F)
    mean_err = max(abs(indiv[:, i].mean() - d @ x[i]) for i in range(env.n))
    var_err = max(abs(indiv[:, i].var() - x[i] @ V @ x[i]) for i in range(env.n))
    records.append({"test": "loss_mean_max_abs_error", "value": float(mean_err), "tolerance": 0.002, "passed": mean_err < 0.002})
    records.append({"test": "loss_variance_max_abs_error", "value": float(var_err), "tolerance": 0.002, "passed": var_err < 0.002})
    # Cross-solver and equilibrium certificate.
    small = make_environment(n=3, m=6, kappa=2, regime="clustered", seed=779)
    _, _, Vs = small.risk_matrices()
    br = cyclic_best_response(small, V=Vs, tol=1e-10, max_sweeps=1000)
    pg = projected_pseudogradient(small, V=Vs, step_size=0.12, adaptive=True, tol=1e-10, maxiter=30000)
    exploit = exploitability(small, br.x, Vs)
    vi = vi_residual(small, br.x, Vs)
    spread = np.linalg.norm(br.x - pg.x)
    records.append({"test": "equilibrium_exploitability", "value": exploit, "tolerance": 1e-7, "passed": exploit < 1e-7})
    records.append({"test": "equilibrium_vi_residual", "value": vi, "tolerance": 2e-6, "passed": vi < 2e-6})
    records.append({"test": "cross_solver_l2_spread", "value": float(spread), "tolerance": 5e-4, "passed": spread < 5e-4})
    return write_records(records, DIRS["processed"] / "validation_report.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=list("ABCDEF") + ["validation", "all"], default="all")
    args = parser.parse_args()
    manifest = {
        "description": "Fixed synthetic stress-test manifest for all paper experiments.",
        "experiments": {
            "A": {"seeds": 5, "n": 4, "m": 12, "kappa": "1..12", "mc_samples": 80000},
            "B": {"seeds": 10, "m": 16, "kappa": 4, "mc_samples": 120000, "fit_samples": [250, 1000, 5000]},
            "C": {"seeds": 8, "m": 16, "kappa": [1, 4], "global_share": [0, .1, .25, .4, .55]},
            "D": {"seeds": 5, "n": [1, 2, 4, 8, 12], "b_scale": [.5, 1.5, 5], "total_capital": 4},
            "E": {"seeds": 2, "sizes": [[2,4],[4,8],[8,12],[12,16]], "conditions": ["easy","clustered","stress"]},
            "F": {"seeds": 5, "m": 12, "kappa": 4, "train_samples": 30000, "eval_samples": 180000},
        },
    }
    save_json(manifest, DIRS["configs"] / "experiment_manifest.json")
    fns = {"A": experiment_a, "B": experiment_b, "C": experiment_c, "D": experiment_d, "E": experiment_e, "F": experiment_f, "validation": validation_report}
    targets = list("ABCDEF") + ["validation"] if args.experiment == "all" else [args.experiment]
    for target in targets:
        t0 = time.perf_counter()
        print(f"Starting experiment {target}", flush=True)
        fns[target]()
        print(f"Finished experiment {target} in {time.perf_counter()-t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
