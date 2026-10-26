# Liquid Restaking Correlated-Risk Experiments

This repository contains the computational experiments for the paper **Collateral Architecture and Correlated Slashing in Liquid Restaking**.

The implementation provides a projected pseudo-gradient equilibrium framework for studying collateral reuse, correlated slashing risk, and strategic allocation in liquid restaking systems. It incorporates capped-simplex projection, analytical proportional-reward gradients, projected equilibrium algorithms, correlated Bernoulli slashing models, and alternative risk objectives.

The repository includes the source code, experimental configurations, validation procedures, and scripts required to reproduce the computational results presented in the paper.

## Model implemented

Allocator `i` chooses position exposure `x[i,j]` subject to:

```text
0 <= x[i,j] <= W[i]
sum_j x[i,j] <= kappa[i] * W[i]
```

Each allocator receives a proportional share of fixed reward pools.

Slash events are constructed from thresholded Gaussian factors, producing an attainable joint Bernoulli distribution. The model computes:

```text
C = Cov(F)
V = diag(alpha) C diag(alpha)
```

The mean-variance payoff is:

```text
reward - expected gross slash claims - (eta_i / 2) x_i^T V x_i
```

The implementation supports an optional minimum aggregate exposure through `minimum_exposure_multiplier`. The paper experiments allow optional restaking with a lower bound of zero.

## Requirements and installation

Python 3.11 or later is required.

The repository provides `requirements-lock.txt` for installing the package versions used in the research environment.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

## Reproduce all experiments

To reproduce the complete computational study from a clean checkout, run:

```bash
python scripts/reproduce_all.py
```

The reproduction workflow:

1. Cleans previously generated outputs.
2. Runs the test suite and numerical validation.
3. Executes experiments A–F using fixed experimental configurations.
4. Generates the experimental CSV datasets.
5. Reconstructs the publication figures and summary tables.
6. Produces numerical summaries and environment metadata.
7. Checks the expected outputs.
8. Generates a SHA-256 reproducibility manifest.

**Pre-generated CSV files, figures, and tables are not required.** They are produced by the reproduction workflow.

For a quicker mathematical and software validation without running the complete experimental suite:

```bash
python scripts/reproduce_all.py --mode validation
```

Detailed instructions are available in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Run individual components

The experiments and reporting tools can also be executed separately:

```bash
python -m pytest
python scripts/run_all_experiments.py --experiment validation
python scripts/run_all_experiments.py --experiment A
python scripts/run_all_experiments.py --experiment all
python scripts/make_figures.py
python scripts/make_www_summary_figure.py
python scripts/make_tables.py
python scripts/summarize_results.py
python scripts/write_run_metadata.py
python scripts/check_outputs.py
python scripts/write_manifest.py
```

The figure-generation and reporting scripts require the relevant experimental CSV files to have been generated first.

Equivalent Makefile targets include:

```bash
make test
make validation
make experiments
make figures
make tables
make reproduce
```

## Experiment families

The repository implements six families of controlled synthetic experiments.

**A — Collateral architecture frontier**

Vary the collateral reuse cap from unique stake to broad reuse under different dependence structures. Evaluate exposure, diversification, rewards, expected slashing claims, and tail risk.

**B — Dependence misspecification**

Compare covariance-aware allocations against allocations constructed under independence assumptions or estimated low-rank covariance models.

**C — Factor topology and allocation heuristics**

Vary dependence arising from global, operator, software-client, and cloud-provider factors. Compare covariance-aware allocations with transparent allocation heuristics.

**D — Competition versus centralized planning**

Compare equilibrium allocations with centralized planner solutions while varying the number of competing allocators and reward-offset parameters.

**E — Solver behavior**

Compare best-response methods, fixed-step and adaptive projected pseudo-gradient methods, square-root-decay step schedules, and residual minimization across different problem sizes and conditions.

**F — Risk-objective robustness**

Compare allocations under mean-variance, entropic-risk, and CVaR-premium objectives.

These experiments are **synthetic stress tests**, not estimates of actual protocol returns or real-world slashing frequencies.

The fixed experimental design is recorded in `configs/experiment_manifest.json` and described in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Generated outputs

After a full reproduction run, the repository produces:

```text
data/processed/
    Experimental datasets and allocation CSV files

figures/
    Publication figures in PDF and PNG
    WWW summary figure
    Figure contact sheet

results/tables/
    Summary tables in CSV and LaTeX

results/logs/
    Reproduction logs

results/run_metadata.json
    Software and platform information

results/reproduction_check.json
    Automated output verification results

RESULTS_SUMMARY.md
    Numerical summary generated from experimental data

REPRODUCIBILITY_MANIFEST.txt
    SHA-256 checksums for the repository snapshot
```

These files are generated artifacts rather than prerequisites for running the experiments.

Publication figures are constructed from saved experimental CSV files rather than directly from in-memory solver results.

See [`docs/FIGURE_INDEX.md`](docs/FIGURE_INDEX.md) for the source data associated with each figure and [`docs/PAPER_OUTPUTS.md`](docs/PAPER_OUTPUTS.md) for the correspondence between repository outputs and the manuscripts.

## Equilibrium certification

A solver's success flag is not treated as sufficient evidence of equilibrium.

Reported multiplayer profiles are evaluated using:

- Best-response exploitability.
- Projected variational-inequality residual.

The simultaneous projected method supports three update approaches:

- `schedule="constant"`: fixed-step updates corresponding to the paper's theoretical analysis.
- `adaptive=True`: a practical projected-residual line search.
- `schedule="sqrt_decay"`: a square-root-decay step schedule.

These diagnostics assess numerical equilibrium quality rather than establishing exact equilibrium solely from solver convergence.

## Repository structure

```text
.
├── README.md
├── Makefile
├── pyproject.toml
├── requirements.txt
├── requirements-lock.txt
│
├── configs/
│   └── experiment_manifest.json
│
├── src/
│   └── restaking_risk/
│       ├── __init__.py
│       ├── environment.py
│       ├── projection.py
│       ├── model.py
│       ├── solvers.py
│       ├── generators.py
│       ├── metrics.py
│       ├── risk_objectives.py
│       └── experiment_utils.py
│
├── scripts/
│   ├── reproduce_all.py
│   ├── run_all_experiments.py
│   ├── make_figures.py
│   ├── make_www_summary_figure.py
│   ├── make_tables.py
│   ├── summarize_results.py
│   ├── check_outputs.py
│   ├── write_run_metadata.py
│   └── write_manifest.py
│
├── tests/
│   └── test_core.py
│
└── docs/
    ├── REPRODUCIBILITY.md
    ├── EXPERIMENTS.md
    ├── FIGURE_INDEX.md
    └── PAPER_OUTPUTS.md
```

The `data/processed/`, `figures/`, and `results/` directories are populated by the reproduction workflow.

## Reproducibility scope

The experiments use fixed random-number seeds and model parameters.

Numerical results are expected to reproduce within ordinary floating-point and optimizer tolerances, although small differences may arise across computing environments.

Execution times and runtime-related figures are machine-dependent.

The reproduction workflow records the software environment and generates output checks to support independent verification.
