# Reproducibility guide

## Recommended environment

Use Python 3.11 or later in a fresh virtual environment. The archived package
versions are listed in `requirements-lock.txt`.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

## One-command full reproduction

```bash
python scripts/reproduce_all.py
```

The command performs these steps in order:

1. remove generated CSVs, figures, tables, logs, and manifests;
2. run the unit tests;
3. run experiments A--F and the Monte Carlo validation report;
4. generate every individual PDF and PNG figure from saved CSVs;
5. generate the three-panel WWW summary figure from saved CSVs;
6. generate CSV and LaTeX tables;
7. generate `RESULTS_SUMMARY.md` from the current outputs;
8. verify expected row counts, convergence certificates, validation tests, and
   output files;
9. record Python, platform, and package metadata;
10. write SHA-256 checksums for the repository snapshot.

Console output is also saved to `results/logs/reproduction.log`.

## Fast validation

```bash
python scripts/reproduce_all.py --mode validation
```

This runs the unit tests and the independent Monte Carlo/cross-solver
validation without running the full experiment grid.

## Determinism and tolerances

- Every scenario and Monte Carlo draw has a fixed random seed.
- Figures are generated only from saved CSVs.
- Model outputs should reproduce up to ordinary floating-point and optimizer
  tolerances.
- Runtime fields and runtime plots are machine dependent.
- Iteration counts can vary slightly across BLAS, SciPy, or processor versions
  even when equilibrium allocations agree numerically.
- The exact software environment for the archived run is recorded in
  `results/run_metadata.json`.

## Output verification

The repository checks:

- expected row counts for every experiment;
- all six mathematical validation tests pass;
- all architecture and competition equilibria converge;
- best-response exploitability and VI residuals meet stored tolerances;
- every expected figure and table exists and is nonempty;
- core result columns contain finite values.

The machine-readable report is `results/reproduction_check.json`.

## Rebuilding selected outputs

```bash
python scripts/run_all_experiments.py --experiment B
python scripts/make_figures.py
python scripts/make_www_summary_figure.py
python scripts/make_tables.py
python scripts/summarize_results.py
python scripts/write_run_metadata.py
python scripts/check_outputs.py
python scripts/write_manifest.py
```

## Integrity check

After reproduction, verify the snapshot with:

```bash
sha256sum -c REPRODUCIBILITY_MANIFEST.txt
```

The manifest excludes `.git`, Python caches, pytest caches, and runtime logs.


The exact mapping from generated outputs to the two manuscript versions is documented in [`PAPER_OUTPUTS.md`](PAPER_OUTPUTS.md).
