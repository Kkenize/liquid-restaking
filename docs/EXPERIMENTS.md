# Computational experiments

The experiments are synthetic stress tests designed to isolate the mechanisms
in the paper. They are not estimates of current protocol returns, slash
frequencies, or realized correlations. Every output records a fixed seed and is
written to CSV before any figure is rendered.

## Common model

Allocator `i` chooses exposure `x[i,j]` subject to

- `0 <= x[i,j] <= W[i]`, and
- `sum_j x[i,j] <= kappa[i] W[i]`.

Rewards are proportional shares of fixed position pools. Slash events are
thresholded Gaussian factors with global, operator, software-client, cloud,
and idiosyncratic components. The mean-variance game uses analytical
gradients. Equilibrium candidates are checked with both best-response
exploitability and a projected variational-inequality residual.

## Numerical equilibrium methods

Experiments A and D use the simultaneous projected pseudo-gradient method
as the primary equilibrium solver. The method combines Euclidean projection,
analytical payoff gradients, and simultaneous strategy updates.

Cyclic best response provides an independent equilibrium check and serves
as a fallback solver. Experiment E compares constant-step, adaptive, and
square-root-decay projected methods.

The risk model uses the loss covariance

```text
V = diag(alpha) Cov(F) diag(alpha)
```

and generates `Cov(F)` from attainable Gaussian-threshold Bernoulli events.

## A. Collateral architecture frontier

- Four allocators and twelve slashable positions.
- Five fixed seeds.
- Reuse cap `kappa = 1, ..., 12`.
- Near-independent, clustered, and global-plus-clustered factor structures.
- The same 80,000 slash scenarios are reused across all values of `kappa`
  within a seed.
- Outputs: reward, gross exposure, effective breadth, expected claims,
  standard deviation, and 99% CVaR.

This experiment asks whether relaxing the collateral-reuse cap raises rewards,
exposure, and tail risk at the same rate.

## B. Dependence misspecification

- One allocator and sixteen positions.
- Ten fixed seeds and `kappa = 4`.
- Compare the true covariance-aware optimum with a diagonal-covariance model.
- Fit rank-three covariance approximations from 250, 1,000, and 5,000
  simulated observations.
- Evaluate all candidates on 120,000 independent scenarios.

This experiment measures utility regret, true variance, tail risk, and
allocation turnover caused by ignoring or imperfectly estimating dependence.

## C. Factor topology

- One allocator and sixteen positions.
- Eight fixed seeds.
- Compare `kappa = 1` and `kappa = 4`.
- Hold total systematic latent variance fixed while shifting weight to a global
  factor.
- Compare the covariance-aware optimum with reward-yield, equal-weight,
  lowest-probability, and one-per-operator-cluster heuristics.

This experiment asks whether simple diversification rules remain effective when
positions share several kinds of infrastructure.

## D. Competition and welfare

- Eight positions and five fixed seeds.
- `kappa = 2` and total strategic capital fixed at four units.
- Number of allocators in `{1, 2, 4, 8, 12}`.
- Background-exposure scale in `{0.5, 1.5, 5}`.
- Compare the unique Nash equilibrium with a centralized planner.

This experiment isolates the reward-capture externality created by competition
for shares of fixed reward pools.

## E. Solver behavior

- Twenty-four environments: three conditioning regimes, four problem sizes,
  and two fixed seeds.
- Compare cyclic, reverse-order, and randomized best response; fixed-step,
  adaptive, and square-root-decay projected pseudo-gradient; and a natural
  residual minimizer on small instances.
- Record convergence, runtime, iterations, exploitability, VI residual, and
  cross-solver solution spread.

Runtime comparisons are machine dependent. Equilibrium residuals and solution
agreement are the primary scientific diagnostics.

## F. Risk-objective robustness

- One allocator, twelve positions, five fixed seeds, and `kappa = 4`.
- Optimize on 30,000 scenarios and evaluate on 180,000 independent scenarios.
- Compare mean-variance with exact entropic risk over a matched sensitivity
  grid.
- Compare CVaR-premium weights from 0.05 to 1.

This experiment tests which allocation conclusions depend on the quadratic
risk approximation.

## Commands

Run one family:

```bash
python scripts/run_all_experiments.py --experiment A
```

Run all families and validation:

```bash
python scripts/run_all_experiments.py --experiment all
```

Run the complete clean workflow, including figures, tables, summaries, checks,
and manifests:

```bash
python scripts/reproduce_all.py
```

The current numerical summary is generated from the CSV outputs and stored in
`RESULTS_SUMMARY.md`.
