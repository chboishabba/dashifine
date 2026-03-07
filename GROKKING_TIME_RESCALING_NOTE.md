# Time-Rescaled Delayed Generalization in Near-Critical Grokking

## Abstract

Training trajectories in the grokking regime exhibit delayed generalization:
models rapidly memorize the training set while test accuracy remains near
chance for a long interval before rising sharply to near-perfect performance.

For the mod-97 task in the near-critical band
`weight_decay ∈ {0.20, 0.22, 0.24, 0.25, 0.30, 0.35, 0.40}`, the observed
test-accuracy curves collapse well under time normalization by `t50`, while
final test accuracy remains nearly unchanged. This suggests that regularization
primarily controls the timescale of the late generalization phase rather than
its trajectory shape.

This note records a reduced-model conjecture and theorem target matching that
empirical picture.

## Empirical observation

Let:

- `A_test(t; λ)` denote test accuracy at epoch `t` under weight decay `λ`
- `t10, t50, t95` denote the first epochs where test accuracy crosses 10%,
  50%, and 95%

The current 7-point dataset supports:

1. Fast memorization:
   - training accuracy saturates quickly (`t_fit ≈ 80`)
2. Delayed generalization:
   - test accuracy stays near chance for a long plateau
3. Late transition:
   - test accuracy rises sharply only after the plateau
4. Trajectory collapse:
   - `A_test(t; λ)` aligns well under `t / t50(λ)`
5. Onset scaling:
   - among simple 2-parameter screens, `t95 ~ 1 / λ` is currently the best fit
6. Shared rise law after onset alignment:
   - a shared logistic rise fits well after shifting by a common normalized
     onset `t0 ≈ 0.81 * t50`

## Informal reduced-model conjecture

Once memorization has saturated, the remaining evolution of generalization is
effectively one-dimensional. Weight decay primarily rescales the speed along
the same late-time trajectory rather than changing its shape.

Equivalently, there exists a smooth proxy `G(t; λ)` and a common delayed curve
`F` such that

`G(t; λ) ≈ F((t - T_mem) / τ(λ))`

where:

- `T_mem` is the rapid memorization time
- `τ(λ)` is a weight-decay-dependent slow timescale
- `F` is common across the near-critical band

Empirically, `t50(λ)` is the best current proxy for `τ(λ)`.

## Metastable-escape interpretation

The present trajectory family is also consistent with a metastable-escape
picture:

1. rapid approach to a memorization regime,
2. long residence in a quasi-stable delayed-generalization plateau,
3. slower regularization-controlled escape,
4. transition into the late generalization rise.

In this reading, the system first reaches a memorization attractor or slow
manifold quickly, then drifts for a long time before escaping toward the
generalizing branch. The observed `t95 ~ 1 / λ` trend is more suggestive of
deterministic slow drift than of rare-event stochastic barrier hopping.

This remains a mechanistic hypothesis, not a proved statement.

## Minimal assumptions

### A1. Fast memorization phase

There exists `T_mem(λ)` such that:

- `T_mem(λ) << τ(λ)`
- after `T_mem`, the training system lies near a memorization manifold `M`

### A2. Slow reduced variable

There exists a scalar latent variable `z(t; λ)` and a monotone map `Φ` such
that:

`G(t; λ) = Φ(z(t; λ))`

### A3. Weight decay enters mainly through timescale

For `λ` in the near-critical regime, the slow variable obeys:

`ż = ε(λ) f(z)`

with `f` independent of `λ`.

### A4. Single delayed transition

There is a unique transition interval in `z` over which `Φ(z)` rises from
near-chance to near-grokking, with no secondary reversals.

## Candidate theorem target

Under A1-A4, there exists a common profile `F` and a timescale `τ(λ)` such
that:

`G(t; λ) = F((t - T_mem(λ)) / τ(λ))`

up to an error that vanishes in the reduced-model limit.

Consequences:

1. Trajectory collapse under time normalization by `τ(λ)`
2. Shape invariance across the near-critical band
3. Timing control by regularization without materially changing the final
   grokked state

## Metastable fast-slow theorem target

An equivalent theorem target is a two-stage fast-slow model:

- a fast variable controlling memorization completion
- a slow variable controlling post-memorization escape/generalization

The empirical conjecture is that training rapidly enters a memorizing regime
and then undergoes a slower regularization-controlled transition into
generalization.

## Stronger theorem target: inverse-law onset scaling

If the slow rate scales linearly with weight decay,

`ε(λ) = c λ`

for `c > 0`, then any threshold time `t_α(λ)` satisfying

`G(t_α(λ); λ) = α`

obeys:

`t_α(λ) = T_mem(λ) + C_α / λ`

for constants `C_α` depending only on the reduced flow and the threshold.

This is the mathematical form of the current empirical screen:

- `t10`
- `t50`
- `t95`

all scale approximately like `1 / λ` in the tested regime.

## Flagship conjecture

In the near-critical grokking regime, regularization does not primarily alter
the shape of the generalization trajectory; it rescales the clock of a common
delayed-generalization flow.

## Current rise-phase law

The strongest current shape result is now more specific than a generic
"sigmoid-like" statement.

On the current 7-point dataset:

- a shared-parameter Gompertz candidate on `epoch / t50` did not win
- a simple logistic baseline did better on the normalized full trajectories
- the cleanest rise-phase fit comes from aligning onset at a shared normalized
  location and then fitting one shared logistic curve

Empirically, the rise phase is well described by:

`A_test(t; λ) ≈ 1 / (1 + exp(-k(((t / t50(λ)) - c) - x0))))`

with:

- shared timescale proxy `t50(λ)`
- shared normalized onset `c ≈ 0.8055`
- shared logistic shape parameters across the tested band

This one-shift law gives `mse≈0.000360`, essentially matching the more flexible
per-run fitted-onset version (`mse≈0.000351`).

So the current evidence supports:

- a strong time-rescaled family claim
- a shared logistic post-escape rise once onset is aligned
- and a plausible metastable-escape interpretation for the delayed plateau

The rise-phase law is therefore no longer just "model selection in progress."
The current open question is whether this shared-onset logistic law survives
other architectures, optimizers, or tasks.

## Candidate dynamical interpretation of the rise law

A natural reduced-model route is:

- a latent slow variable `z`
- slow deterministic escape from a memorization regime
- a monotone observation map from `z` to the measured generalization proxy

Under that interpretation:

- metastability explains the delayed plateau and onset timing
- the shared logistic law describes the post-escape growth phase

This is the cleanest current theorem target for the observed rise shape.

## Post-escape logistic interpretation

The current normalized trajectories may also be read as:

- metastable delayed plateau before escape
- logistic-like rise after escape begins

That is, metastability explains the delayed onset, while a sigmoid-like law
may describe the growth phase after the system has begun leaving the
memorization regime.

On the current 7-point normalized-accuracy fit, the simple logistic baseline
outperforms the shared-parameter Gompertz candidate. So the best current
shape-law wording is now stronger than “sigmoid-like post-escape rise”:

- shared normalized onset near `0.81 * t50`
- shared logistic rise after that onset

not “established Gompertz law.”

## Limitation statement

Current evidence supports the existence of a common time-rescaled
delayed-generalization family in one architecture/task regime. It does not yet
establish universality across architectures, datasets, or optimizers.
