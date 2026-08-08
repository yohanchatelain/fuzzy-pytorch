# Experiment plan: one test per proposition in Section 4

Each entry states the claim, the design, the quantity measured, and the outcome
that would falsify it.

## E1 — prop:unbiased.  E[s_n] = s exactly
Claim: under ideal SR with a fresh draw per operation, the reduction is an
unbiased estimator of the exact sum; for **any** fixed summation tree, **any** n,
and with **no** smallness assumption on nu.

Design: fixed vector c (n terms, exactly representable at precision t so that
the test isolates accumulation from input rounding). Reference sum s computed by
Kahan summation in binary64. R independent SR reductions; report
z = (mean - s) / (sd/sqrt(R)).
Vary: summation order (left-to-right, reversed, pairwise tree); n; and a regime
with nu >> 1/2 to probe the "no smallness assumption" clause.
Control: the same reduction under RN, which should show |z| >> 3.

Falsified if |z| grows with R (a real bias), or if any summation order shows a
systematic offset.

## E2 — prop:azuma / eq:azuma.  Concentration
Claim: with probability >= 1-delta, and provided nu <= 1/2,
|s_n - s| <= 2u e^{2nu} sqrt(2 log(2/delta) [sum c_i^2 + sum_{j>=2} A_j^2]).

Design: same harness inside the stated precondition (nu <= 1/2). Measure the
empirical violation rate over R trials at delta = 0.05, and the tightness ratio
bound / max observed error.
Separately: RMS error versus n for SR and RN, fitted as a power law, to test the
sqrt(n) versus n scaling that motivates the bound.

Falsified if the violation rate exceeds delta, or if SR scales like n.

## E3 — prop:stagnation / eq:window.  Swamping
Claim: RN(S+mu) = S when mu < ulp(S)/2; complete swamping impossible when
S < mu/u and guaranteed when S >= 2mu/u, so the transition window is
S/mu in [1/u, 2/u]. Summing identical mu, RN stalls at k_* = Theta(1/u), while
SR still satisfies E[s_n] = n·mu.

Design: accumulate mu = 1 repeatedly under RN at precision t; record the k at
which the sum stops advancing. Compare k_* against the predicted window
[2^t, 2^{t+1}]. Sweep t. Run the same accumulation under SR and compare the mean
against n·mu.

Falsified if k_* falls outside the window, or if SR also stalls.

## E4 — eq:coherence.  Coherence factor kappa
Claim: kappa = ||(s_2..s_n)||_1 / ||(s_2..s_n)||_2 in [1, sqrt(n-1)], equal to 1
when one partial sum dominates and sqrt(n-1) when all are comparable. It compares
RN drift *exposure* to SR fluctuation *scale* -- explicitly not realized
quantities.

Design: three vectors with constructed kappa -- comparable partial sums (high),
one dominant partial sum (low), and an intermediate. For each, measure the
realized |RN bias| / SR standard deviation and compare to kappa.

Expected outcome: the bound on kappa holds exactly (it is an inequality between
norms), while the correspondence to realized quantities is directional at best.
The paper already says this; the experiment is to show how loose it is.

## E5 — eq:transfer.  Transfer through the activation
Claim: E[g(z+eta) - g(z)] = g''(z) tau^2 / 2 + O(E|eta|^3). For GELU
g'' = (2 - z^2) phi(z) is bounded, so an O(u) fluctuation gives O(u^2) activation
bias; for ReLU the same calculation gives a *first-order* E|eta|/2 at z = 0.

Design: fix z, inject zero-mean Gaussian eta of scale tau, measure the mean
output shift over many draws, and compare with the second-order prediction.
Sweep tau over two decades and fit the exponent: GELU should give slope 2 and
ReLU slope 1.

Falsified if GELU shows first-order scaling, or if the measured coefficient
disagrees with g''(z)/2.

## E6 — prop:jensen.  The convexity penalty
Claim: E[L(lambda+xi, y)] >= L(lambda, y) for any zero-mean xi.

Design: random logits, zero-mean perturbations of several covariances, measure
the mean loss change. It should be non-negative always, and approach
tr(H_L Sigma)/2 as the perturbation shrinks.

Falsified by any negative mean loss change outside sampling error.

## E7 — prop:compare / eq:second-order / eq:criterion.  The decomposition
Claim: E[L(lambda+Delta) - L(lambda)] = grad^T b + b^T H_L b / 2
+ tr(H_L Sigma) / 2 + R, and SR beats RN exactly when the variance tax is
smaller than the drift removed.

Design: apply perturbations with known bias b and covariance Sigma; compare the
measured mean loss change against each predicted term; report the residual R and
verify it shrinks faster than second order as the perturbation scale falls.

Falsified if R is comparable to the terms it corrects.

## E8 — eq:gain.  Covariance gain
Claim: Gamma(P,Sigma) = sqrt(tr(P Sigma P^T)/tr Sigma) <= ||P||_2, with
Gamma(P, tau^2 I) = ||P||_F / sqrt(d).

Design: random matrices of several shapes and conditionings; evaluate both sides.
These are algebraic identities, so the test is for correctness of the statement
as written rather than for an empirical effect.

Falsified by any counterexample.
