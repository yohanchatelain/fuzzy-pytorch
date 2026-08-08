"""E5-E7 with antithetic sampling.

A first attempt measured these directly and could not resolve them: the
quantities are second order in the perturbation scale tau, while the sampling
error of a direct estimator is first order, so the noise exceeded the signal at
every scale tested. Pairing each draw eta with -eta cancels the first-order term
exactly and leaves an estimator whose expectation is the second-order quantity
itself. This is a statement about the estimator, not about the theory.
"""

import numpy as np

rng = np.random.default_rng(20260808)
SQRT2PI = np.sqrt(2 * np.pi)


def phi(z):
    return np.exp(-0.5 * z * z) / SQRT2PI


def Phi(z):
    from scipy.special import ndtr
    return ndtr(z)


def gelu(z):
    return z * Phi(z)


def relu(z):
    return np.maximum(z, 0.0)


def gelu_dd(z):
    return (2 - z * z) * phi(z)


def antithetic_shift(g, z0, tau, N):
    """E[(g(z+eta) + g(z-eta))/2 - g(z)] -- second order, first order cancelled."""
    eta = rng.normal(0.0, tau, N)
    vals = 0.5 * (g(z0 + eta) + g(z0 - eta)) - g(z0)
    return float(vals.mean()), float(vals.std() / np.sqrt(N))


def e5():
    print("\n=== E5  eq:transfer (antithetic) ===")
    N = 2_000_000
    print(f"{'act':<6} {'z':>6} {'tau':>8} {'measured':>13} {'std err':>11} "
          f"{'predicted':>13} {'ratio':>7}")
    for z0 in (0.0, 0.5, -1.0):
        for tau in (1e-2, 1e-3, 1e-4):
            m, se = antithetic_shift(gelu, z0, tau, N)
            pred = 0.5 * gelu_dd(z0) * tau**2
            print(f"{'GELU':<6} {z0:>6.1f} {tau:>8.0e} {m:>13.4e} {se:>11.2e} "
                  f"{pred:>13.4e} {m/pred:>7.3f}")

    print("\n  -- scaling exponent (signal-to-noise checked at each tau) --")
    taus = np.logspace(-3, -1, 7)
    for name, g, want in (("GELU", gelu, 2), ("ReLU", relu, 1)):
        xs, ys = [], []
        for tau in taus:
            m, se = antithetic_shift(g, 0.0, tau, 1_000_000)
            if abs(m) > 5 * se:            # only fit where the signal is resolved
                xs.append(tau); ys.append(abs(m))
        slope = np.polyfit(np.log(xs), np.log(ys), 1)[0]
        print(f"   {name}: slope {slope:.3f} over {len(xs)}/{len(taus)} resolved "
              f"points (predicted {want})")


def softmax(l):
    e = np.exp(l - l.max())
    return e / e.sum()


def ce(l, y):
    return float(-l[y] + np.log(np.exp(l - l.max()).sum()) + l.max())


def ce_batch(L, y):
    m = L.max(axis=1, keepdims=True)
    return (-L[:, y] + np.log(np.exp(L - m).sum(axis=1)) + m[:, 0])


def grad_hess(l, y):
    p = softmax(l)
    g = p.copy()
    g[y] -= 1.0
    return g, np.diag(p) - np.outer(p, p)


def e6():
    print("\n=== E6  prop:jensen (antithetic) ===")
    d, N = 50, 400_000
    print(f"{'trial':<6} {'tau':>7} {'mean dL':>13} {'std err':>11} "
          f"{'tr(H S)/2':>13} {'ratio':>7} {'>=0':>5}")
    allok = True
    for trial in range(3):
        l = rng.normal(0, 2.0, d)
        y = int(rng.integers(d))
        L0 = ce(l, y)
        _, H = grad_hess(l, y)
        for tau in (0.3, 0.1, 0.03, 0.01):
            xi = rng.normal(0, tau, (N, d))
            v = 0.5 * (ce_batch(l + xi, y) + ce_batch(l - xi, y)) - L0
            m, se = float(v.mean()), float(v.std() / np.sqrt(N))
            pred = 0.5 * np.trace(H) * tau**2
            ok = m >= -3 * se
            allok &= ok
            print(f"{trial:<6} {tau:>7.2f} {m:>13.6e} {se:>11.2e} {pred:>13.6e} "
                  f"{m/pred:>7.3f} {'yes' if ok else 'NO':>5}")
    print(f"  all non-negative: {allok}")


def e7():
    print("\n=== E7  eq:second-order (antithetic on the fluctuation) ===")
    d, N = 30, 400_000
    l = rng.normal(0, 1.5, d)
    y = int(rng.integers(d))
    L0 = ce(l, y)
    g, H = grad_hess(l, y)
    bdir = rng.normal(0, 1, d)
    bdir /= np.linalg.norm(bdir)

    print(f"{'scale':>7} {'measured':>13} {'std err':>10} {'predicted':>13} "
          f"{'residual':>12} {'R/s^3':>9}")
    rows = []
    for s in (0.3, 0.1, 0.03, 0.01):
        b = s * bdir
        xi = rng.normal(0, s, (N, d))
        v = 0.5 * (ce_batch(l + b + xi, y) + ce_batch(l + b - xi, y)) - L0
        m, se = float(v.mean()), float(v.std() / np.sqrt(N))
        pred = float(g @ b + 0.5 * b @ H @ b + 0.5 * np.trace(H) * s**2)
        R = m - pred
        rows.append((s, R, se))
        print(f"{s:>7.3f} {m:>13.6e} {se:>10.2e} {pred:>13.6e} {R:>12.4e} "
              f"{R/s**3:>9.4f}")
    resolved = [r for r in rows if abs(r[1]) > 3 * r[2]]
    print(f"  residual resolved above noise at {len(resolved)}/{len(rows)} scales")
    if len(resolved) >= 2:
        sl = np.polyfit(np.log([r[0] for r in resolved]),
                        np.log([abs(r[1]) for r in resolved]), 1)[0]
        print(f"  residual scales as tau^{sl:.2f} (third order predicts 3)")


if __name__ == "__main__":
    e5(); e6(); e7()
