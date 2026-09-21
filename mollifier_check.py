"""Numerical verification of the smoothing lemma (Lemma A.1): mollified truncated alpha-affine demand."""

import math
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import optimize, special

_NODES, _WEIGHTS = leggauss(240)

def _bump_raw(u):
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u)
    inside = np.abs(u) < 1.0
    out[inside] = np.exp(-1.0 / (1.0 - u[inside] ** 2))
    return out

_BUMP_NORM = float(np.sum(_WEIGHTS * _bump_raw(_NODES)))

def eta(u):
    return _bump_raw(u) / _BUMP_NORM

def r_star(alpha):
    if alpha == 0:
        return float(-special.lambertw(-math.exp(-2), -1).real)
    phi = lambda r: r * (1 - alpha * (r - 1)) ** (1 / alpha) - (1 + alpha) ** (-1 / alpha)
    return optimize.brentq(phi, 1 + 1e-13, 1 + 1 / alpha - 1e-13, xtol=1e-14)

class Smoothed:

    def __init__(self, alpha, tau0, eps):
        self.alpha, self.tau0, self.eps = alpha, tau0, eps
        self.delta = eps ** 2
        self.q = tau0
        if alpha > 0:
            self.Aq = (1 - alpha * (tau0 - 1)) ** (1 / alpha)
            self.pbar = self.q + eps
        else:
            self.pbar = math.inf

    def g(self, p):
        p = np.asarray(p, dtype=float)
        a, q, e = self.alpha, self.q, self.eps
        if a > 0:
            a1 = 1 - a * (p - 1)
            a2 = self.Aq ** a * (q + e - p) / e
            return np.minimum(a1, a2)
        l1 = -(p - 1)
        l2 = -(q - 1) - (p - q) / e
        return np.minimum(l1, l2)

    def gprime(self, p):
        p = np.asarray(p, dtype=float)
        a, q, e = self.alpha, self.q, self.eps
        if a > 0:
            s1, s2 = -a, -self.Aq ** a / e
        else:
            s1, s2 = -1.0, -1.0 / e
        return np.where(p < q, s1, s2)

    def G(self, p):
        p = np.asarray(p, dtype=float)
        d = self.delta
        shifted = p[..., None] - d * _NODES
        return np.sum(_WEIGHTS * eta(_NODES) * self.g(shifted), axis=-1)

    def Gprime(self, p):
        p = np.asarray(p, dtype=float)
        d = self.delta
        shifted = p[..., None] - d * _NODES
        return np.sum(_WEIGHTS * eta(_NODES) * self.gprime(shifted), axis=-1)

    def D(self, p):
        Gv = self.G(p)
        if self.alpha > 0:
            return np.where(Gv > 0, np.maximum(Gv, 0.0) ** (1 / self.alpha), 0.0)
        return np.exp(Gv)

    def mu(self, p):
        Gv, Gp = self.G(p), self.Gprime(p)
        if self.alpha > 0:
            return -self.alpha * Gv / Gp
        return -1.0 / Gp

    def revenue(self, p):
        return np.asarray(p) * self.D(p)

    def Pi(self, p):
        return (np.asarray(p) - self.mu(p)) * self.D(p)

def argmax_on(f, lo, hi, n=6001):
    grid = np.linspace(lo, hi, n)
    vals = f(grid)
    i = int(np.argmax(vals))
    a = grid[max(i - 2, 0)]
    b = grid[min(i + 2, n - 1)]
    res = optimize.minimize_scalar(lambda x: -f(np.array([x]))[0],
                                   bounds=(a, b), method="bounded",
                                   options={"xatol": 1e-12})
    return float(res.x), float(-res.fun)

def check(alpha, eps_list):
    rs = r_star(alpha)
    tau0 = 1 + 0.9 * (rs - 1)
    print(f"\nalpha = {alpha:g}   r* = {rs:.6f}   tau0 = {tau0:.6f}")
    print(f"{'eps':>8} {'delta':>10} {'p_M(D_eps)':>12} {'p_DM':>10} "
          f"{'ratio':>10} {'w*/p_M':>9} {'in window':>10}")
    ok = True
    last_ratio = None
    for eps in eps_list:
        S = Smoothed(alpha, tau0, eps)
        d, q = S.delta, S.q

        for p in [0.3, 0.9, q - 2.5 * d, min(q + 2.5 * d, q + eps * 0.9)]:
            if abs(S.G(np.array([p]))[0] - S.g(np.array([p]))[0]) > 1e-10:
                print(f"  FAIL C1 at p={p}")
                ok = False

        hi = q + eps * 0.999 if alpha > 0 else q + 8.0
        pp = np.linspace(1e-3, hi, 4001)
        Gv = S.G(pp)
        second = np.diff(Gv, 2)
        if second.max() > 1e-9:
            print(f"  FAIL C2 concavity: max second difference {second.max():.2e}")
            ok = False
        if S.Gprime(pp).max() >= 0:
            print("  FAIL C2 strict decrease")
            ok = False

        pm_hat, _ = argmax_on(S.revenue, 1e-3, hi)
        if abs(pm_hat - 1.0) > 1e-6:
            print(f"  FAIL C3: integrated price {pm_hat:.8f} != 1")
            ok = False

        pdm, _ = argmax_on(S.Pi, 1.0, hi)
        in_window = (q - d) < pdm < (q + eps)
        if not in_window:
            print(f"  FAIL C4: p_DM = {pdm:.8f} outside ({q-d:.6f}, {q+eps:.6f})")
            ok = False

        wstar = pdm - float(S.mu(np.array([pdm]))[0])
        if wstar <= 1.0:
            print(f"  FAIL C5: w* = {wstar:.6f} <= p_M")
            ok = False

        ratio = pdm / 1.0
        print(f"{eps:>8g} {d:>10.2e} {pm_hat:>12.8f} {pdm:>10.6f} "
              f"{ratio:>10.6f} {wstar:>9.4f} {str(in_window):>10}")
        last_ratio = ratio

    if abs(last_ratio - tau0) > eps_list[-1] + 1e-6:
        print(f"  FAIL convergence: |ratio - tau0| = {abs(last_ratio - tau0):.2e}")
        ok = False
    else:
        print(f"  ratio -> tau0: |{last_ratio:.6f} - {tau0:.6f}| = "
              f"{abs(last_ratio - tau0):.2e} <= eps = {eps_list[-1]:g}   OK")
    return ok

def main():
    eps_list = [0.08, 0.04, 0.02, 0.01]
    all_ok = True
    for alpha in [0.0, 0.5, 1.0, 2.0]:
        all_ok &= check(alpha, eps_list)
    print()
    if all_ok:
        print("ALL MOLLIFIER CHECKS PASSED")
        return 0
    print("MOLLIFIER CHECKS FAILED")
    return 1

if __name__ == "__main__":
    sys.exit(main())
