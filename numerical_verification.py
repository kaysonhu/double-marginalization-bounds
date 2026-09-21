"""Numerical solution of the vertical game for all demand specifications; emits LaTeX tables and results.json."""

import json
import math

import numpy as np
from scipy import integrate, optimize, special, stats

ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
BAR = "=" * 100

def r_star(alpha):
    if alpha == 0:
        return float(-special.lambertw(-math.exp(-2), -1).real)
    phi = lambda r: r * (1 - alpha * (r - 1)) ** (1 / alpha) - (1 + alpha) ** (-1 / alpha)
    lo, hi = 1 + 1e-13, 1 + 1 / alpha - 1e-13
    return optimize.brentq(phi, lo, hi, xtol=1e-14)

def naive_bound(alpha):
    return (2 + alpha) / (1 + alpha)

def global_argmax(f, lo, hi, n=4001):
    xs = np.linspace(lo, hi, n)
    vals = np.array([f(x) for x in xs])
    i = int(np.argmax(vals))
    a_, b_ = xs[max(i - 1, 0)], xs[min(i + 1, n - 1)]
    res = optimize.minimize_scalar(lambda x: -f(x), bounds=(a_, b_), method="bounded",
                                   options={"xatol": 1e-12})
    if -res.fun >= vals[i]:
        return float(res.x), float(-res.fun)
    return float(xs[i]), float(vals[i])

class Market:

    def __init__(self, name, D, pbar):
        self.name, self.D, self.pbar = name, D, pbar

    def integrated(self):
        pM, revM = global_argmax(lambda q: q * self.D(q), 0.0, self.pbar)
        return pM, revM

    def retail_price(self, w):
        p, _ = global_argmax(lambda q: (q - w) * self.D(q), w, self.pbar, n=1501)
        return p

    def dm_equilibrium(self):
        wstar, piM = global_argmax(lambda w: w * self.D(self.retail_price(w)),
                                   0.0, self.pbar, n=601)
        pDM = self.retail_price(wstar)
        return wstar, pDM, piM

    def cs(self, p):
        v, _ = integrate.quad(self.D, p, self.pbar, limit=400)
        return v

    def solve(self):
        pM, revM = self.integrated()
        wstar, pDM, piM_up = self.dm_equilibrium()
        out = {
            "name": self.name,
            "p_M": pM, "w_star": wstar, "p_DM": pDM,
            "ratio": pDM / pM, "w_over_pM": wstar / pM,
            "pi_int": revM,
            "pi_up": piM_up,
            "pi_down": (pDM - wstar) * self.D(pDM),
            "CS_M": self.cs(pM), "CS_DM": self.cs(pDM),
            "W_c": self.cs(0.0),
        }
        out["W_M"] = out["CS_M"] + out["pi_int"]
        out["W_DM"] = out["CS_DM"] + out["pi_up"] + out["pi_down"]
        out["DWL_M"] = out["W_c"] - out["W_M"]
        out["DWL_DM"] = out["W_c"] - out["W_DM"]
        out["DWL_ratio"] = out["DWL_DM"] / out["DWL_M"] if out["DWL_M"] > 1e-12 else float("nan")
        return out

def check_alpha_concave(D, pbar, alpha, n=2000, tol=1e-7):
    ps = np.linspace(pbar * 1e-6, pbar * (1 - 1e-6), n)
    vals = np.array([D(x) for x in ps])
    ok = vals > 1e-300
    ps, vals = ps[ok], vals[ok]
    g = np.log(vals) if alpha == 0 else vals ** alpha
    second = g[2:] - 2 * g[1:-1] + g[:-2]
    return bool(np.all(second <= tol * max(1.0, np.abs(g).max())))

def h_based(name, h, alpha):
    def D(p):
        v = h(p)
        return v ** (1 / alpha) if v > 0 else 0.0
    return Market(f"{name}", D, 1.0)

def truncated_affine(alpha, frac=0.95, S=5000.0):
    rs = r_star(alpha)
    pM = alpha / (1 + alpha)
    q = (1 + frac * (rs - 1)) * pM
    hq = 1 - q
    pbar = q + hq / S

    def D(p):
        v = (1 - p) if p <= q else hq - S * (p - q)
        return v ** (1 / alpha) if v > 0 else 0.0

    return Market(f"truncated affine (q={q:.4f}, S={S:.0f})", D, pbar), q

def truncated_exponential(q=3.0, S=5000.0):
    def D(p):
        if p <= q:
            return math.exp(-p)
        return math.exp(-q - S * (p - q))
    pbar = q + 40.0 / S
    return Market(f"truncated exponential (q={q}, S={S:.0f})", D, pbar)

SMOOTH_H = [
    ("affine  h=1-p (alpha-affine)", lambda p: 1 - p),
    ("h = 1-p^2", lambda p: 1 - p ** 2),
    ("h = 1-p^3", lambda p: 1 - p ** 3),
    ("h = cos(pi p/2)", lambda p: math.cos(math.pi * p / 2)),
    ("h = 1-p/2-p^2/2", lambda p: 1 - p / 2 - p ** 2 / 2),
    ("h = log2(2-p)", lambda p: math.log(2 - p) / math.log(2)),
    ("h = sqrt(1-p)", lambda p: math.sqrt(max(1 - p, 0.0))),
]

def alpha_zero_markets():
    ms = [Market("exponential  e^-p", lambda p: math.exp(-p), 60.0)]
    ms.append(Market("normal survival  1-Phi((p-1)/0.5)",
                     lambda p: float(stats.norm.sf(p, loc=1.0, scale=0.5)), 12.0))
    ms.append(Market("logistic survival  1/(1+e^((p-1)/0.3))",
                     lambda p: 1.0 / (1.0 + math.exp((p - 1.0) / 0.3)), 30.0))
    ms.append(Market("Gumbel survival (mu=1, b=0.4)",
                     lambda p: float(-np.expm1(-np.exp(-(p - 1.0) / 0.4))), 25.0))
    return ms

def fmt_row(res, extra=""):
    return (f"  {res['name']:<38} pM={res['p_M']:.4f}  w*={res['w_star']:.4f}  "
            f"w*/pM={res['w_over_pM']:.3f}  pDM={res['p_DM']:.4f}  "
            f"ratio={res['ratio']:.4f}{extra}")

def main():
    results = {"alphas": {}, "alpha0": [], "r_star": {}, "naive": {}}

    print(BAR)
    print("TIGHT BOUND r*(alpha) vs CONJECTURED BOUND (2+alpha)/(1+alpha)")
    print(BAR)
    print(f"  {'alpha':>6}  {'r*(alpha)':>10}  {'(2+a)/(1+a)':>12}")
    for al in [0.0] + ALPHAS:
        print(f"  {al:>6}  {r_star(al):>10.6f}  {naive_bound(al):>12.6f}")
        results["r_star"][str(al)] = r_star(al)
        results["naive"][str(al)] = naive_bound(al)

    grand_ok = True

    for al in ALPHAS:
        rs, nb = r_star(al), naive_bound(al)
        print()
        print(BAR)
        print(f"ALPHA = {al}    naive bound = {nb:.6f}    tight bound r* = {rs:.6f}")
        print(BAR)
        rows = []
        for name, h in SMOOTH_H:
            m = h_based(name, h, al)
            assert check_alpha_concave(m.D, m.pbar, al), f"{name} not {al}-concave!"
            res = m.solve()
            rows.append(res)
            tag = ""
            if name.startswith("affine"):
                err = abs(res["ratio"] - nb)
                ok = err < 2e-4
                grand_ok &= ok
                tag = f"   [= naive bound, err={err:.1e} {'OK' if ok else 'FAIL'}]"
            else:
                ok = res["ratio"] <= rs + 1e-6
                grand_ok &= ok
                tag = f"   [<= r*: {'OK' if ok else 'FAIL'}]"
            print(fmt_row(res, tag))

        mt, q = truncated_affine(al)
        assert check_alpha_concave(mt.D, mt.pbar, al), "truncation broke alpha-concavity!"
        rt = mt.solve()
        above = rt["ratio"] > nb + 1e-4
        below = rt["ratio"] <= rs + 1e-6
        grand_ok &= (above and below)
        print(fmt_row(rt, f"   [> naive: {'YES - CONJECTURE VIOLATED' if above else 'no'};"
                          f" <= r*: {'OK' if below else 'FAIL'}]"))
        rows.append(rt)

        print(f"  -> truncated alpha-affine: w*/p_M = {rt['w_over_pM']:.3f} "
              f"(wholesale price EXCEEDS integrated price; the step that breaks the"
              f" conjectured proof)")
        results["alphas"][str(al)] = rows

    al, rs, nb = 0.0, r_star(0.0), 2.0
    print()
    print(BAR)
    print(f"ALPHA = 0 (log-concave)    naive bound = 2    tight bound r* = {rs:.6f}")
    print(BAR)
    for m in alpha_zero_markets():
        assert check_alpha_concave(m.D, m.pbar, 0.0), f"{m.name} not log-concave!"
        res = m.solve()
        if m.name.startswith("exponential"):
            err = abs(res["ratio"] - 2.0)
            ok = err < 2e-4
            tag = f"   [= 2 exactly, err={err:.1e} {'OK' if ok else 'FAIL'}]"
        else:
            ok = res["ratio"] < 2.0 + 1e-6
            tag = f"   [< 2: {'OK' if ok else 'FAIL'}]"
        grand_ok &= ok
        print(fmt_row(res, tag))
        results["alpha0"].append(res)

    mt = truncated_exponential(q=3.0)
    assert check_alpha_concave(mt.D, mt.pbar, 0.0), "truncated exp not log-concave!"
    rt = mt.solve()
    above = rt["ratio"] > 2.0 + 1e-3
    below = rt["ratio"] <= rs + 1e-6
    grand_ok &= (above and below)
    print(fmt_row(rt, f"   [> 2: {'YES - CONJECTURE VIOLATED' if above else 'no'};"
                      f" <= r* = {rs:.4f}: {'OK' if below else 'FAIL'}]"))
    results["alpha0"].append(rt)

    print()
    print("  Approaching the tight bound with truncated exponentials (S = 5000):")
    for q in [2.0, 2.5, 3.0, 3.14]:
        res = truncated_exponential(q=q).solve()
        print(f"    q = {q:<5}  ratio = {res['ratio']:.4f}   (r*(0) = {rs:.4f})")
        results["alpha0"].append(res)

    print()
    print(BAR)
    print("WELFARE (alpha-affine family): closed-form cross-check + numerics")
    print(BAR)
    print(f"  {'alpha':>6} {'DWL_M':>10} {'DWL_DM':>10} {'DWL ratio':>10} "
          f"{'CS_M':>10} {'CS_DM':>10} {'Pi_int':>10} {'Pi_DM':>10} {'W_DM/W_M':>10}")
    for al in ALPHAS:
        res = [r for r in results["alphas"][str(al)] if r["name"].startswith("affine")][0]

        A = 1 + al
        DWL_M_cf = al / A * (1 - A ** (-1 / al) * (2 + al) / A)
        DWL_DM_cf = al / A - al / (A * A ** (2 * (1 + al) / al)) - al * (2 + al) / A ** (2 + 2 / al)
        ok1 = abs(res["DWL_M"] - DWL_M_cf) < 1e-5
        ok2 = abs(res["DWL_DM"] - DWL_DM_cf) < 1e-5
        grand_ok &= (ok1 and ok2)
        print(f"  {al:>6} {res['DWL_M']:>10.5f} {res['DWL_DM']:>10.5f} "
              f"{res['DWL_ratio']:>10.5f} {res['CS_M']:>10.5f} {res['CS_DM']:>10.5f} "
              f"{res['pi_int']:>10.5f} {res['pi_up'] + res['pi_down']:>10.5f} "
              f"{res['W_DM'] / res['W_M']:>10.5f}"
              f"   [closed-form check {'OK' if ok1 and ok2 else 'FAIL'}]")

    print()
    print(BAR)
    print("LATEX TABLE 1: price ratios")
    print(BAR)
    print(r"\begin{tabular}{lccccc}")
    print(r"\hline")
    print(r"$\alpha$ & $\frac{2+\alpha}{1+\alpha}$ & $\alpha$-affine & "
          r"worst smooth non-affine & truncated $\alpha$-affine & $r^*(\alpha)$ \\")
    print(r"\hline")
    for al in [0.0] + ALPHAS:
        if al == 0:
            aff = 2.0
            others = max(r["ratio"] for r in results["alpha0"][1:4])
            trunc = [r for r in results["alpha0"] if r["name"].startswith("truncated")][0]["ratio"]
        else:
            rows = results["alphas"][str(al)]
            aff = rows[0]["ratio"]
            others = max(r["ratio"] for r in rows[1:7])
            trunc = rows[7]["ratio"]
        astr = f"{al:g}"
        print(f"{astr} & {naive_bound(al):.4f} & {aff:.4f} & {others:.4f} & "
              f"{trunc:.4f} & {r_star(al):.4f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")

    print()
    print(BAR)
    print("LATEX TABLE 2: welfare, alpha-affine family")
    print(BAR)
    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"$\alpha$ & $DWL_M/W_c$ & $DWL_{DM}/W_c$ & $DWL_{DM}/DWL_M$ & $W_{DM}/W_M$ \\")
    print(r"\hline")
    for al in ALPHAS:
        res = [r for r in results["alphas"][str(al)] if r["name"].startswith("affine")][0]
        print(f"{al:g} & {res['DWL_M'] / res['W_c']:.4f} & {res['DWL_DM'] / res['W_c']:.4f} & "
              f"{res['DWL_ratio']:.4f} & {res['W_DM'] / res['W_M']:.4f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")

    with open("results.json", "w") as fh:
        json.dump(results, fh, indent=1)

    print()
    print(BAR)
    print("ALL NUMERICAL CHECKS PASSED" if grand_ok else "*** SOME CHECKS FAILED ***")
    print(BAR)
    return 0 if grand_ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
