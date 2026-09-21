"""Symbolic verification of every closed form in the paper."""

import sympy as sp

a = sp.Symbol('alpha', positive=True)
p, w, c, t = sp.symbols('p w c t', positive=True)

BAR = "=" * 78

def sec(title):
    print()
    print(BAR)
    print(title)
    print(BAR)

sec("1. ALPHA-AFFINE BENCHMARK  D(p) = (1-p)^(1/alpha) on [0,1]")

D = (1 - p) ** (1 / a)

R = p * D
foc_int = sp.together(sp.diff(R, p))
pM = sp.solve(sp.numer(foc_int), p)
pM = [s for s in pM if sp.simplify(s - 1) != 0][0]
pM = sp.simplify(pM)
print("Integrated monopoly price  p_M =", pM)
assert sp.simplify(pM - a / (1 + a)) == 0
print("   -> equals alpha/(1+alpha):  VERIFIED")

prof_R = (p - w) * D
foc_R = sp.together(sp.diff(prof_R, p))
pstar = sp.solve(sp.numer(foc_R), p)
pstar = sp.simplify([s for s in pstar if sp.simplify(s - 1) != 0][0])
print("Retailer best reply        p*(w) =", pstar)
assert sp.simplify(pstar - (a + w) / (1 + a)) == 0
print("   -> equals (alpha+w)/(1+alpha):  VERIFIED")
rho = sp.simplify(sp.diff(pstar, w))
print("Pass-through rho = dp*/dw =", rho, " (constant = 1/(1+alpha))")

prof_M = w * D.subs(p, pstar)
foc_M = sp.together(sp.diff(prof_M, w))
wstar = sp.solve(sp.numer(foc_M), w)
wstar = [s for s in wstar if sp.simplify(s - 1) != 0][0]
wstar = sp.simplify(wstar)
print("Optimal wholesale price    w* =", wstar)
assert sp.simplify(wstar - a / (1 + a)) == 0
print("   -> equals alpha/(1+alpha) = p_M:  VERIFIED (w* = p_M, alpha-affine case)")

pDM = sp.simplify(pstar.subs(w, wstar))
print("DM retail price            p_DM =", sp.factor(pDM))
assert sp.simplify(pDM - a * (2 + a) / (1 + a) ** 2) == 0
print("   -> equals alpha(2+alpha)/(1+alpha)^2:  VERIFIED")

ratio = sp.simplify(pDM / pM)
print("Price ratio          p_DM/p_M =", ratio)
assert sp.simplify(ratio - (2 + a) / (1 + a)) == 0
print("   -> equals (2+alpha)/(1+alpha):  VERIFIED")

mu = sp.simplify(-D / sp.diff(D, p))
print("Markup function mu(p) = -D/D' =", mu, "  (= alpha(1-p), affine)")
print("mu'(p) =", sp.simplify(sp.diff(mu, p)), "  (= -alpha identically)")

marg_R = sp.simplify(pDM - wstar)
print("Retailer margin p_DM - w* =", marg_R)
split = sp.simplify(wstar / marg_R)
print("Manufacturer/retailer profit split w*/(p_DM - w*) =", split)
assert sp.simplify(split - (1 + a)) == 0
print("   -> upstream earns (1+alpha) times downstream:  VERIFIED")

lam = sp.Symbol('lambda', positive=True)
p0, w0 = sp.symbols('p0 w0', positive=True)
D0 = sp.exp(-lam * p0)
pM0 = sp.solve(sp.diff(p0 * D0, p0), p0)[0]
ps0 = sp.solve(sp.diff((p0 - w0) * D0, p0), p0)[0]
ws0 = sp.solve(sp.diff(w0 * D0.subs(p0, ps0), w0), w0)[0]
pDM0 = ps0.subs(w0, ws0)
print("alpha = 0 (D = exp(-lambda p)):  p_M =", pM0, ", p*(w) =", ps0,
      ", w* =", ws0, ", p_DM =", pDM0, ", ratio =", sp.simplify(pDM0 / pM0))
assert sp.simplify(pDM0 / pM0 - 2) == 0
print("   -> ratio = 2 = limit of (2+alpha)/(1+alpha):  VERIFIED")

sec("2. CURVATURE IDENTITY AND PASS-THROUGH BOUND (general demand)")

Df = sp.Function('D', positive=True)
x = sp.Symbol('x')
Dg, D1, D2 = Df(x), sp.diff(Df(x), x), sp.diff(Df(x), x, 2)

mug = -Dg / D1
mup = sp.simplify(sp.diff(mug, x))
print("mu'(p) simplifies to:", mup)
ident = sp.simplify(mup - (-1 + Dg * D2 / D1 ** 2))
assert ident == 0
print("   -> mu' = -1 + D D''/(D')^2 :  VERIFIED")

lhs = sp.diff(Dg ** a, x, 2)
rhs = a * Dg ** (a - 2) * (Dg * D2 + (a - 1) * D1 ** 2)
assert sp.simplify(lhs - rhs) == 0
print("(D^a)'' = a D^(a-2) [ D D'' + (a-1)(D')^2 ] :  VERIFIED")
print("   => alpha-concavity (D^a)''<=0  <=>  D D'' <= (1-alpha)(D')^2")
print("   => mu' = -1 + D D''/(D')^2 <= -1 + (1-alpha) = -alpha")
print("   => ALPHA-CONCAVITY  <=>  mu'(p) <= -alpha   (key inequality)")

pm = sp.Function('p_m')
F = Df(x) + (x - c) * D1
Fp = sp.diff(F, x)
Fc = sp.diff(F, c)
rho_g = sp.simplify(-Fc / Fp)
print("rho(c) = dp_m/dc =", rho_g)

rho_foc = sp.simplify(rho_g.subs(x - c, -Dg / D1))

rho_manual = D1 ** 2 / (2 * D1 ** 2 - Dg * D2)
print("Using FOC (p-c) = -D/D':  rho = (D')^2 / (2(D')^2 - D D'')")

diff_expr = sp.together(rho_manual - 1 / (1 + a))
num = sp.simplify(sp.numer(diff_expr))
print("rho - 1/(1+alpha) has numerator:", sp.factor(num))

target = Dg * D2 - (1 - a) * D1 ** 2
assert sp.simplify(num - target) == 0
rel = sp.simplify(target - Dg ** (2 - a) / a * sp.diff(Dg ** a, x, 2))
assert rel == 0
print("   numerator = D^(2-a)/a * (D^a)''  <= 0 under alpha-concavity")
print("   denominator = (1+alpha)(2(D')^2 - D D'') ; and 2(D')^2 - D D''")
print("     >= 2(D')^2 - (1-a)(D')^2 = (1+a)(D')^2 > 0 under alpha-concavity")
print("   => rho(c) <= 1/(1+alpha) for every cost c :  VERIFIED  (Lemma 1)")
print("   Equivalently rho = 1/(1 - mu'(p_m(c))) and mu' <= -alpha.")

assert sp.simplify(rho_manual - 1 / (1 - mup)) == 0
print("   rho = 1/(1 - mu') :  VERIFIED")

sec("3. WELFARE FOR ALPHA-AFFINE DEMAND: CS, PROFITS, DWL, DWL RATIO")

CS = sp.integrate(D, (p, t, 1))
CS = sp.simplify(CS)
print("CS(t) = int_t^1 D =", CS)

Wc = sp.simplify(CS.subs(t, 0))
print("First-best welfare W_c = CS(0) =", Wc)

CS_M = sp.simplify(CS.subs(t, pM))
PI_M = sp.simplify(pM * D.subs(p, pM))
W_M = sp.simplify(CS_M + PI_M)
DWL_M = sp.simplify(Wc - W_M)
print("CS_M   =", CS_M)
print("Pi_int =", PI_M)
print("DWL_M  =", sp.simplify(DWL_M))

CS_DM = sp.simplify(CS.subs(t, pDM))
PI_up = sp.simplify(wstar * D.subs(p, pDM))
PI_dn = sp.simplify((pDM - wstar) * D.subs(p, pDM))
PI_DM = sp.simplify(PI_up + PI_dn)
W_DM = sp.simplify(CS_DM + PI_DM)
DWL_DM = sp.simplify(Wc - W_DM)
print("CS_DM  =", CS_DM)
print("Pi_up  =", PI_up, "   Pi_down =", PI_dn)
print("Industry profit Pi_DM =", PI_DM)
print("DWL_DM =", sp.simplify(DWL_DM))

ratio_DWL = sp.simplify(DWL_DM / DWL_M)
print("DWL ratio  DWL_DM/DWL_M =", sp.simplify(ratio_DWL))

r_at_1 = sp.simplify(ratio_DWL.subs(a, 1))
print("   at alpha = 1:", r_at_1)
assert sp.simplify(r_at_1 - sp.Rational(9, 4)) == 0
print("   -> equals 9/4 (classic linear-demand result):  VERIFIED")

r_at_0 = sp.limit(ratio_DWL, a, 0, '+')
print("   limit alpha -> 0+:", sp.simplify(r_at_0), "=", sp.N(r_at_0, 8))
r_at_inf = sp.limit(ratio_DWL, a, sp.oo)
print("   limit alpha -> oo:", r_at_inf)

ratio_fn = sp.lambdify(a, ratio_DWL, "mpmath")
import mpmath
scan = [(x / 1000, ratio_fn(x / 1000)) for x in range(1, 10001)]
scan_min = min(v for _, v in scan)
scan_max_a, scan_max = max(((x, v) for x, v in scan), key=lambda t: t[1])
print(f"   range scan on (0,10], step 1e-3: min = {mpmath.nstr(scan_min, 8)},"
      f" max = {mpmath.nstr(scan_max, 8)} at alpha = {scan_max_a}")
assert float(r_at_0) >= 2.2479 and scan_min >= 2.2479 and scan_max <= 2.2564
print("   -> DWL ratio stays within [2.2479, 2.2564] on alpha in [0, 10]:  VERIFIED")

share_M = sp.simplify(DWL_M / Wc)
share_DM = sp.simplify(DWL_DM / Wc)
print("DWL_M/W_c  =", share_M)
print("DWL_DM/W_c =", share_DM)
for aval in [sp.Rational(1, 2), 1, 2]:
    print(f"   alpha = {aval}:  DWL_M/Wc = {sp.N(share_M.subs(a, aval), 6)},"
          f"  DWL_DM/Wc = {sp.N(share_DM.subs(a, aval), 6)},"
          f"  DWL ratio = {sp.N(ratio_DWL.subs(a, aval), 6)}")

sec("4. CORRECTED TIGHT BOUND r*(alpha)")

tau = sp.Symbol('tau', positive=True)
phi = tau * (1 - a * (tau - 1)) ** (1 / a)
rhs_r = (1 + a) ** (-1 / a)
print("Defining equation:  phi_a(r) := r (1 - a(r-1))^(1/a) = (1+a)^(-1/a)")

dlogphi = sp.simplify(sp.diff(sp.log(phi), tau))
print("d log phi / d tau =", sp.simplify(dlogphi))
print("   = 1/tau - (1+a-... ) : negative on (1, 1+1/a) since 1-a(tau-1) < 1 < tau")

eq1 = sp.Eq(phi.subs(a, 1), rhs_r.subs(a, 1))
sols = sp.solve(eq1, tau)
sols = [sp.nsimplify(sp.simplify(s), [sp.sqrt(2)]) for s in sols]
print("alpha = 1 solutions:", sols)
rstar1 = [s for s in sols if sp.simplify(s - 1) > 0][-1]
rstar1 = sp.simplify(sp.Max(*sols))
print("r*(1) =", sp.sqrt(2)/2 + 1, "check:", sp.simplify(rstar1 - (1 + sp.sqrt(2) / 2)) == 0)
assert sp.simplify(rstar1 - (1 + sp.sqrt(2) / 2)) == 0
print("   -> r*(1) = 1 + sqrt(2)/2 =", sp.N(1 + sp.sqrt(2) / 2, 10), ":  VERIFIED")

lim_lhs = sp.limit(phi, a, 0, '+')
lim_rhs = sp.limit(rhs_r, a, 0, '+')
print("limit of phi as a->0+ :", lim_lhs, "   limit of RHS:", lim_rhs)
assert sp.simplify(lim_lhs - tau * sp.exp(1 - tau)) == 0
assert sp.simplify(lim_rhs - sp.exp(-1)) == 0
print("   -> alpha = 0 equation:  r e^(1-r) = e^(-1)  i.e.  r e^(-r) = e^(-2)")
rstar0 = -sp.LambertW(-sp.exp(-2), -1)
print("r*(0) = -W_{-1}(-e^(-2)) =", sp.N(rstar0, 10))
resid = sp.N(rstar0 * sp.exp(-rstar0) - sp.exp(-2), 30)
print("   residual r e^(-r) - e^(-2) =", resid, " ~ 0:  VERIFIED")

naive = (2 + a) / (1 + a)
gap = sp.simplify(phi.subs(tau, naive) / rhs_r)
print("phi_a((2+a)/(1+a)) / (1+a)^(-1/a) =", gap)
assert sp.simplify(gap - naive) == 0
print("   = (2+a)/(1+a) > 1  =>  r*(alpha) > (2+alpha)/(1+alpha) STRICTLY, all alpha")
print("   (the conjectured alpha-affine bound is NOT the tight distribution-free bound)")

print()
print(" alpha     r*(alpha)     (2+a)/(1+a)      gap")
import mpmath
mpmath.mp.dps = 20
for aval in [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10]:
    if aval == 0:
        rv = float(-mpmath.lambertw(-mpmath.e ** -2, -1).real)
    else:
        f = lambda r, av=aval: r * (1 - av * (r - 1)) ** (1 / av) - (1 + av) ** (-1 / av)
        lo, hi = 1.0 + 1e-12, 1.0 + 1.0 / aval - 1e-12
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
        rv = 0.5 * (lo + hi)
    nv = (2 + aval) / (1 + aval)
    print(f"  {aval:<6}  {rv:<12.6f}  {nv:<12.6f}  {rv - nv:+.6f}")

sec("5. COUNTEREXAMPLE LOGIC (truncated alpha-affine demand)")

print("""Interior manufacturer optimum under alpha-affine demand with integrated
price m and D_M := D(m):   Pi_int = m * D_M * (1+a)^(-1/a).
Cliff at retail price q = tau*m (truncation): profit -> q D(q) = m*D_M*phi_a(tau).
Cliff dominates interior  <=>  phi_a(tau) > (1+a)^(-1/a)  <=>  tau < r*(alpha).""")

lhs5 = sp.simplify(wstar * D.subs(p, pDM))
rhs5 = sp.simplify(pM * D.subs(p, pM) * (1 + a) ** (-1 / a))
assert sp.simplify(lhs5 - rhs5) == 0
print("Pi_int = w* D(p_DM) = p_M D(p_M) (1+a)^(-1/a):  VERIFIED symbolically")

tt = sp.Symbol('t', positive=True)
cliff = tt * (1 - tt)
eq = sp.solve(sp.Eq(cliff, sp.Rational(1, 8)), tt)
print("alpha = 1: cliff t(1-t) = 1/8 at t =", eq, "-> larger root / p_M =",
      sp.simplify(sp.Max(*eq) / sp.Rational(1, 2)))
assert sp.simplify(sp.Max(*eq) * 2 - (1 + sp.sqrt(2) / 2)) == 0
print("   -> ratio bound 1 + sqrt(2)/2 recovered from the cliff construction: VERIFIED")

print()
print(BAR)
print("ALL SYMBOLIC CHECKS PASSED")
print(BAR)
