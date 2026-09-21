"""Builds figure1 and figure2 from results.json."""

import json
import math

import numpy as np
from scipy import optimize, special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

with open("results.json") as fh:
    RES = json.load(fh)

def r_star(alpha):
    if alpha == 0:
        return float(-special.lambertw(-math.exp(-2), -1).real)
    phi = lambda r: r * (1 - alpha * (r - 1)) ** (1 / alpha) - (1 + alpha) ** (-1 / alpha)
    return optimize.brentq(phi, 1 + 1e-13, 1 + 1 / alpha - 1e-13, xtol=1e-14)

fig, ax = plt.subplots(figsize=(5.2, 3.6))

agrid = np.linspace(0, 10, 400)
ax.plot(agrid, [r_star(a) for a in agrid], "-", color="black", lw=1.4,
        label=r"tight bound $r^*(\alpha)$ (Theorem 1)")
ax.plot(agrid, (2 + agrid) / (1 + agrid), "--", color="0.35", lw=1.2,
        label=r"$\alpha$-affine benchmark $\frac{2+\alpha}{1+\alpha}$")

alphas = [float(a) for a in RES["alphas"].keys()]
aff_x, aff_y, tr_x, tr_y, sm_x, sm_y = [], [], [], [], [], []
for a in alphas:
    rows = RES["alphas"][f"{a:g}" if f"{a:g}" in RES["alphas"] else str(a)]
    for r in rows:
        if r["name"].startswith("affine"):
            aff_x.append(a); aff_y.append(r["ratio"])
        elif r["name"].startswith("truncated"):
            tr_x.append(a); tr_y.append(r["ratio"])
        else:
            sm_x.append(a); sm_y.append(r["ratio"])

for r in RES["alpha0"]:
    if r["name"].startswith("exponential"):
        aff_x.append(0.0); aff_y.append(r["ratio"])
    elif r["name"].startswith("truncated") and abs(r["ratio"] - 3.14) < 0.005:
        tr_x.append(0.0); tr_y.append(r["ratio"])
    elif not r["name"].startswith("truncated"):
        sm_x.append(0.0); sm_y.append(r["ratio"])

ax.scatter(sm_x, sm_y, marker="x", s=14, color="0.55", lw=0.8, zorder=3,
           label="other $\\alpha$-concave demands")
ax.scatter(aff_x, aff_y, marker="o", s=18, facecolors="white",
           edgecolors="black", lw=0.9, zorder=4,
           label=r"$\alpha$-affine demands")
ax.scatter(tr_x, tr_y, marker="^", s=22, color="black", zorder=4,
           label=r"truncated $\alpha$-affine demands")

ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"price ratio  $p_{DM}/p_{M}$")
ax.set_xlim(-0.2, 10.2)
ax.set_ylim(0.95, 3.35)
ax.annotate(r"$r^*(0)=-W_{-1}(-e^{-2})\approx 3.146$",
            xy=(0, r_star(0)), xytext=(1.1, 3.02),
            arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4"), fontsize=8.5)
ax.annotate(r"$r^*(1)=1+\frac{\sqrt{2}}{2}\approx 1.707$",
            xy=(1, r_star(1)), xytext=(2.1, 1.95),
            arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4"), fontsize=8.5)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figure1.pdf")
fig.savefig("figure1.eps")
fig.savefig("figure1.png")
plt.close(fig)
print("figure1.pdf written")

def dwl_ratio(a):
    A = 1 + a
    dwl_m = a / A * (1 - A ** (-1 / a) * (2 + a) / A)
    dwl_dm = a / A - a / (A * A ** (2 * (1 + a) / a)) - a * (2 + a) / A ** (2 + 2 / a)
    return dwl_dm / dwl_m

fig, ax = plt.subplots(figsize=(5.2, 3.4))
agrid = np.linspace(0.02, 10, 500)
ax.plot(agrid, [dwl_ratio(a) for a in agrid], "-", color="black", lw=1.4,
        label=r"closed form, $\alpha$-affine demand")
lim0 = (1 - 3 * math.exp(-2)) / (1 - 2 * math.exp(-1))
ax.scatter([0], [lim0], marker="s", s=20, facecolors="white", edgecolors="black",
           lw=0.9, zorder=4, label=r"$\alpha\to 0$ limit $\frac{1-3e^{-2}}{1-2e^{-1}}\approx 2.2479$")
ax.axhline(2.25, ls="--", lw=0.9, color="0.45", label=r"$9/4$ (exact at $\alpha=1$)")

xs, ys = [], []
for a_str, rows in RES["alphas"].items():
    for r in rows:
        if r["name"].startswith("affine"):
            xs.append(float(a_str)); ys.append(r["DWL_ratio"])
ax.scatter(xs, ys, marker="o", s=16, facecolors="white", edgecolors="black",
           lw=0.9, zorder=4, label="numerical verification")

ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$DWL_{DM}\,/\,DWL_{M}$")
ax.set_xlim(-0.2, 10.2)
ax.set_ylim(2.244, 2.260)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figure2.pdf")
fig.savefig("figure2.eps")
fig.savefig("figure2.png")
plt.close(fig)
print("figure2.pdf written")
