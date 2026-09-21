# Replication code

**Distribution-free bounds on double marginalization under α-concave demand**  
Qianqian Wen and Kayson Hu, *Economics Letters*, forthcoming. Accepted 12 September 2026.  
https://doi.org/10.1016/j.econlet.2026.113244

This repository contains the symbolic and numerical code used to verify the results, tables and figures in the paper. The paper uses no empirical data.

## Files

| File | Description |
|---|---|
| `run_all.py` | Runs the full verification suite and exits nonzero if any check fails |
| `sympy_verification.py` | Symbolic checks for the closed-form results in the paper |
| `numerical_verification.py` | Numerically solves the vertical game across 81 demand specifications and produces the reported tables and `results.json` |
| `mollifier_check.py` | Checks the smoothing construction in Lemma A.1 numerically |
| `make_figures.py` | Reproduces Figures 1 and 2 from `results.json` in PDF, EPS and PNG formats |
| `requirements.txt` | Pinned Python dependencies |
| `reference_output/` | Reference console output, numerical results and reproduced figures |

## Running the code

The code was run under Python 3.12 and uses SymPy, SciPy, NumPy and Matplotlib.

Install the required packages with:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python run_all.py
```

The full suite takes approximately ten minutes.

All computations are deterministic. Numerical optimisation uses a dense-grid search followed by bounded golden-section refinement. With the pinned dependencies, rerunning the scripts should reproduce the supplied reference output to the reported precision.

Numerical tolerances are specified in the relevant scripts. The main equilibrium-ratio checks use a tolerance of `2e-4`, while the welfare closed-form cross-checks use `1e-5`.

## What is checked

### Symbolic results

`sympy_verification.py` verifies:

- the α-concavity condition and pass-through bound in Lemma 1;
- the α-affine benchmark in Section 3.1;
- the exponential-demand benchmark;
- Example 1 and the failure of the condition \(w^* \leq p_M\);
- the defining equation and special cases for the bound in Theorem 1;
- the welfare expressions reported in Section 4.

### Numerical results

`numerical_verification.py` independently solves the vertical game over the demand specifications used in the paper.

It checks the benchmark solutions, verifies the bound from Theorem 1 across the numerical examples, and reproduces the sequences used to illustrate sharpness. It also generates the numerical entries used in Table 1 and independently checks the welfare calculations.

### Smoothing construction

`mollifier_check.py` provides a numerical check of the smoothing argument in Lemma A.1 using a \(C^\infty\) bump kernel.

For the parameter values reported in the reference output, it checks:

- α-concavity;
- strict monotonicity of demand;
- preservation of the integrated monopoly price;
- localisation of the relevant maximiser around the truncation point; and
- convergence of the resulting price ratio.

The supplied runs use

\[
\alpha \in \{0,0.5,1,2\}
\]

and

\[
\varepsilon \in \{0.08,0.04,0.02,0.01\}.
\]

## Map from paper to code

| Result in the paper | Verification |
|---|---|
| Lemma 1: α-concavity iff \(\mu'(p)\leq-\alpha\), and \(\rho=1/(1-\mu')\leq1/(1+\alpha)\) | `sympy_output.txt`, §2 |
| Section 3.1 benchmark: \(p_M=\alpha/(1+\alpha)\), \(p^*(w)=(\alpha+w)/(1+\alpha)\), \(w^*=p_M\), price ratio \((2+\alpha)/(1+\alpha)\), profit split \(1+\alpha\), and exponential ratio 2 | `sympy_output.txt`, §1; numerical checks in `numerical_output.txt` |
| Example 1: \(w^*\leq p_M\) can fail; for \(\alpha=1,\ q=0.85\), \(0.1275>1/8\) | `sympy_output.txt`, §5; truncated-affine rows in `numerical_output.txt` |
| Lemma A.1 smoothing construction | `mollifier_output.txt` |
| Theorem 1: defining equation, monotonicity of \(\phi\), \(r^*(1)=1+\sqrt{2}/2\), \(r^*(0)=-W_{-1}(-e^{-2})\approx3.146193\), and comparison with the benchmark ratio | `sympy_output.txt`, §4 |
| Numerical verification of the Theorem 1 bound | `numerical_output.txt` |
| Sharpness of the bound | “Approaching the tight bound” section of `numerical_output.txt` |
| Table 1 | “LATEX TABLE 1” block in `numerical_output.txt`, rows α ∈ {0, 0.5, 1, 2, 5}; the α = 0 truncated entry (3.1400) is the q = 3.14 run in the “Approaching the tight bound” section |
| Welfare results in Section 4 | `sympy_output.txt`, §3; WELFARE block in `numerical_output.txt` |
| Figures 1 and 2 | `make_figures.py`, using `results.json` |
| Maximum price reduction under log-concavity | \(1-1/r^*(0)=0.6822\) |

For the log-concave sharpness calculation, the truncated-exponential examples with \(q=2,\ 2.5,\ 3,\ 3.14\) produce price ratios \(2.0000,\ 2.5000,\ 3.0000,\ 3.1400\), approaching

\[
r^*(0)\approx3.1462.
\]

For positive α, the corresponding truncated α-affine examples approach \(r^*(\alpha)\).

## Reference output

The `reference_output/` directory contains the console output from each verification script, the generated `results.json`, and the reproduced figures.

These files are included so that the numerical results can be compared directly with a fresh run of the code.

## License

MIT License. See `LICENSE`.

Copyright © 2026 Qianqian Wen and Kayson Hu.
