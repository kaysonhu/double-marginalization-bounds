"""Runs the full verification suite in order; exits nonzero on any failure."""

import pathlib
import subprocess
import sys

STEPS = [
    ("sympy_verification.py", "sympy_output.txt", "ALL SYMBOLIC CHECKS PASSED"),
    ("numerical_verification.py", "numerical_output.txt", "ALL NUMERICAL CHECKS PASSED"),
    ("mollifier_check.py", "mollifier_output.txt", "ALL MOLLIFIER CHECKS PASSED"),
    ("make_figures.py", "figures_log.txt", "figure2.pdf written"),
]

def main():
    here = pathlib.Path(__file__).resolve().parent
    for script, outfile, marker in STEPS:
        print(f"running {script} ...", flush=True)
        with open(here / outfile, "w", encoding="utf-8") as fh:
            ret = subprocess.run(
                [sys.executable, str(here / script)],
                stdout=fh, stderr=subprocess.STDOUT, cwd=here,
            ).returncode
        text = (here / outfile).read_text(encoding="utf-8", errors="replace")
        if ret != 0 or marker not in text:
            print(f"FAILED: {script} (exit {ret}); see {outfile}")
            return 1
        print(f"  OK: {marker}")
    print("ALL VERIFICATION STEPS PASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
