"""Anchor cross-validation for the a2sp kernel.

Every expected value below is a *published* number, not a number produced by
this codebase:

- Paper B SI section 7.1.2 dual-path anchor (r=0.8, pB=0.45, alpha=1):
  walking formula = 3.016736, consistent with the direct series to 4e-16.
- Paper B SI section 7.1.1 worked quantities at the same anchor: b1 = 0.495.
- symer kernel README (v0.1.3) regression anchors: frozen state
  (alpha=3, r=0.9): b0f = 0.05676, b1f = 0.08648, Xn = 19; Carothers
  Xn = (1+r)/(1+r-2pB); Flory D = 1+p -> 2 at alpha = 1; trajectory
  b1 = 2(b0^(alpha/2) - b0)/(2 - alpha); frozen inventory
  (x1, x2) = (mu, alpha)/(alpha + mu).

Run:  python tests/test_anchors.py   (or pytest tests/test_anchors.py)
"""

from __future__ import annotations

import math
import pathlib
import sys

try:
    import a2sp  # noqa: F401  (prefer the installed package)
except ImportError:  # dev checkout / zero-install usage
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from a2sp.core import reduced_rk4, reduced_solution, residual  # noqa: E402
from a2sp.models import polymer as poly  # noqa: E402
from a2sp.models import rumor as rum  # noqa: E402

RESULTS: list[tuple[str, float, float, float, bool]] = []


def check(name: str, expected: float, computed: float, tol: float) -> None:
    diff = abs(computed - expected)
    ok = diff <= tol
    RESULTS.append((name, expected, computed, diff, ok))


# ---------------------------------------------------------------- polymer


def test_trajectory_alpha1_b1() -> None:
    """b1(b0=0.3025; alpha=1) = 0.495 (paper B SI 7.1.1 anchor values)."""
    b1 = poly.b1_trajectory(0.3025, 1.0)
    check("b1(b0=0.3025, alpha=1)", 0.495, b1, 1e-12)


def test_carothers_anchor() -> None:
    """Xn(r=0.8, pB=0.45) = 1.8/0.9 = 2 (Eq. 9)."""
    check("carothers Xn(0.8, 0.45)", 2.0, poly.carothers_xn(0.8, 0.45), 1e-12)


def test_xw_dual_path_anchor() -> None:
    """Xw(r=0.8, pB=0.45, alpha=1) = 3.016736 (SI S7.7; = 721/239)."""
    xw = poly.xw_all(0.8, 0.45, 1.0)
    check("Xw dual-path anchor (published)", 3.016736, xw, 5e-7)
    check("Xw dual-path anchor (exact 721/239)", 721.0 / 239.0, xw, 1e-12)


def test_flory_equal_reactivity() -> None:
    """alpha = 1 recovers Flory: 1 - pB = sqrt(b0) along the trajectory."""
    worst = 0.0
    for k in range(1, 100):
        b0 = k / 100.0
        pB = poly.pB_of(b0, poly.b1_trajectory(b0, 1.0))
        worst = max(worst, abs(1.0 - pB - math.sqrt(b0)))
    check("Flory identity max|1-pB-sqrt(b0)| (alpha=1)", 0.0, worst, 1e-12)


def test_arrest_anchor() -> None:
    """Frozen state (alpha=3, r=0.9): b0f=0.05676, b1f=0.08648, Xn=19."""
    t = poly.terminal_arrest(3.0, 0.9)
    check("b0f (alpha=3, r=0.9, published)", 0.05676, t["b0f"], 5e-6)
    check("b1f (alpha=3, r=0.9, published)", 0.08648, t["b1f"], 5e-6)
    check("Xn_f all-species (1+r)/(1-r)", 19.0, t["xn_f_all"], 1e-12)


def test_arrest_b1f_identity() -> None:
    """Exact identity on the arrest root: b1(b0f; a) = 2(1 - r - b0f)."""
    alpha, r = 3.0, 0.9
    b0f = poly.arrest_b0f(alpha, r)
    lhs = poly.b1_trajectory(b0f, alpha)
    rhs = 2.0 * (1.0 - r - b0f)
    check("arrest identity b1(b0f)=2(1-r-b0f)", rhs, lhs, 1e-12)


def test_terminal_r_gt_1_consistency() -> None:
    """S7.8 closed form = S7.7 evaluated at the r>1 terminal state (r=2)."""
    r = 2.0
    s78 = poly.xw_terminal_r_gt_1(r)
    pA, qB, w = 1.0 / r, 0.0, (r - 1.0) / r
    s77 = 1.0 + 2.0 * 1.0 * (3.0 + pA - qB) / ((1.0 + r) * w)
    check("terminal Xw r=2: S7.8 vs S7.7", s78, s77, 1e-12)


# ------------------------------------------------------- reduced law core


def test_reduced_residual_off_resonance() -> None:
    """RK4 vs closed form: kappa = 1.5 and 0.25, to round-off level."""
    check("residual kappa=1.5", 0.0, residual(1.5, 1.0, 0.0, 0.05), 1e-9)
    check("residual kappa=0.25", 0.0, residual(0.25, 1.0, 0.0, 0.05), 1e-9)


def test_reduced_residual_log_branch() -> None:
    """kappa = 1 (resonance): exact log branch vs RK4."""
    y_ana = reduced_solution(0.01, 1.0, 1.0, 0.0)
    y_num = reduced_rk4(1.0, 1.0, 0.0, 0.01, n=200000)
    check("residual kappa=1 (log branch)", y_ana, y_num, 1e-7)


# ----------------------------------------------------------------- rumor


def test_frozen_inventory_roundtrip() -> None:
    """(x1,x2) = (mu,alpha)/(alpha+mu); fingerprint inverts linearly."""
    worst_inv = 0.0
    worst_sum = 0.0
    for alpha, mu in [(2.21, 1.0), (0.5, 2.0), (3.0, 0.7), (0.1, 10.0)]:
        x1, x2 = rum.frozen_inventory(alpha, mu)
        worst_sum = max(worst_sum, abs(x1 + x2 - 1.0))
        worst_inv = max(worst_inv, abs(rum.fingerprint_invert(x1, x2, mu) - alpha))
        check(
            f"frozen inventory x1 (alpha={alpha}, mu={mu})",
            mu / (alpha + mu),
            x1,
            1e-12,
        )
    check("fingerprint inversion max|alpha_hat - alpha|", 0.0, worst_inv, 1e-12)
    check("frozen inventory max|x1+x2-1|", 0.0, worst_sum, 1e-12)


def test_rumor_layer_residuals() -> None:
    """Merged (kappa=mu) and senior (kappa=gamma) layers: RK4 vs closed form."""
    for mu in (0.5, 2.0, 1.0):
        tol = 1e-7 if mu == 1.0 else 1e-9
        y_ana = rum.merged_layer(0.01, mu, 0.5, 0.4)
        y_num = reduced_rk4(mu, 0.5, 0.4, 0.01, n=200000)
        check(f"merged layer residual (mu={mu})", y_ana, y_num, tol)
    gamma = 2.21 + 1.0
    y_ana = rum.senior_layer(0.01, 2.21, 1.0, 0.5, 0.05)
    y_num = reduced_rk4(gamma, 0.5, 0.05, 0.01, n=200000)
    check("senior layer residual (gamma=3.21)", y_ana, y_num, 1e-9)


# ----------------------------------------------------------------- runner


def _main() -> int:
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    width = max(len(r[0]) for r in RESULTS)
    failed = 0
    print(f"{'anchor':<{width}}  {'expected':>14}  {'computed':>16}  {'|diff|':>10}  verdict")
    print("-" * (width + 60))
    for name, exp, comp, diff, ok in RESULTS:
        if not ok:
            failed += 1
        print(f"{name:<{width}}  {exp:>14.6g}  {comp:>16.10g}  {diff:>10.2e}  {'PASS' if ok else 'FAIL'}")
    print("-" * (width + 60))
    print(f"{len(RESULTS) - failed}/{len(RESULTS)} anchors PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
