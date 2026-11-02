"""Example 2 — step-growth polymerization: the kernel source (paper B).

Run:  python examples/ex2_polymer.py          (~10 s)
      python examples/ex2_polymer.py --quick

What it shows
-------------
1. The reduced trajectory family b1(b0; alpha) = 2(b0^(alpha/2) - b0)/(2 - alpha)
   with the exact logarithmic branch at alpha = 2 — the carrier [A] has
   cancelled; only the asymmetry alpha remains.
2. Terminal arrest for r < 1: trajectories stop at (b0f, b1f) solving
   (1 - alpha) b0f + b0f^(alpha/2) = (2 - alpha)(1 - r).
   ("Arrest" = freezing by carrier exhaustion: sigma -> 0 stops every
   channel at once and a resumable frozen state with residual activated
   inventory survives — not a chain-growth mechanistic "termination".)
3. The two-site ODEs in real time (carrier sigma = [A]/[BB]_0 included)
   converge exactly to the closed-form arrest state, and the along-trajectory
   carrier-cancellation residual is at round-off.
4. Published regression anchors (Carothers, Flory, Xw dual-path, frozen state).

Outputs: printed tables + examples/output/ex2_polymer.png + ex2_polymer_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import a2sp  # noqa: F401  (prefer the installed package)
except ImportError:  # dev checkout / zero-install usage
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a2sp.core import integrate, reduced_solution  # noqa: E402
from a2sp.models import polymer as poly  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
R = 0.9  # B-deficient feed for the arrest demonstration


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    alphas = [0.5, 2.0, 3.0] if args.quick else [0.5, 1.0, 2.0, 3.0, 10.0]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # ---- panel A: reduced trajectories + arrest endpoints -----------------
    b0 = np.logspace(-3, 0, 400)
    for a in alphas:
        b1 = poly.b1_trajectory(b0, a)
        ax1.plot(b0, b1, lw=1.8, label=f"$\\alpha={a}$")
    for a in alphas:
        if abs(a - 2.0) < 1e-9:
            continue
        st = poly.terminal_arrest(a, R)
        ax1.plot(st["b0f"], st["b1f"], "o", color="black", ms=5)
    ax1.set_xlim(1, 0)  # reaction progresses right -> left
    ax1.set_xlabel("$b_0$ (intact BB fraction)")
    ax1.set_ylabel("$b_1$ (activated B* fraction)")
    ax1.set_title(f"Reduced trajectories, arrest at $r={R}$ (dots)")
    ax1.legend(frameon=False, fontsize=8)

    # ---- panel B: two-site ODE converges to the closed-form arrest --------
    a = 3.0
    ts, Y = integrate(lambda t, y: poly.two_site_rhs(t, y, a, R), [1.0, 0.0], 0.0, 200.0, 40_000)
    st = poly.terminal_arrest(a, R)
    ax2.plot(ts, Y[:, 0], lw=1.8, label="$b_0(\\tau)$ ODE")
    ax2.plot(ts, Y[:, 1], lw=1.8, label="$b_1(\\tau)$ ODE")
    ax2.axhline(st["b0f"], ls=":", color="gray")
    ax2.axhline(st["b1f"], ls=":", color="gray")
    ax2.set_xlabel(r"$\tau = 2k[BB]_0 t$")
    ax2.set_ylabel("fraction")
    ax2.set_title(f"Two-site kinetics, $\\alpha={a}$, $r={R}$: ODE $\\to$ arrest")
    ax2.legend(frameon=False, fontsize=8)
    db0 = abs(Y[-1, 0] - st["b0f"])
    db1 = abs(Y[-1, 1] - st["b1f"])

    # ---- carrier-cancellation check: the real-time ODE trajectory must lie --
    # exactly on the reduced closed-form curve b1 = f(b0; alpha) — that is the
    # statement that the carrier sigma has cancelled.
    curve_dev = float(
        max(
            abs(y1 - float(reduced_solution(b0, a / 2.0, 1.0, 0.0)))
            for b0, y1 in zip(Y[:, 0], Y[:, 1])
        )
    )

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "ex2_polymer.png", dpi=150)

    # ---- anchors -----------------------------------------------------------
    print("=" * 72)
    print("Published regression anchors (paper B / symer README)")
    print("=" * 72)
    xn = poly.carothers_xn(0.8, 0.45)
    xw = poly.xw_all(0.8, 0.45, 1.0)
    frz = poly.terminal_arrest(3.0, 0.9)
    checks = [
        ("Carothers Xn(0.8, 0.45)", xn, 2.0, 1e-12),
        ("Xw dual-path anchor (r=0.8, pB=0.45, a=1)", xw, 721 / 239, 1e-9),
        ("frozen b0f (a=3, r=0.9)", frz["b0f"], 0.05676, 5e-6),
        ("frozen b1f (a=3, r=0.9)", frz["b1f"], 0.08648, 5e-6),
        ("Xn_f all-species (r=0.9)", frz["xn_f_all"], 19.0, 1e-9),
        ("ODE endpoint b0 vs arrest", Y[-1, 0], st["b0f"], 5e-3),
        ("ODE endpoint b1 vs arrest", Y[-1, 1], st["b1f"], 5e-3),
        ("reduced-curve max dev (carrier cancelled)", curve_dev, 0.0, 1e-7),
    ]
    failed = 0
    for name, got, want, tol in checks:
        ok = abs(got - want) <= tol
        failed += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<44} got {got: .8g}  want {want: .8g}")

    payload = {
        "r": R,
        "ode_endpoint": {"b0": float(Y[-1, 0]), "b1": float(Y[-1, 1])},
        "arrest_closed_form": {k: float(v) for k, v in st.items()},
        "reduced_curve_max_dev": curve_dev,
        "anchors": {name: {"got": float(g), "want": float(w)} for name, g, w, _ in checks},
    }
    (OUT / "ex2_polymer_results.json").write_text(
        json.dumps(
            payload,
            indent=1,
            default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o),
        )
    )
    print(f"  saved {OUT / 'ex2_polymer.png'} and ex2_polymer_results.json")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
