"""Example 1 — rumor spreading: the first application of the A2SP kernel.

Run:  python examples/ex1_rumor.py          (~1 min)
      python examples/ex1_rumor.py --quick  (smoke test, seconds)

What it shows
-------------
1. Mean-field ODEs (paper A eqs 1a-1e) integrate to the exact terminal
   history-resolved composition (x1, x2) = (mu, alpha)/(alpha + mu)
   (Theorem 2, frozen inventory law) — for every seeding size.
2. Gillespie SSA (well-mixed, pair self-correction) converges to the same
   composition from finite-N stochastic dynamics.
3. A configuration-model network CTMC ("our paper's network setting")
   approaches the same composition at large N.

Outputs: printed tables + examples/output/ex1_*.png + ex1_rumor_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import a2sp  # noqa: F401  (prefer the installed package)
except ImportError:  # dev checkout / zero-install usage
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a2sp.core import integrate  # noqa: E402
from a2sp.models import rumor as rum  # noqa: E402
from a2sp.sim import network as netsim  # noqa: E402
from a2sp.sim import wellmixed as wm  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
ALPHA, MU = 2.0, 1.0  # flagship parameter point (alpha = 2 mu)


def part_a_mean_field() -> list[dict]:
    print("=" * 72)
    print("A. Frozen-inventory law (Theorem 2): exact quadrature + adaptive ODE")
    print("=" * 72)
    rows = []
    for alpha, mu in [(2.0, 1.0), (0.5, 2.0), (2.21, 1.0)]:
        exact = rum.frozen_inventory_exact(alpha, mu)
        res = rum.terminal_composition_ode(alpha, mu)
        ok_exact = abs(exact["x2"] - exact["theory_x2"]) < 1e-12 and exact["quad_abs_err"] < 1e-6
        ok_ode = res["converged"] and abs(res["x2"] - res["theory_x2"]) < 1e-8
        ok = ok_exact and ok_ode
        rows.append(
            {
                "alpha": alpha,
                "mu": mu,
                "x1_exact": exact["x1"],
                "x2_exact": exact["x2"],
                "quad_abs_err": exact["quad_abs_err"],
                "x1": res["x1"],
                "x2": res["x2"],
                "converged": res["converged"],
                "ode_abs_err": abs(res["x2"] - res["theory_x2"]),
                "theory_x1": exact["theory_x1"],
                "theory_x2": exact["theory_x2"],
                "ok": ok,
            }
        )
        print(
            f"  (alpha, mu) = ({alpha:>5}, {mu:>4}) : "
            f"exact x2 = {exact['x2']:.10f} (quad err {exact['quad_abs_err']:.1e})  "
            f"adaptive ODE x2 = {res['x2']:.10f} (err {abs(res['x2'] - res['theory_x2']):.1e}, "
            f"converged={res['converged']})  {'PASS' if ok else 'FAIL'}"
        )
    return rows


def part_b_phase_plane() -> None:
    print("=" * 72)
    print("B. Phase-plane S/i vs i — the Law-I trichotomy (deterministic)")
    print("=" * 72)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    eps = 1e-4
    for mu, color in [(0.5, "#C25759"), (1.0, "#49B3FC"), (2.0, "#61AB83")]:
        y0 = [1 - eps, eps, 0.0, 0.0, 0.0]
        ts, Y = integrate(
            lambda t, y: rum.mean_field_rhs(t, y, ALPHA, mu), y0, 0.0, 2000.0, 200_000,
            record_every=400,
        )
        i, S = Y[:, 0], Y[:, 1] + Y[:, 2]
        m = i > 1e-9
        ax.loglog(i[m], (S[m] / i[m]), color=color, lw=1.8, label=f"$\\mu={mu}$")
        # closed-form guide: S/i = C i^(mu-1) - 1/(1-mu)  (mu != 1); C - ln i (mu = 1)
        ig = np.logspace(-8, -1, 100)
        if mu != 1.0:
            C = (eps + (1 - eps) / (1 - mu)) / (1 - eps) ** mu
            guide = C * ig ** (mu - 1) - 1.0 / (1.0 - mu)
        else:
            # audit L2 fix: C = S0/i0 + ln(i0); the previous sign was wrong
            C = eps / (1 - eps) + np.log(1 - eps)
            guide = C - np.log(ig)
        ax.loglog(ig, guide, ls=":", color=color, lw=1.0)
    ax.set_xlabel("ignorant fraction  $i$")
    ax.set_ylabel("spread ratio  $S/i$")
    ax.set_title(f"Phase-plane asymptotics ($\\alpha={ALPHA}$); dots: ODE, lines: closed form")
    ax.legend(frameon=False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "ex1_phase_plane.png", dpi=150)
    print(f"  saved {OUT / 'ex1_phase_plane.png'}")


def part_c_ssa(quick: bool) -> list[dict]:
    print("=" * 72)
    print("C. Gillespie SSA, well-mixed (pair self-correction)")
    print("=" * 72)
    N = 8_000 if quick else 60_000
    seeds = [0] if quick else [0, 1, 2]
    rows = []
    for s in seeds:
        t0 = time.time()
        r = wm.gillespie_wellmixed(ALPHA, MU, N=N, eps=1e-3, seed=s)
        r["wall_s"] = round(time.time() - t0, 2)
        rows.append(r)
        print(
            f"  seed {s}: x1 = {r['x1']:.4f}  x2 = {r['x2']:.4f}  "
            f"(theory {r['theory_x1']:.4f} / {r['theory_x2']:.4f})  "
            f"reach = {r['reach']:.4f}  [{r['events']} events, {r['wall_s']} s]"
        )
    x2s = [r["x2"] for r in rows]
    err = abs(float(np.mean(x2s)) - ALPHA / (ALPHA + MU))
    print(f"  mean x2 = {np.mean(x2s):.4f} vs theory {ALPHA/(ALPHA+MU):.4f}   |dev| = {err:.4f}")
    return rows


def part_d_network(quick: bool) -> list[dict]:
    print("=" * 72)
    print("D. Configuration-model network CTMC (paper's network setting)")
    print("=" * 72)
    N = 1_000 if quick else 5_000
    seeds = [1] if quick else [1, 2]
    rows = []
    for s in seeds:
        t0 = time.time()
        adj = netsim.configuration_model(N, mean_degree=8.0, seed=s)
        r = netsim.network_ctmc(adj, ALPHA, MU, eps=1e-3, seed=100 + s)
        r["wall_s"] = round(time.time() - t0, 2)
        rows.append(r)
        print(
            f"  seed {s}: x1 = {r['x1']:.4f}  x2 = {r['x2']:.4f}  "
            f"(theory {r['theory_x1']:.4f} / {r['theory_x2']:.4f})  "
            f"reach = {r['reach']:.4f}  [{r['events']} events, {r['wall_s']} s]"
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smoke-test mode")
    args = ap.parse_args()

    rows_a = part_a_mean_field()
    part_b_phase_plane()
    rows_c = part_c_ssa(args.quick)
    rows_d = part_d_network(args.quick)

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "parameters": {"alpha": ALPHA, "mu": MU},
        "mean_field": rows_a,
        "ssa": rows_c,
        "network": rows_d,
        "note": "all random seeds recorded for reproducibility",
    }
    (OUT / "ex1_rumor_results.json").write_text(
        json.dumps(
            payload,
            indent=1,
            default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o),
        )
    )

    ok_ode = all(r["ok"] for r in rows_a)
    x2s = [r["x2"] for r in rows_c]
    ok_ssa = abs(float(np.mean(x2s)) - ALPHA / (ALPHA + MU)) < (0.08 if args.quick else 0.03)
    x2n = [r["x2"] for r in rows_d]
    ok_net = abs(float(np.mean(x2n)) - ALPHA / (ALPHA + MU)) < (0.15 if args.quick else 0.08)
    print("=" * 72)
    verdicts = [("mean-field = Theorem 2", ok_ode), ("SSA -> frozen inventory", ok_ssa), ("network -> frozen inventory", ok_net)]
    for name, ok in verdicts:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("=" * 72)
    return 0 if all(ok for _, ok in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
