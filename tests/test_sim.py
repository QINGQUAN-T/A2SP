"""Fast stochastic smoke tests (SSA + network) against the frozen inventory law."""

from __future__ import annotations

import pathlib
import sys

try:
    import a2sp  # noqa: F401  (prefer the installed package)
except ImportError:  # dev checkout / zero-install usage
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from a2sp.sim import network as netsim  # noqa: E402
from a2sp.sim import wellmixed as wm  # noqa: E402


def test_ssa_frozen_inventory() -> None:
    r = wm.gillespie_wellmixed(2.0, 1.0, N=4_000, eps=1e-3, seed=11)
    assert r["saturated"], "mu = 1 must saturate the ignorant pool"
    assert abs(r["x2"] - 2.0 / 3.0) < 0.06, r
    assert r["reach"] > 0.90, r


def test_ssa_second_point() -> None:
    r = wm.gillespie_wellmixed(0.5, 2.0, N=4_000, eps=1e-3, seed=12)
    assert abs(r["x2"] - 0.5 / 2.5) < 0.05, r


def test_network_smoke() -> None:
    adj = netsim.configuration_model(600, mean_degree=8.0, seed=5)
    r = netsim.network_ctmc(adj, 2.0, 1.0, eps=0.01, seed=7)
    assert r["events"] > 0 and 0.0 < r["reach"] <= 1.0
    assert abs(r["x2"] - 2.0 / 3.0) < 0.15, r


if __name__ == "__main__":
    test_ssa_frozen_inventory()
    test_ssa_second_point()
    test_network_smoke()
    print("3/3 simulation smoke tests PASS")
