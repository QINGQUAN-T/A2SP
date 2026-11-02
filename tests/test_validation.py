"""Regression tests for the v0.1.0 audit findings (S1-S4, H1-H3, M1-M6, L1-L3).

Every test here corresponds to a defect reported in the external audit of
2026-11-02 and its fix. Expected behaviours:

S1  unreachable pB must raise, not fabricate a weight average
S2  arrest equation must validate alpha and bracket signs
S3  reduced law must be stable at |kappa - 1| ~ 1e-13 (expm1 form)
S4  terminal composition: exact quadrature + adaptive ODE (no 1e-4 truncation)
H1  configuration_model(mean_degree=0) must raise, not hang
H2  simulation engines must reject non-positive rates
H3  integrate(n_steps=0) must raise; negative n_steps must raise
M1  eps outside (0, 1) must raise
M2  truncated runs must be flagged, never silently returned as terminal
M3  reduced_rk4 must validate kappa and x0
M4  reduced_solution must handle numpy scalars / 0-d arrays
M6  terminal_arrest must guard the critical line r -> 1
L1  record_every logic must behave for every input
L3  __version__ must match the release version
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

try:
    import a2sp  # noqa: F401  (prefer the installed package, as in the other suites)
except ImportError:  # dev checkout / zero-install usage
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from a2sp import __version__  # noqa: E402
from a2sp.core import integrate, reduced_rk4, reduced_solution  # noqa: E402
from a2sp.models import polymer as poly  # noqa: E402
from a2sp.models import rumor as rum  # noqa: E402
from a2sp.sim import network as netsim  # noqa: E402
from a2sp.sim import wellmixed as wm  # noqa: E402


# ------------------------------------------------------------------- S1


def test_s1_unreachable_pB_raises():
    try:
        poly.xw_all(0.9, 0.8, 1e-5)
    except ValueError as e:
        assert "not reachable" in str(e)
    else:
        raise AssertionError("unreachable pB must raise ValueError")


def test_s1_reachable_pB_unchanged():
    # the published anchor must be untouched by the validation logic
    assert abs(poly.xw_all(0.8, 0.45, 1.0) - 721.0 / 239.0) < 1e-12


# ------------------------------------------------------------------- S2


def test_s2_arrest_rejects_alpha_le_0():
    for a in (0.0, -1.0, -5.0):
        try:
            poly.arrest_b0f(a, 0.9)
        except ValueError:
            continue
        raise AssertionError(f"alpha={a} must raise ValueError")


def test_s2_arrest_rejects_bad_r():
    for r in (0.0, 1.0, 1.2, -0.5):
        try:
            poly.arrest_b0f(3.0, r)
        except ValueError:
            continue
        raise AssertionError(f"r={r} must raise ValueError")


def test_s2_arrest_anchor_unchanged():
    st = poly.terminal_arrest(3.0, 0.9)
    assert abs(st["b0f"] - 0.05676) < 5e-6
    assert abs(st["b1f"] - 0.08648) < 1e-5


# ------------------------------------------------------------------- S3


def test_s3_near_resonance_stable():
    truth = -0.05 * math.log(0.05)  # log-branch limit value
    for kappa in (1.0 + 1e-13, 1.0 - 1e-13, 1.0 + 1e-9, 1.0 - 1e-9):
        y = float(reduced_solution(0.05, kappa, 1.0, 0.0))
        # the true power-branch value deviates from the log limit by O(|kappa-1|)
        assert abs(y - truth) < 5e-9, (kappa, y, truth)
    # far from resonance the classic anchor value is unchanged
    assert abs(float(reduced_solution(0.3025, 0.5, 1.0, 0.0)) - 0.495) < 1e-12


# ------------------------------------------------------------------- S4


def test_s4_exact_quadrature_machine_precision():
    for alpha, mu in ((2.0, 1.0), (0.5, 2.0), (0.3, 0.5)):
        d = rum.frozen_inventory_exact(alpha, mu)
        assert abs(d["x2"] - d["theory_x2"]) < 1e-14
        assert d["quad_abs_err"] < 1e-6


def test_s4_adaptive_ode_converges():
    # the old fixed-t_end version truncated at |x2 - theory| ~ 2.5e-4 here
    d = rum.terminal_composition_ode(0.3, 0.5)
    assert d["converged"], d
    assert abs(d["x2"] - d["theory_x2"]) < 1e-9, d
    d2 = rum.terminal_composition_ode(0.5, 2.0)
    assert d2["converged"] and abs(d2["x2"] - d2["theory_x2"]) < 1e-9


# ------------------------------------------------------------------- H1


def test_h1_zero_mean_degree_raises():
    try:
        netsim.configuration_model(100, 0.0)
    except ValueError as e:
        assert "mean_degree" in str(e)
    else:
        raise AssertionError("mean_degree=0 must raise, not hang")


def test_h1_negative_mean_degree_raises():
    try:
        netsim.configuration_model(100, -2.0)
    except ValueError:
        return
    raise AssertionError("negative mean_degree must raise")


# ------------------------------------------------------------------- H2


def test_h2_wellmixed_rejects_zero_and_negative_rates():
    for a, m in ((0.0, 0.0), (-1.0, 1.0), (2.0, 0.0), (0.0, 1.0)):
        try:
            wm.gillespie_wellmixed(a, m, N=200, eps=0.05)
        except ValueError:
            continue
        raise AssertionError(f"({a}, {m}) must raise ValueError")


def test_h2_network_rejects_zero_and_negative_rates():
    adj = netsim.configuration_model(100, 8.0, seed=1)
    for a, m in ((0.0, 0.0), (-1.0, 1.0)):
        try:
            netsim.network_ctmc(adj, a, m, eps=0.05)
        except ValueError:
            continue
        raise AssertionError(f"({a}, {m}) must raise ValueError")


# ------------------------------------------------------------------- H3


def test_h3_integrate_validates_steps():
    for n in (0, -5):
        try:
            integrate(lambda t, y: y, [1.0], 0.0, 1.0, n)
        except ValueError:
            continue
        raise AssertionError(f"n_steps={n} must raise ValueError")


def test_h3_integrate_record_every():
    ts, ys = integrate(lambda t, y: [t], [0.0], 0.0, 1.0, 10, record_every=3)
    assert len(ts) == 5  # steps 0,3,6,9 + final point
    assert abs(ts[-1] - 1.0) < 1e-12


# ------------------------------------------------------------------- M1


def test_m1_eps_validation():
    for eps in (0.0, -0.1, 1.0, 1.5):
        try:
            wm.gillespie_wellmixed(2.0, 1.0, N=200, eps=eps)
        except ValueError:
            continue
        raise AssertionError(f"eps={eps} must raise ValueError")


# ------------------------------------------------------------------- M2


def test_m2_truncation_is_flagged():
    r = wm.gillespie_wellmixed(2.0, 1.0, N=2000, eps=1e-3, seed=3, max_events=50)
    assert r["truncated"] is True
    r_full = wm.gillespie_wellmixed(2.0, 1.0, N=2000, eps=1e-3, seed=3)
    assert r_full["truncated"] is False
    assert r_full["saturated"] is True


# ------------------------------------------------------------------- M3


def test_m3_reduced_rk4_validates():
    try:
        reduced_rk4(-1.0, 1.0, 0.0, 0.5, 100)
    except ValueError:
        return
    raise AssertionError("kappa < 0 must raise ValueError")


# ------------------------------------------------------------------- M4


def test_m4_numpy_inputs():
    # kappa = 2 (power branch, NOT the resonance line) through a numpy integer:
    # Y = C X^2 - X/(1-2) with C = -1  ->  Y(5) = -25 + 5 = -20
    y = float(reduced_solution(np.int64(5), 2.0, 1.0, 0.0))
    assert abs(y - (-20.0)) < 1e-12
    # 0-d array must behave as a scalar and return a python float
    # kappa = 0.5: Y = 2 sqrt(X) - 2X -> Y(0.25) = 1 - 0.5 = 0.5
    y0 = reduced_solution(np.array(0.25), 0.5, 1.0, 0.0)
    assert isinstance(y0, float) and abs(y0 - 0.5) < 1e-12
    # true arrays return ndarrays with correct values
    arr = reduced_solution(np.array([0.25, 0.04]), 0.5, 1.0, 0.0)
    assert isinstance(arr, np.ndarray) and arr.shape == (2,)
    assert abs(arr[0] - 0.5) < 1e-12
    assert abs(arr[1] - 2.0 * (0.2 - 0.04)) < 1e-12


# ------------------------------------------------------------------- M6


def test_m6_critical_line_guard():
    try:
        poly.terminal_arrest(3.0, 1.0 - 1e-14)
    except ValueError as e:
        assert "critical" in str(e) or "diverg" in str(e)
    else:
        raise AssertionError("r -> 1 must be guarded")


# ------------------------------------------------------------------- L3


def test_l3_version_consistency():
    assert __version__ == "0.1.0"


# ------------------------------------------------------------------- M5/L2


def test_s3_residual_far_from_resonance():
    # the carrier-cancellation residual must stay at round-off after the fix
    from a2sp.core import residual

    for kappa in (0.25, 1.5, 0.999, 1.001):
        assert residual(kappa, 1.0, 0.0, 0.05, 20000) < 1e-9, kappa


# ------------------------------------------------- audit round 2 (N1-N6)


def test_n1_richardson_derivative_check():
    # proper (4*D(h/2) - D(h))/3 must reach ~1e-10 (was ~1e-5 before the fix)
    for alpha, mu in ((0.3, 0.5), (2.0, 1.0), (0.5, 0.5)):
        d = rum.frozen_inventory_exact(alpha, mu)
        assert d["deriv_check_err"] < 1e-8, (alpha, mu, d["deriv_check_err"])


def test_n2_no_trapz_deprecation_warning():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rum.frozen_inventory_exact(2.0, 1.0)
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not dep, [str(w.message) for w in dep]


def test_n3_pbmax_cached():
    assert hasattr(poly._pB_max, "cache_clear"), "_pB_max must be lru_cached"
    assert poly._pB_max(1.0) == poly._pB_max(1.0)


def test_n4_overflow_consistent_across_paths():
    # scalar and array paths must BOTH raise on true overflow (was: silent nan)
    for args in ((1e300, 3.0, 1e-300, 0.0),):
        try:
            reduced_solution(*args)
        except OverflowError:
            pass
        else:
            raise AssertionError("scalar path must raise OverflowError")
        try:
            reduced_solution(np.array([args[0]]), *args[1:])
        except OverflowError:
            pass
        else:
            raise AssertionError("array path must raise OverflowError")


def test_n5_reach_key_present():
    d = rum.terminal_composition_ode(2.0, 1.0)
    assert "reach" in d, "v0.1.0 API: 'reach' key must be present"
    assert 0.0 < d["reach"] <= 1.0 and d["reach"] > 0.99  # saturated case


def test_n6_extrapolation_exact_ratio():
    # exact-ratio weights must still converge and stay accurate
    d = rum.terminal_composition_ode(0.3, 0.5)
    assert d["converged"] and abs(d["x2"] - d["theory_x2"]) < 1e-9


if __name__ == "__main__":
    fails = 0
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"{len(fns) - fails}/{len(fns)} validation tests PASS")
    raise SystemExit(1 if fails else 0)
