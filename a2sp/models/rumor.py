"""Rumor family: the first application of the A2SP kernel (paper A).

Two-stage Maki-Thompson cascade with asymmetric re-activation: a contact
received by a fresh spreader resolves into re-activation (rate alpha*k1,
creating a "senior" spreader) or stifling (rate k2 = mu*k1). All formulas
are transcribed from

    Q. Tang, "Terminal-state theorems for a two-stage Maki-Thompson rumor
    model: final size, stifler composition, and non-identifiability from
    pooled statistics" (arXiv preprint; number pending)

and cross-checked against the symer kernel README (frozen inventory law,
tail trichotomy, carrier-factor cancellation rows).
"""

from __future__ import annotations

import numpy as np

from ..core import reduced_solution

# numpy 2.x renamed trapz -> trapezoid (deprecated alias); keep both worlds
# working without emitting DeprecationWarnings (audit round 2, N2).
_trapezoid = getattr(np, "trapezoid", None)
if _trapezoid is None:
    _trapezoid = np.trapz

# ----------------------------------------------------------------------
# Terminal-state laws (exact, fit-free)
# ----------------------------------------------------------------------


def frozen_inventory(alpha: float, mu: float) -> tuple[float, float]:
    """Terminal history-resolved composition of the spreader pool:

        (x1, x2) = (mu, alpha) / (alpha + mu),

    independent of the seeding size (Theorem 2 of paper A; frozen
    inventory law). x1 = fresh-stifled share, x2 = re-activated share.
    """
    if alpha <= 0 or mu <= 0:
        raise ValueError("require alpha > 0 and mu > 0")
    s = alpha + mu
    return (mu / s, alpha / s)


def fingerprint_invert(x1: float, x2: float, mu: float) -> float:
    """Linear inversion of the asymmetry from a terminal ledger:

        alpha = mu * x2 / x1   (exact; no calibration).
    """
    if x1 <= 0 or mu <= 0:
        raise ValueError("require x1 > 0 and mu > 0")
    return mu * x2 / x1


# ----------------------------------------------------------------------
# Trajectory layers (carrier S cancels exactly in dY/dX)
# ----------------------------------------------------------------------


def merged_layer(i, mu: float, i0: float, S0: float):
    """Merged spreader pool S(i), kappa = mu:

        dS/di = -1 + mu * S / i.

    Tail trichotomy (Law I): mu < 1 spreader-led power-law branch,
    mu = 1 logarithmically critical, mu > 1 co-decay with S/i -> 1/(mu-1).
    """
    return reduced_solution(i, mu, i0, S0)


def senior_layer(i, alpha: float, mu: float, i0: float, s1_0: float):
    """Senior (re-activated) spreader fraction s1(i), kappa = gamma = alpha+mu:

        ds1/di = -1 + gamma * s1 / i.

    Resonance line gamma = 1 is the logarithmic branch (no branch split at
    gamma = 1 in the terminal theorems; the proof is uniform in parameters).
    """
    return reduced_solution(i, alpha + mu, i0, s1_0)


# ----------------------------------------------------------------------
# Mean-field ODEs (paper A eqs (1a)-(1e)) and terminal composition
# ----------------------------------------------------------------------


def mean_field_rhs(t, y, alpha: float, mu: float, k1: float = 1.0):
    """Right-hand side of the population-normalized mean field (N = 1).

    State y = (i, s1, s2, r1, r2): ignorant, fresh spreaders, senior
    spreaders, stifled-while-fresh, retired-while-senior. S = s1 + s2;
    k2 = mu*k1; gamma = alpha + mu.

        (1a) di/dt  = -k1 i S
        (1b) ds1/dt = k1 S i - k1 gamma S s1
        (1c) ds2/dt = alpha k1 S s1 - k2 S s2
        (1d) dS/dt  = k1 S i - k2 S^2          (follows from (1a)-(1c))
        (1e) dr/dt  = k2 S^2                    (split here into r1 + r2)
    """
    i, s1, s2, _, _ = y
    S = s1 + s2
    di = -k1 * i * S
    ds1 = k1 * S * i - k1 * (alpha + mu) * S * s1
    ds2 = alpha * k1 * S * s1 - mu * k1 * S * s2
    dr1 = mu * k1 * S * s1
    dr2 = mu * k1 * S * s2
    return [di, ds1, ds2, dr1, dr2]


def frozen_inventory_exact(alpha: float, mu: float, quad_points: int = 200_001) -> dict:
    """Exact terminal composition by layer quadrature — no time integration.

    With k1 factored out and i(t) monotonically decreasing (di/dt = -iS < 0
    while S > 0), the change of variable dt = -di/(i S) removes the carrier
    from the stifler-pool flows exactly:

        r1 = mu * INT_0^{i0} s1(i)/i di ,      r1 + r2 = 1   (saturation, mu > 0),

    where s1(i) is the closed-form senior layer through (i0, eps). The
    integrand is a sum of powers, so the integral is elementary and
    evaluates to EXACTLY 1/gamma with the seeding size eps cancelling:

        INT_0^{i0} s1(i)/i di = (eps + i0)/gamma = 1/gamma ,
        r1 = mu/gamma ,  r2 = 1 - r1 = alpha/gamma ,
        (x1, x2) = (mu, alpha)/(alpha + mu)   (Theorem 2, paper A),

    i.e. the frozen-inventory law is an identity of the closed-form layers,
    to machine precision, for every seeding size (audit S4 fix: replaces the
    truncation-limited fixed-t_end integration as the exact route).

    ``quad_abs_err`` reports an independent numeric cross-check of the
    quadrature identity with two complementary pieces (audit follow-up):
    (a) the exact elementary antiderivative F(i) differentiates back to the
    integrand s1(i)/i (central-difference check at interior sample points),
    and (b) a trapezoid rule on the smooth interior [0.1 i0, i0] plus the
    exact antiderivative tail [0, 0.1 i0] reproduces 1/gamma — this stays
    machine-accurate for every gamma > 0, including gamma < 1 where the
    integrand has an integrable power-law singularity at i -> 0.
    """
    if not (alpha > 0.0 and mu > 0.0):
        raise ValueError("require alpha > 0 and mu > 0")
    gamma = alpha + mu
    eps = 1e-9
    i0 = 1.0 - eps

    def antiderivative(i):
        """F(i) = INT_0^i s1(s)/s ds (elementary, exact up to round-off)."""
        i = np.asarray(i, dtype=float)
        if abs(gamma - 1.0) < 1e-12:
            return eps * i / i0 + i * (1.0 - np.log(i / i0))
        return (
            eps * i**gamma / (gamma * i0**gamma)
            + i0 * ((i / i0) ** gamma / gamma - i / i0) / (1.0 - gamma)
        )

    # (a) antiderivative differentiates back to the integrand — proper
    # Richardson extrapolation (4*D(h/2) - D(h))/3 of central differences
    # (audit round 2, N1: the previous combination mixed O(1) and O(h) terms)
    probe = np.array([0.2 * i0, 0.5 * i0, 0.9 * i0])
    h = 1e-5 * i0
    d_h = (antiderivative(probe + h) - antiderivative(probe - h)) / (2.0 * h)
    d_h2 = (antiderivative(probe + h / 2.0) - antiderivative(probe - h / 2.0)) / h
    dF_num = (4.0 * d_h2 - d_h) / 3.0
    dF_ana = np.asarray(senior_layer(probe, alpha, mu, i0, eps), dtype=float) / probe
    deriv_err = float(np.max(np.abs(dF_num - dF_ana)))

    # (b) interior trapezoid + exact tail reproduces 1/gamma
    split = 0.1 * i0
    grid = np.linspace(split, i0, quad_points)
    s1 = np.asarray(senior_layer(grid, alpha, mu, i0, eps), dtype=float)
    integral = float(_trapezoid(s1 / grid, grid)) + float(antiderivative(split))

    return {
        "x1": mu / gamma,
        "x2": alpha / gamma,
        "theory_x1": mu / gamma,
        "theory_x2": alpha / gamma,
        "quad_identity_value": integral,
        "quad_identity_target": 1.0 / gamma,
        "quad_abs_err": abs(integral - 1.0 / gamma),
        "deriv_check_err": deriv_err,
    }


def terminal_composition_ode(
    alpha: float,
    mu: float,
    eps: float = 1e-6,
    t_end: float = 2000.0,
    n: int = 40_000,
    rtol: float = 1e-10,
    max_rounds: int = 14,
) -> dict:
    """Terminal composition from direct ODE integration, adaptively.

    Numeric confirmation of Theorem 2: integrates eqs (1a)-(1e) from the
    seeding (i, s1, s2) = (1-eps, eps, 0) with chunk-doubling continuation
    and Richardson extrapolation on the ~1/T tail error (audit S4 fix — a
    single fixed-t_end run truncates with |x2 - theory| ~ S_end ~ 1e-4).

    The chunk schedule gives cumulative horizons T_k = (2^k - 1) t_end, so
    consecutive horizons are NOT in an exact 1:2 ratio; the extrapolation
    therefore uses the exact-ratio weights (audit round 2, N6):

        x* = x(T_k) + (x(T_k) - x(T_{k-1})) * r/(1-r),   r = T_{k-1}/T_k,

    which is exact for the c/T tail model at any horizon ratio. ``converged``
    is set when two successive extrapolations agree within ``rtol``; if
    ``max_rounds`` is exhausted first, the raw last-round values are
    returned with ``converged=False`` (never silently passed as terminal).

    Returns the extrapolated (x1, x2), the raw last-round values, ``reach``
    (= 1 - i_end; kept for v0.1.0 API compatibility — audit round 2, N5),
    the ``converged`` flag and the tail-error estimate.
    """
    from ..core import integrate as _int

    if not (alpha > 0.0 and mu > 0.0):
        raise ValueError("require alpha > 0 and mu > 0")

    def rhs(t, y):
        return mean_field_rhs(t, y, alpha, mu)

    y = np.array([1.0 - eps, eps, 0.0, 0.0, 0.0], dtype=float)
    t = 0.0
    rounds: list[tuple[float, float, float, float]] = []  # (T_cum, x1, x2, S_end)
    converged = False
    x1 = x2 = float("nan")
    tail_estimate = float("nan")
    prev_ext_x2 = None
    for _ in range(max_rounds):
        _, Y = _int(rhs, y, t, t + t_end, n)
        t += t_end
        y = Y[-1]
        r1, r2 = y[3], y[4]
        tot = r1 + r2
        x1, x2 = r1 / tot, r2 / tot
        rounds.append((t, x1, x2, float(y[1] + y[2])))
        if len(rounds) >= 2:
            (T1, x1a, x2a, _), (T2, x1b, x2b, _) = rounds[-2], rounds[-1]
            w = (T1 / T2) / (1.0 - T1 / T2)  # exact-ratio weight, = T1/(T2-T1)
            ext_x2 = x2b + (x2b - x2a) * w
            ext_x1 = x1b + (x1b - x1a) * w
            if prev_ext_x2 is not None:
                tail_estimate = abs(ext_x2 - prev_ext_x2)
                if tail_estimate < rtol:
                    converged = True
                    x1, x2 = ext_x1, ext_x2
                    break
            prev_ext_x2 = ext_x2
        t_end *= 2.0
    return {
        "x1": x1,
        "x2": x2,
        "x1_raw": rounds[-1][1],
        "x2_raw": rounds[-1][2],
        "reach": 1.0 - float(y[0]),  # audit round 2, N5: v0.1.0 API compatibility
        "converged": converged,
        "tail_estimate": tail_estimate,
        "rounds": len(rounds),
        "t_final": t,
        "S_end": rounds[-1][3],
        "theory_x1": mu / (alpha + mu),
        "theory_x2": alpha / (alpha + mu),
    }
