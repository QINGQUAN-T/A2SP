"""The A2SP reduced trajectory law: dY/dX = -1 + kappa*Y/X.

Structural origin (paper B, unified theory of generalized step-growth
polymerization): the two-stage pairing channel carries a common *carrier
factor* (the partner-pool size). In the trajectory equations dY/dX that
factor cancels *exactly* — not perturbatively — leaving a closed,
fit-free law in the dimensionless asymmetry parameter(s) alone.

Instances of the same law across families (cf. the symer kernel README):

    polymer, growth stage   kappa = alpha/2                (b1 vs b0)
    rumor, merged layer     kappa = mu                     (S vs i)
    rumor, senior layer     kappa = gamma = alpha + mu     (s1 vs i)

The resonance line kappa = 1 separates the two asymptotic regimes. It is
handled by an exact logarithmic branch AND — for kappa != 1 — by an
expm1-based evaluation that is uniformly stable all the way into the
near-resonance neighbourhood |kappa - 1| ~ 1e-13 (the naive power form
    C = (y0 + x0/(1-kappa)) / x0**kappa ;  Y = C X**kappa - X/(1-kappa)
suffers catastrophic cancellation there: two ~1e13 terms cancel to ~1e-1,
losing ~7 significant digits — audit finding S3, fixed here).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["reduced_solution", "reduced_rk4", "residual"]


def reduced_solution(x, kappa: float, x0: float, y0: float):
    """Closed-form solution of dY/dX = -1 + kappa*Y/X through (x0, y0).

    kappa == 1 (exact resonance, log branch):
        Y(X) = X * (y0/x0 - ln(X/x0))
    kappa != 1 (uniformly stable evaluation):
        Y(X) = y0 (X/x0)**kappa + X * expm1((kappa-1) ln(X/x0)) / (1 - kappa)

    The second form is algebraically identical to the textbook power form
    C X**kappa - X/(1-kappa) but cancels nothing: as kappa -> 1 the expm1
    term tends to X * (-ln(X/x0)), i.e. the log branch, smoothly and to
    full double precision (audit S3 fix).

    Parameters
    ----------
    x : float, numpy scalar, or array-like of floats
        Points at which to evaluate Y(X). Strictly positive.
    Returns
    -------
    float for scalar/0-d input; numpy ndarray for array input.
    """
    kappa = float(kappa)
    x0 = float(x0)
    y0 = float(y0)
    if not (math.isfinite(kappa) and kappa > 0.0):
        raise ValueError("kappa must be a positive finite number")
    if not (math.isfinite(x0) and x0 > 0.0) or y0 < 0.0 or not math.isfinite(y0):
        raise ValueError("require x0 > 0 finite and y0 >= 0 finite")

    if isinstance(x, np.ndarray) and x.ndim == 0:
        return reduced_solution(float(x), kappa, x0, y0)  # 0-d array -> scalar (audit M4)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return _reduced_one(float(x), kappa, x0, y0)
    xa = np.asarray(x, dtype=float)
    if np.any(xa <= 0.0):
        raise ValueError("x must be strictly positive")
    L = np.log(xa / x0)
    if kappa == 1.0:
        return xa * (y0 / x0 - L)
    z = (kappa - 1.0) * L
    with np.errstate(over="ignore", invalid="ignore"):
        term_near = xa * np.expm1(z) / (1.0 - kappa)
        term_far = (np.exp(z + np.log(xa)) - xa) / (1.0 - kappa)
        term = np.where(np.abs(z) <= 700.0, term_near, term_far)
    y = y0 * (xa / x0) ** kappa + term
    if not np.all(np.isfinite(y)):
        # audit round 2, N4: a true overflow must not come back as a silent
        # nan — the scalar path raises, so must the array path.
        bad = int(np.size(y) - np.count_nonzero(np.isfinite(y)))
        raise OverflowError(
            f"reduced_solution overflowed the double range for {bad} point(s) "
            f"(kappa={kappa:g}, x0={x0:g}); reduce the domain or rescale inputs"
        )
    return y


def _expm1_term(xi: float, kappa: float, L: float) -> float:
    """X * expm1((kappa-1)*L) / (1-kappa), overflow-safe (audit follow-up).

    For |z| = |(kappa-1) L| <= 700 the plain expm1 form is exact and safe.
    Beyond that, exp(z) overflows even when the true product X*exp(z) is
    finite; factoring X into the exponent,
        X * (exp(z) - 1) = exp(z + ln X) - X,
    is finite whenever the result is finite, and its cancellation error is
    O(eps) because |exp(z)| >> 1 in that regime.
    """
    z = (kappa - 1.0) * L
    if abs(z) <= 700.0:
        return xi * math.expm1(z) / (1.0 - kappa)
    return (math.exp(z + math.log(xi)) - xi) / (1.0 - kappa)


def _reduced_one(xi: float, kappa: float, x0: float, y0: float) -> float:
    if xi <= 0.0 or not math.isfinite(xi):
        raise ValueError("x must be strictly positive and finite")
    L = math.log(xi / x0)
    if kappa == 1.0:
        y = xi * (y0 / x0 - L)
    else:
        try:
            y = y0 * (xi / x0) ** kappa + _expm1_term(xi, kappa, L)
        except OverflowError as exc:
            raise OverflowError(
                f"reduced_solution overflowed at x={xi:g} for kappa={kappa:g}: {exc}"
            ) from None
    if not math.isfinite(y):
        # audit round 2, N4: same semantics as the array path — a true
        # overflow raises instead of returning a silent nan (0 * inf).
        raise OverflowError(
            f"reduced_solution overflowed at x={xi:g} for kappa={kappa:g} "
            "(result not finite); reduce the domain or rescale inputs"
        )
    return y


def reduced_rk4(kappa: float, x0: float, y0: float, x_end: float, n: int = 20000):
    """Classical RK4 integration of dY/dX = -1 + kappa*Y/X from x0 to x_end.

    Used to verify that the closed form above and direct numerical
    integration agree to tight tolerance along whole trajectories (the
    carrier-cancellation residual check of the toolkit's acceptance list).
    """
    kappa = float(kappa)
    x0 = float(x0)
    if not (math.isfinite(kappa) and kappa > 0.0):
        raise ValueError("kappa must be a positive finite number")  # audit M3
    if not (math.isfinite(x0) and x0 > 0.0) or y0 < 0.0:
        raise ValueError("require x0 > 0 finite and y0 >= 0")  # audit M3
    if n <= 0:
        raise ValueError("n must be a positive integer")
    h = (x_end - x0) / n
    x, y = x0, y0
    for _ in range(n):
        k1 = -1.0 + kappa * y / x
        k2 = -1.0 + kappa * (y + 0.5 * h * k1) / (x + 0.5 * h)
        k3 = -1.0 + kappa * (y + 0.5 * h * k2) / (x + 0.5 * h)
        k4 = -1.0 + kappa * (y + h * k3) / (x + h)
        y += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        x += h
    return y


def residual(kappa: float, x0: float, y0: float, x_end: float, n: int = 20000) -> float:
    """Absolute difference between RK4 integration and the closed form."""
    y_num = reduced_rk4(kappa, x0, y0, x_end, n)
    y_ana = reduced_solution(x_end, kappa, x0, y0)
    return abs(y_num - y_ana)
