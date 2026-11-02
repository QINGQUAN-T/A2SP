"""Polymer family: the kernel source (paper B).

Closed-form laws of generalized AA + BB step-growth polymerization with
kinetic asymmetry alpha = k2/k (two-site kinetic scheme). All formulas are
transcribed from the manuscript / Supporting Information of

    Q. Tang, "A Unified Theory for Generalized Step-Growth Polymerization:
    From Flory's Equal-Reactivity to Kinetic Symmetry-Breaking"

and cross-validated against the regression anchors published in the symer
kernel README (github.com/QINGQUAN-T/symer, Zenodo 10.5281/zenodo.23282847).

State variables (fractions of BB units): b0 intact, b1 half-reacted (B*),
b2 = 1 - b0 - b1 fully reacted; pB = b1/2 + b2 = 1 - b0 - b1/2;
pA = pB / r (stoichiometric equivalence); feed ratio r = N_A0/N_B0.

Terminology — "arrest" vs "termination": for r < 1 the reaction stops by
carrier exhaustion (sigma = [A]/[BB]_0 -> 0): every channel rate vanishes
simultaneously and a *resumable* frozen non-equilibrium state is trapped,
retaining a residual activated inventory (b0f, b1f > 0). Following the
kernel paper this is called (kinetic) arrest — as in dynamical/colloidal
arrest and arrested thermoset curing. It is NOT a chain-growth-style
"termination": no termination reaction exists, no active center is
destroyed (re-fuel with A-monomer and the reaction resumes), and
"termination" carries no established meaning in step-growth kinetics.
"Terminate(s)" is used only as an ordinary verb.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from ..core import reduced_solution

# ----------------------------------------------------------------------
# Trajectory law (carrier [A] cancels exactly)
# ----------------------------------------------------------------------


def b1_trajectory(b0, alpha: float):
    """Activated-fraction trajectory b1(b0; alpha), IC b1(1) = 0.

    Reduced law db1/db0 = -1 + (alpha/2) * b1 / b0, i.e. the A2SP kernel
    with kappa = alpha/2:

        b1 = 2 (b0**(alpha/2) - b0) / (2 - alpha)   (alpha != 2)
        b1 = -b0 ln b0                              (alpha == 2, log branch)
    """
    return reduced_solution(b0, alpha / 2.0, 1.0, 0.0)


def pB_of(b0: float, b1: float) -> float:
    """B-group conversion pB = b1/2 + b2 = 1 - b0 - b1/2 (SI eq. S7.3)."""
    return 1.0 - b0 - b1 / 2.0


# ----------------------------------------------------------------------
# Averages
# ----------------------------------------------------------------------


def carothers_xn(r: float, pB: float) -> float:
    """All-species number-average DP, Eq. (9):  Xn = (1+r)/(1+r-2 pB)."""
    if 1.0 + r - 2.0 * pB <= 0:
        raise ValueError("requires 1 + r - 2 pB > 0")
    return (1.0 + r) / (1.0 + r - 2.0 * pB)


def _pB_max(alpha: float) -> float:
    """Largest conversion pB reachable on the b1(b0; alpha) trajectory.

    Mathematically pB -> 1 only in the limit b0 -> 0 over ~1/alpha decades,
    which is unreachable in floating point for small alpha (b0**alpha/2
    saturates at ~1 for every representable b0). The practical maximum is
    obtained by scanning b0 on a dense log grid down to the denormal range.
    (Audit S1 fix: requests beyond this maximum now raise instead of
    silently returning a fabricated weight average.)

    Cached per alpha (audit round 2, N3: the 4096-point scan is pure
    recomputation on repeated calls otherwise).
    """
    b = np.logspace(-323, 0, 4096)
    pB = 1.0 - b - b1_trajectory(b, alpha) / 2.0
    return float(np.max(pB))


_pB_max = lru_cache(maxsize=None)(_pB_max)


def _solve_b0(pB: float, alpha: float, tol: float = 1e-15, max_iter: int = 200) -> float:
    """Invert pB = 1 - b0 - b1(b0; alpha)/2 for b0 by bisection on (0, 1).

    Raises ValueError if pB exceeds the reachability maximum of the
    trajectory (audit S1 fix) or if the bracket degenerates.
    """
    if not (math.isfinite(alpha) and alpha > 0.0):
        raise ValueError("alpha must be a positive finite number")
    pB_max = _pB_max(alpha)
    if pB > pB_max * (1.0 + 1e-9) + 1e-12:
        raise ValueError(
            f"pB={pB:g} is not reachable on the alpha={alpha:g} trajectory: "
            f"the maximum conversion along it is pB_max={pB_max:.6g} "
            "(small alpha caps the attainable conversion)."
        )

    def g(b: float) -> float:
        return 1.0 - b - b1_trajectory(b, alpha) / 2.0 - pB

    lo, hi = 1e-300, 1.0
    flo, fhi = g(lo), g(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if (flo < 0.0) == (fhi < 0.0):
        raise ValueError("bracket failure in _solve_b0 (defensive check)")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = g(mid)
        if fm == 0.0 or hi - lo < tol:
            return mid
        if (fm < 0.0) == (flo < 0.0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def xw_all(r: float, pB: float, alpha: float) -> float:
    """All-species weight-average DP by bidirectional walking, SI eq. (S7.7).

        Xw = 1 + 2 pB (3 + pA - qB) / ((1 + r) w),
        pA = pB/r,  qB = b1/(2 pB),  w = 1 - pA (1 - qB),

    with (b0, b1) taken on the trajectory b1(b0; alpha) at the given pB.
    """
    if r <= 0 or not 0.0 < pB < 1.0:
        raise ValueError("require r > 0 and 0 < pB < 1")
    b0 = _solve_b0(pB, alpha)
    b1 = b1_trajectory(b0, alpha)
    pA = pB / r
    qB = b1 / (2.0 * pB)
    w = 1.0 - pA * (1.0 - qB)
    if w <= 0:
        raise ValueError("walking denominator w must be positive")
    return 1.0 + 2.0 * pB * (3.0 + pA - qB) / ((1.0 + r) * w)


# ----------------------------------------------------------------------
# Terminal states (0 < r < 1 arrest; r > 1 exhaustion)
# ----------------------------------------------------------------------


def arrest_b0f(alpha: float, r: float, tol: float = 1e-15, max_iter: int = 200) -> float:
    """Terminal intact-BB fraction b0f for 0 < r < 1, arrest equation:

        (1 - alpha) b0f + b0f**(alpha/2) = (2 - alpha) (1 - r)   (Eqs. 13-17)

    alpha == 2 is a degenerate point of the closed form; obtain it as the
    numerical limit alpha -> 2 (handled by :func:`terminal_arrest`).
    """
    if not 0.0 < r < 1.0:
        raise ValueError("arrest regime requires 0 < r < 1")
    if not (math.isfinite(alpha) and alpha > 0.0):
        raise ValueError(
            "alpha must be a positive finite number: alpha <= 0 has no "
            "two-step channel and the arrest equation has no physical root "
            "(audit S2 fix)"
        )
    if abs(alpha - 2.0) < 1e-12:
        raise ValueError("alpha = 2 is degenerate here; use terminal_arrest")
    rhs = (2.0 - alpha) * (1.0 - r)

    def f(b: float) -> float:
        return (1.0 - alpha) * b + b ** (alpha / 2.0) - rhs

    lo, hi = 1e-300, 1.0
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if (flo < 0.0) == (fhi < 0.0):
        raise ValueError("arrest equation has no sign change on (0, 1) for these parameters (defensive check)")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0.0 or hi - lo < tol:
            return mid
        if (flo < 0.0) == (fm < 0.0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def terminal_arrest(alpha: float, r: float) -> dict:
    """Terminal observables for the A-deficient arrest regime (0 < r < 1).

    Returns b0f, b1f = 2(1 - r - b0f), the all-species terminal
    number-average Xn_f = (1 + r)/(1 - r)  (Eq. 16), and the terminal
    weight-average Xw_f = (16 r^2/b1f + 1 - r)/(1 + r)  (SI eq. S7.9).
    alpha = 2 is obtained as the numerical limit alpha -> 2.
    """
    eps = 1e-8
    if abs(alpha - 2.0) < 1e-12:
        b_lo = arrest_b0f(2.0 - eps, r)
        b_hi = arrest_b0f(2.0 + eps, r)
        b0f = 0.5 * (b_lo + b_hi)
    else:
        b0f = arrest_b0f(alpha, r)
    b1f = 2.0 * (1.0 - r - b0f)
    if b1f <= 1e-12:
        raise ValueError(
            f"r={r:g} is too close to the critical line r=1: the terminal "
            "activated inventory b1f vanishes and the terminal weight "
            "average Xw_f diverges (physical critical behaviour, audit M6)."
        )
    return {
        "b0f": b0f,
        "b1f": b1f,
        "xn_f_all": (1.0 + r) / (1.0 - r),
        "xw_f_all": (16.0 * r * r / b1f + 1.0 - r) / (1.0 + r),
    }


def xw_terminal_r_gt_1(r: float) -> float:
    """Terminal all-species weight average for r > 1, SI eq. (S7.8):

        Xw_f = (r^2 + 6r + 1) / ((r - 1)(r + 1)),

    obtained from (S7.7) with pB = 1, pA = 1/r, qB = 0, w = (r-1)/r.
    """
    if r <= 1.0:
        raise ValueError("requires r > 1")
    return (r * r + 6.0 * r + 1.0) / ((r - 1.0) * (r + 1.0))


# ----------------------------------------------------------------------
# Two-site kinetics in real time (carrier sigma included)
# ----------------------------------------------------------------------


def two_site_rhs(t, y, alpha: float, r: float, k: float = 1.0):
    """Two-site ODEs in reduced time tau = 2 k [BB]_0 t (paper B SI S1.x).

    State y = (b0, b1) with b2 = 1 - b0 - b1; pB = 1 - b0 - b1/2;
    sigma = [A]/[BB]_0 = 2 (r - pB).

        db0/dtau = -sigma * b0
        db1/dtau =  sigma * (b0 - (alpha/2) b1)

    Dividing db1/db0 cancels sigma exactly, recovering the reduced law
    d b1/d b0 = -1 + (alpha/2) b1 / b0 — the carrier-factor cancellation
    that defines the A2SP kernel.
    """
    b0, b1 = y
    pB = 1.0 - b0 - b1 / 2.0
    sigma = 2.0 * (r - pB)
    return [-sigma * b0, sigma * (b0 - 0.5 * alpha * b1)]
