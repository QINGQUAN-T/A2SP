"""Small fixed-step RK4 integrator (numpy-based) for the toolkit's ODE demos.

Note (audit M5): fixed-step RK4 has no automatic error control. Where a
quantitative accuracy claim is made (e.g. rumor.terminal_composition_ode),
the toolkit validates convergence explicitly by step/halving or continuation
with Richardson extrapolation instead of trusting a single run.
"""

from __future__ import annotations

import numpy as np


def integrate(rhs, y0, t0: float, t_end: float, n_steps: int, record_every: int = 1):
    """Integrate dy/dt = rhs(t, y) with classic RK4.

    Parameters
    ----------
    rhs : callable(t, y) -> array-like
    y0 : initial state (array-like)
    n_steps : number of steps; must be a positive integer (audit H3 fix)
    record_every : record every k-th step (1 = every step)
    Returns (t_array, y_array) with shape (n_records,) and (n_records, dim),
    always including the final point.
    """
    if not isinstance(n_steps, (int, np.integer)) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")  # audit H3 fix
    y = np.array(y0, dtype=float)
    h = (t_end - t0) / n_steps
    ts, ys = [], []
    t = t0
    step_rec = max(1, int(record_every))  # audit L1 fix (dead branch removed)
    for k in range(n_steps):
        if k % step_rec == 0:
            ts.append(t)
            ys.append(y.copy())
        k1 = np.asarray(rhs(t, y), dtype=float)
        k2 = np.asarray(rhs(t + 0.5 * h, y + 0.5 * h * k1), dtype=float)
        k3 = np.asarray(rhs(t + 0.5 * h, y + 0.5 * h * k2), dtype=float)
        k4 = np.asarray(rhs(t + h, y + h * k3), dtype=float)
        y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t = t0 + (k + 1) * h
    ts.append(t)
    ys.append(y.copy())
    return np.array(ts), np.array(ys)
