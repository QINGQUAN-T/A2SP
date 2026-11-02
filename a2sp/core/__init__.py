"""Core of the A2SP framework: the reduced trajectory law.

Every A2SP system (polymerization, rumor spreading, or a new domain) reduces,
after the exact cancellation of the common carrier factor, to the same
one-parameter family of trajectory equations

    dY/dX = -1 + kappa * Y / X ,   kappa > 0,

with the resonance line kappa = 1 (logarithmic branch). This module holds
that law, its closed-form solution, a numerical integrator used to verify
the cancellation to round-off, and a small ODE workhorse.
"""

from .reduced import reduced_solution, reduced_rk4, residual
from .integrate import integrate

__all__ = ["reduced_solution", "reduced_rk4", "residual", "integrate"]
