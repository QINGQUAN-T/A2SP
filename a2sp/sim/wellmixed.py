"""Exact Gillespie SSA for the two-stage Maki-Thompson cascade (well-mixed).

Channels (population N, rates per ordered pair; cf. paper A eqs (1a)-(1e)):

  recruit     (S, I)   -> I becomes fresh S1           rate k1
  activate    (S1, S') -> S1 promoted to senior S2     rate alpha*k1
  stifle      (S1, S') -> S1 retires to R (fresh exit) rate mu*k1
  retire      (S2, S') -> S2 retires to R (senior exit) rate mu*k1  (=k2)

Spreader-spreader channels carry the pair self-correction (S_count - 1)
factor: the teller must be another spreader. The activate/stifle pair shares
the same ordered pairs and is drawn as one packaged clock with outcome
probabilities alpha/gamma and mu/gamma, gamma = alpha + mu (paper A eq. 1b).

Terminal history-resolved stifler composition (Theorem 2, paper A):
(x1, x2) = (mu, alpha)/(alpha + mu), independent of the seeding size.
"""

from __future__ import annotations

import random


def gillespie_wellmixed(
    alpha: float,
    mu: float,
    N: int = 60_000,
    k1: float = 1.0,
    eps: float = 1e-3,
    seed: int = 0,
    max_events: int | None = None,
) -> dict:
    """Run one well-mixed episode.

    Parameters
    ----------
    alpha : re-activation rate factor (channel rate alpha*k1 per ordered pair)
    mu    : stifling-to-recruitment ratio (k2 = mu*k1)
    N     : population size
    eps   : initial seed fraction (eps*N fresh spreaders S1 at t = 0)
    Returns a dict with terminal composition, reach, event count and clock.
    Raises ValueError on non-positive rates (audit H2 fix) or seeding
    fraction outside (0, 1) (audit M1 fix).
    """
    alpha = float(alpha)
    mu = float(mu)
    if not (alpha > 0.0 and mu > 0.0):
        raise ValueError(
            "require alpha > 0 and mu > 0: alpha = 0 removes the "
            "re-activation channel, mu = 0 removes stifling (the cascade "
            "never terminates), negative rates are unphysical (audit H2)"
        )
    if not (k1 > 0.0):
        raise ValueError("k1 must be positive")
    if not (0.0 < eps < 1.0):
        raise ValueError("eps (initial spreader fraction) must lie in (0, 1) (audit M1)")
    if max_events is None:
        max_events = 40 * N
    rng = random.Random(seed)
    gamma = alpha + mu

    status = [0] * N  # 0=I, 1=S1(fresh), 2=S2(senior), 3=R(stifled)
    pools: list[list[int]] = [[], [], []]  # members of class I, S1, S2
    pos = [-1] * N

    def _add(node: int, c: int) -> None:
        pos[node] = len(pools[c])
        pools[c].append(node)

    def _remove(node: int, c: int) -> None:
        lst = pools[c]
        j = pos[node]
        last = lst[-1]
        lst[j] = last
        pos[last] = j
        lst.pop()
        pos[node] = -1

    n_seed = max(1, int(round(eps * N)))
    if n_seed >= N:
        raise ValueError("seed pool must be strictly smaller than N")
    seeds = rng.sample(range(N), n_seed)
    for v in range(N):  # all nodes start ignorant
        _add(v, 0)
    for v in seeds:
        _remove(v, 0)
        status[v] = 1
        _add(v, 1)

    r1 = r2 = 0  # stifled-while-fresh / retired-while-senior
    t = 0.0
    events = 0
    while True:
        nI, nS1, nS2 = len(pools[0]), len(pools[1]), len(pools[2])
        nS = nS1 + nS2
        if nS == 0 or events >= max_events:
            break
        a_rec = k1 * nS * nI / N
        a_s1 = k1 * gamma * nS1 * (nS - 1) / N
        a_ret = mu * k1 * nS2 * (nS - 1) / N
        a0 = a_rec + a_s1 + a_ret
        if a0 <= 0.0:
            break
        t += rng.expovariate(a0)
        u = rng.random() * a0
        events += 1
        if u < a_rec:
            v = pools[0][rng.randrange(nI)]
            _remove(v, 0)
            status[v] = 1
            _add(v, 1)
        elif u < a_rec + a_s1:
            v = pools[1][rng.randrange(nS1)]
            if rng.random() < alpha / gamma:
                _remove(v, 1)
                status[v] = 2
                _add(v, 2)
            else:
                _remove(v, 1)
                status[v] = 3
                r1 += 1
        else:
            v = pools[2][rng.randrange(nS2)]
            _remove(v, 2)
            status[v] = 3
            r2 += 1

    nI = len(pools[0])
    exits = r1 + r2
    truncated = events >= max_events  # audit M2 fix: never silently mid-run
    return {
        "alpha": alpha,
        "mu": mu,
        "N": N,
        "eps": eps,
        "seed": seed,
        "x1": r1 / exits if exits else float("nan"),
        "x2": r2 / exits if exits else float("nan"),
        "reach": 1.0 - nI / N,
        "events": events,
        "t": t,
        "saturated": nI == 0,
        "truncated": truncated,
        "theory_x1": mu / gamma,
        "theory_x2": alpha / gamma,
    }
