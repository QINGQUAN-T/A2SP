"""Configuration-model network CTMC for the two-stage Maki-Thompson cascade.

Per-ordered-contact channel rates, matching the well-mixed convention of
paper A eqs (1a)-(1e):

  recruit : every ordered (spreader -> ignorant) contact fires I -> S1 at k1
  receive on an S1 endpoint: every spreader->S1 contact fires at the packaged
            rate k1*gamma, resolving to activation (S1 -> S2, prob alpha/gamma)
            or stifling (S1 -> R, prob mu/gamma)
  receive on an S2 endpoint: every spreader->S2 contact fires S2 -> R at k2 = mu*k1

The teller always survives (catalytic contact). The terminal history-resolved
stifler composition (x1, x2) = (mu, alpha)/(alpha + mu) is a theorem of the
mean field and is approached by the network engine at large N (paper A Sec. 5).
"""

from __future__ import annotations

import random

import numpy as np


def configuration_model(N: int, mean_degree: float, seed: int = 0) -> list[list[int]]:
    """Undirected configuration model with Poisson degrees.

    Self-loops and multi-edges produced by stub matching are discarded (the
    standard slight degree reduction); returns adjacency lists.
    Raises ValueError for N < 2 or mean_degree <= 0 (audit H1 fix — a
    non-positive mean degree previously spun in an infinite loop).
    """
    if N < 2:
        raise ValueError("N must be at least 2")
    if not (mean_degree > 0.0):
        raise ValueError(
            "mean_degree must be positive: a Poisson(0) degree sequence has "
            "no edges and no even-sum redraw can ever succeed (audit H1)"
        )
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    for _ in range(100):
        deg = nprng.poisson(mean_degree, N)
        if deg.sum() % 2 == 0 and deg.sum() > 0:
            break
    else:
        raise RuntimeError(
            "could not draw a valid Poisson degree sequence after 100 tries"
        )
    stubs = np.repeat(np.arange(N), deg)
    nprng.shuffle(stubs)
    stubs = [int(x) for x in stubs]
    adj: list[set[int]] = [set() for _ in range(N)]
    while len(stubs) >= 2:
        u = stubs.pop()
        v = stubs.pop()
        if u == v or v in adj[u]:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return [sorted(s) for s in adj]


class _Pool:
    """O(1) amortized random-draw pool of ordered pairs with lazy deletion."""

    __slots__ = ("items", "valid", "npos", "n_valid")

    def __init__(self) -> None:
        self.items: list[tuple[int, int]] = []
        self.valid: list[bool] = []
        self.npos: dict[tuple[int, int], int] = {}
        self.n_valid = 0

    def add(self, x: tuple[int, int]) -> None:
        if x in self.npos:
            return
        self.npos[x] = len(self.items)
        self.items.append(x)
        self.valid.append(True)
        self.n_valid += 1

    def discard(self, x: tuple[int, int]) -> None:
        j = self.npos.pop(x, None)
        if j is None:
            return
        self.valid[j] = False
        self.n_valid -= 1
        if self.n_valid * 2 < len(self.items):
            self.items = [it for it, ok in zip(self.items, self.valid) if ok]
            self.valid = [True] * len(self.items)
            self.npos = {it: i for i, it in enumerate(self.items)}

    def draw(self, rng: random.Random) -> tuple[int, int]:
        while True:
            j = rng.randrange(len(self.items))
            if self.valid[j]:
                return self.items[j]


def _edge_entries(su: int, sv: int, u: int, v: int):
    """Ordered (receiver, teller, pool) entries contributed by edge (u, v).

    status: 0=I, 1=S1(fresh), 2=S2(senior), 3=R(stifled). An edge yields a
    rec entry when one endpoint is ignorant and the other a spreader; a
    fresh-receive entry when one endpoint is S1 and the other any spreader;
    a senior-receive entry when one endpoint is S2 and the other any spreader.
    """
    out = []
    if su == 0 and sv in (1, 2):
        out.append((u, v, 0))
    if sv == 0 and su in (1, 2):
        out.append((v, u, 0))
    if su == 1 and sv in (1, 2):
        out.append((u, v, 1))
    if sv == 1 and su in (1, 2):
        out.append((v, u, 1))
    if su == 2 and sv in (1, 2):
        out.append((u, v, 2))
    if sv == 2 and su in (1, 2):
        out.append((v, u, 2))
    return out


def network_ctmc(
    adj: list[list[int]],
    alpha: float,
    mu: float,
    k1: float = 1.0,
    eps: float = 1e-3,
    seed: int = 0,
    max_events: int | None = None,
) -> dict:
    """Event-driven CTMC of the cascade on a fixed network.

    Raises ValueError on non-positive rates (audit H2 fix) or seeding
    fraction outside (0, 1) (audit M1 fix). Returns a ``truncated`` flag
    when the event budget was hit before the pools drained (audit M2 fix).
    """
    N = len(adj)
    if N < 1:
        raise ValueError("adjacency list must be non-empty")
    alpha = float(alpha)
    mu = float(mu)
    if not (alpha > 0.0 and mu > 0.0):
        raise ValueError(
            "require alpha > 0 and mu > 0 (see a2sp.sim.wellmixed: "
            "alpha = 0 or mu = 0 degenerates the two-step channel; audit H2)"
        )
    if not (k1 > 0.0):
        raise ValueError("k1 must be positive")
    if not (0.0 < eps < 1.0):
        raise ValueError("eps (initial spreader fraction) must lie in (0, 1) (audit M1)")
    if max_events is None:
        max_events = 200 * N
    rng = random.Random(seed)
    gamma = alpha + mu

    status = [0] * N  # 0=I, 1=S1, 2=S2, 3=R
    pools = [_Pool(), _Pool(), _Pool()]  # 0=recruit, 1=fresh-receive, 2=senior-receive

    def _flip_entries(node: int, old: int, new: int) -> None:
        if old == new:
            return
        for v in adj[node]:
            for (a, b, p) in _edge_entries(old, status[v], node, v):
                pools[p].discard((a, b))
            for (a, b, p) in _edge_entries(new, status[v], node, v):
                pools[p].add((a, b))

    n_seed = max(1, int(round(eps * N)))
    if n_seed >= N:
        raise ValueError("seed pool must be strictly smaller than N")
    for v in rng.sample(range(N), n_seed):
        _flip_entries(v, status[v], 1)
        status[v] = 1

    r1 = r2 = 0
    t = 0.0
    events = 0
    while True:
        a_rec = k1 * pools[0].n_valid
        a_s1 = k1 * gamma * pools[1].n_valid
        a_ret = mu * k1 * pools[2].n_valid
        a0 = a_rec + a_s1 + a_ret
        if a0 <= 0.0 or events >= max_events:
            break
        t += rng.expovariate(a0)
        u = rng.random() * a0
        events += 1
        if u < a_rec:
            v = pools[0].draw(rng)[0]
            _flip_entries(v, status[v], 1)
            status[v] = 1
        elif u < a_rec + a_s1:
            v = pools[1].draw(rng)[0]
            new = 2 if rng.random() < alpha / gamma else 3
            old = status[v]
            if new == 3:
                r1 += 1
            _flip_entries(v, old, new)
            status[v] = new
        else:
            v = pools[2].draw(rng)[0]
            _flip_entries(v, status[v], 3)
            status[v] = 3
            r2 += 1

    nI = sum(1 for s in status if s == 0)
    exits = r1 + r2
    truncated = events >= max_events  # audit M2 fix
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
        "truncated": truncated,
        "theory_x1": mu / gamma,
        "theory_x2": alpha / gamma,
    }
