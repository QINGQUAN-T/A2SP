# A2SP — Asymmetric Two-Step Pairing

**One pairing kernel, any domain.**

`a2sp` is the reference implementation of the **Asymmetric Two-Step Pairing (A2SP)** framework — a minimal, exact structure for two-step processes in which every channel rate carries a common *carrier factor* (a partner pool: a fuel balance, a spreader pool, …) that cancels **exactly** in the reduced trajectory equations, leaving closed-form, fit-free laws in the asymmetry parameters alone.

- **Kernel source (theory):** Q. Tang, *A Unified Theory for Generalized Step-Growth Polymerization: From Flory's Equal-Reactivity to Kinetic Symmetry-Breaking* (JACS submission). The polymerization "two-site kinetic scheme" is the mathematical origin of the framework.
- **First application:** Q. Tang, *Terminal-state theorems for a two-stage Maki–Thompson rumor model: final size, stifler composition, and non-identifiability from pooled statistics* (preprint; **arXiv:2610.41206**).
- **Citation line:** *Reference implementation of the A2SP framework introduced in [arXiv:2610.41206]; kernel source: the step-growth polymerization unified theory (ChemRxiv 10.26434/chemrxiv.15009468/v1).*

The same kernel serves any domain whose dynamics fit the two-step pairing structure — rumor spreading, polymerization, and by extension fields such as finance or management science (see *Extend A2SP to your domain* below).

---

## The kernel in one equation

After the carrier factor cancels exactly, every A2SP trajectory obeys the one-parameter reduced law

$$\frac{dY}{dX} = -1 + \kappa\, \frac{Y}{X}, \qquad \kappa > 0,$$

with the **resonance line κ = 1** (exact logarithmic branch). Three instances of the same law:

| family | carrier | κ | resonance |
|---|---|---|---|
| step-growth polymerization (kernel source) | fuel balance [A] (consumed) | α/2 | α = 2 (log tail) |
| rumor, merged layer | spreader pool S (catalytic) | μ | μ = 1 (log-critical) |
| rumor, senior (re-activated) layer | spreader pool S | γ = α + μ | γ = 1 (log branch) |

Closed forms in this package: `b1(b0; α) = 2(b0^(α/2) − b0)/(2 − α)`, Carothers `X̄n = (1+r)/(1+r−2pB)`, terminal arrest `(1−α)b0f + b0f^(α/2) = (2−α)(1−r)`, frozen inventory `(x1, x2) = (μ, α)/(α+μ)`, fingerprint inversion `α = μ·x2/x1`, and more (see `a2sp/models`).

> **Terminology:** *arrest* (the kernel paper's term) denotes the stoichiometric freezing of growth when the limiting A-groups are exhausted (r < 1) — every channel rate vanishes at once, trapping a **resumable** frozen state with a residual activated inventory (b0f, b1f > 0). It is deliberately **not** called "termination": in step-growth kinetics there is no termination reaction and no active center is destroyed (re-adding A-monomer resumes the reaction); "termination" is chain-growth vocabulary for a mechanistic destruction event. "Terminate(s)" appears only as an ordinary verb.

## Install (≈ 2 minutes)

```bash
git clone https://github.com/QINGQUAN-T/A2SP.git 
cd A2SP
pip install -r requirements.txt   # everything: numpy + matplotlib (examples plot figures)
# or, library only (numpy alone; matplotlib is an optional extra):
pip install .
# (or `pip install a2sp` once on PyPI.)
```

## Quickstart — two examples, one command each

```bash
python examples/ex2_polymer.py     # kernel source: step-growth polymerization (~10 s)
python examples/ex1_rumor.py       # first application: rumor spreading (~40 s)
```

Each example prints validation tables against published numbers, saves figures to `examples/output/`, and writes a results JSON with every random seed recorded. Add `--quick` for a seconds-long smoke run.

**What "works in 30 minutes" means (measured on a clean machine):** install ≈ 2 min; `python run_tests.py` ≈ 1 min (53 regression tests); `ex2_polymer.py` ≈ 10 s; `ex1_rumor.py` ≈ 1 min including figures. No external data downloads, no compilation, no accounts.

## What the examples show

- `ex2_polymer.py` — 8/8 PASS: Carothers, Flory, the X̄w dual-path anchor (721/239 = 3.016736…), the frozen-state anchor (α=3, r=0.9), ODE↔closed-form arrest agreement, and the real-time two-site ODE lying on the reduced curve to ~1e-10 (the carrier cancellation, verified).
- `ex1_rumor.py` — 3/3 PASS: the frozen-inventory law (Theorem 2) is confirmed two independent ways — exact layer quadrature through the closed-form layers (agrees with the theorem to ~1e-12) and adaptive ODE integration with Richardson-validated convergence (<1e-9) — and well-mixed Gillespie SSA (N = 6×10⁴, pair self-correction) and a configuration-model network cascade (⟨k⟩ = 8) converge to the same terminal composition (x1, x2) = (μ, α)/(α+μ).

## Extend A2SP to your domain

If your system is a two-step pairing process, you need exactly two things: your **asymmetry parameters** and your **carrier kind** (consumed vs catalytic). The reduced law and the closed forms are already written for you:

```python
from a2sp.core import reduced_solution, residual

# 1) derive your reduced law dY/dX = -1 + kappa*Y/X from your channel rates
kappa = ...                     # your asymmetry combination (see a2sp/models for patterns)

# 2) closed-form trajectory through any state, and its round-off verification
Y = reduced_solution(X, kappa, X0, Y0)
dev = residual(kappa, X0, Y0, X_end)          # ~1e-11: exact cancellation, verified
```

The two model files (`a2sp/models/polymer.py`, `a2sp/models/rumor.py`) are worked templates: copy the nearest one, rename the parameters, and keep the same anchor-test discipline. If your carrier is consumed (like [A]), keep the trajectory ODEs in *reduced* form; if it is catalytic (like S), the same reduction applies — only the interpretation of X and Y changes.

## Validation

`python run_tests.py` runs all three suites:

- **23 published-anchor regressions** (`tests/test_anchors.py`): every expected value comes from the manuscripts' SI or the symer kernel README — never from this codebase. Largest deviation from any published anchor: 3.3×10⁻⁶ (5-significant-figure rounding of the published value); exact identities (Carothers, frozen inventory, fingerprint inversion) hold to machine precision.
- **Stochastic smoke tests** (`tests/test_sim.py`): SSA and network engines reproduce the frozen-inventory law within finite-N tolerance.
- **21 audit regressions** (`tests/test_validation.py`): every defect found by the 2026-11-02 external audit (4 correctness, 3 crash/hang, 7 robustness items) now has a dedicated regression test; inputs outside the valid domain raise `ValueError` instead of failing silently.

Provenance note: the anchors originate from the author's own manuscripts and the author's symer reproduction package — i.e., they are *self-published* cross-checks (formula-internal consistency, series-vs-walking agreement to 4×10⁻¹⁶), not yet independent third-party verification. Independent verification becomes possible once the papers are published.

Cross-validation target: [symer v0.1.3](https://github.com/QINGQUAN-T/symer) (Zenodo: 10.5281/zenodo.23282847) — the paper-specific reproduction package for the rumor paper. `symer` archives that paper's figures and data pipelines; `a2sp` is the portable framework core. They are independent and link to each other.

## Repository layout

```
a2sp/                 the package
  core/reduced.py     the reduced law dY/dX = -1 + κY/X (+ RK4 verification)
  core/integrate.py   small RK4 workhorse
  models/polymer.py   kernel source family (paper B closed forms)
  models/rumor.py     first application family (paper A closed forms + mean field)
  sim/wellmixed.py    exact Gillespie SSA, pair self-correction (S−1)
  sim/network.py      configuration-model CTMC
tests/                anchors + stochastic smoke + audit regressions
examples/             ex1_rumor.py, ex2_polymer.py (one command each)
PUBLISH.md            maintainer checklist (GitHub + Zenodo release chain)
.github/workflows/    CI (Python 3.9–3.12)
```

## How to cite

See `CITATION.cff`. Please cite the toolkit's Zenodo DOI (10.5281/zenodo.23883289) together with the theory paper (step-growth polymerization unified theory, ChemRxiv preprint above) and/or the first application (arXiv:2610.41206).

## License

MIT — see `LICENSE`.
