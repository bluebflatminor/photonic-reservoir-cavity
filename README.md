# Shape-Encoded Photonic Reservoir

Cavity geometry as a computational substrate for photonic reservoir computing.
A public-domain research seed — proposal, open questions, explicit kill
conditions, and a toy numerical check.

**Status:** Public domain · **License:** [CC0](LICENSE) · No permission needed to use, fork, or publish.

---

## What this is

Most photonic reservoir computing uses one-dimensional delay-line
architectures, where computational richness is borrowed from time-multiplexing
rather than from physical structure. This seed proposes the opposite: that the
**three-dimensional geometry of an optical cavity is itself a computational
resource**. A plano-hemispherical cavity in a wave-chaotic regime supports a
dense, high-dimensional field whose interference patterns form a natural
reservoir state space; a conformal monolayer-graphene "skin" reads the field
out via evanescent leakage.

The proposal states its own primary failure mode up front (readout SNR under
spatial correlation) and includes explicit falsification thresholds.

## Contents

| File | What it is |
|------|------------|
| `cavity_rc_proposal.html` | The one-page proposal. Open in any browser. Self-contained: typography, a procedurally-drawn schematic, and the full argument. |
| `reff_correlation_toy.py` | A toy numerical check of the proposal's §2.1 risk logic. Requires only `numpy`. |
| `LICENSE` | CC0 1.0 Universal (public domain dedication). |

## The toy model (`reff_correlation_toy.py`)

**This is not a physical simulation.** It is a deliberately minimal,
one-dimensional caricature whose only purpose is to test whether the
absorption-vs-rank tension described in proposal §2.1 is internally coherent,
and to reveal which variable dominates the effective reservoir rank
(`r_eff`). It produces **no physically meaningful `r_eff` number** — real
numbers require full FDTD (e.g. MEEP) on the true 2.5D geometry.

### What it found

1. **The tension is robust.** Across every spectral assumption tested, widening
   a patch's acceptance window raised signal (SNR) while lowering `r_eff` —
   exactly the trade §2.1 describes.
2. **Spectral flatness dominates, not patch geometry.** When the cavity field's
   energy was concentrated in a few modes, `r_eff` collapsed to ≈2–3 regardless
   of how patches were arranged. When spread evenly, patches recovered ≈90% of
   the channel ceiling. The "shape → capacity" question (proposal H1) is
   therefore **upstream** of the readout question and must be settled first.

The script's header also documents a hidden-assumption error caught in its own
first draft (an arbitrary spectral roll-off that pre-determined the rank). It is
left documented rather than quietly fixed — which is the point of a toy check.

### Running it

Requires Python 3 and `numpy`. No other dependencies; no network, no file I/O.

```bash
python3 reff_correlation_toy.py
```

It prints three tables (one per spectral roll-off) and a one-line verdict each.
The random seed is fixed, so output is fully reproducible. Runs in seconds on a
laptop; runs in minutes on a phone (e.g. Pyto / Pythonista on iOS).

## What's still open

The clearest unfilled gap is a real **FDTD model** (MEEP or equivalent) of the
cavity geometry that computes the inter-patch correlation matrix and estimates
achievable `r_eff` against the §2.1 risk — establishing in simulation whether
the readout can clear the `r_eff > 50` threshold before any fabrication. That
model does not yet exist. Anyone is welcome to build it.

Other open questions (proposal §4): the shape→capacity mapping (H1), graphene
skin optimisation (H2), frequency-comb multiplexing limits (H3), and the
fabrication kill conditions (H4).

## How to use this

There is nothing to join and no one to ask. Take any of it, fork it, build on
it, publish it, ignore the source. The only aim is to move the question forward.

## Method note

Developed transparently with AI tools used for research, drafting, and
adversarial review. Numerical results in this repository were reproduced by
hand before publication.
