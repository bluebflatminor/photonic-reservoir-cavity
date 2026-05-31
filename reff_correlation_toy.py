"""
Toy model: does pushing per-patch absorption up actually collapse r_eff
via spatial correlation between graphene skin patches?

This is NOT a physical FDTD simulation. It is a deliberately simple,
falsifiable test of the LOGIC in proposal section 2.1:

    "Increasing per-patch absorption -- longer interaction length,
     cavity-field enhancement -- tends to increase the spatial
     correlation between neighbouring patches ... r_eff collapses."

If the tension does NOT appear even in a generous toy model, the 2.1
framing is wrong and should be revised before publication.

-----------------------------------------------------------------------------
WHAT THIS MODEL CAN AND CANNOT TELL YOU
-----------------------------------------------------------------------------
CAN:  test whether the absorption-vs-rank tension is logically coherent,
      and reveal which variable dominates r_eff.
CANNOT: produce a real r_eff number. It is a 1D rim-sampling caricature
      with hand-built sinusoidal modes, arbitrary noise units, and an
      assumed (not derived) link between absorption and angular window
      width. Real numbers require FDTD (MEEP) on the true 2.5D geometry.

SELF-CAUGHT ARTIFACT (left documented on purpose):
      An earlier version hard-coded a 1/k mode-amplitude roll-off. That
      alone made the field nearly rank-2 BEFORE any patch sampling, so the
      low r_eff it reported had nothing to do with patch correlation -- it
      was measuring an arbitrary spectral choice. The fix was to expose the
      spectral roll-off as its own parameter (SPEC_ROLLOFF) and report the
      intrinsic field rank as the true ceiling. This is exactly the kind of
      hidden-assumption error a toy model exists to surface.

KEY FINDING:
      The 2.1 tension is robust (it appears at every roll-off). But the
      DOMINANT variable is the field's own spectral flatness, not the patch
      geometry -- i.e. the H1 "shape -> capacity" question is UPSTREAM of
      the 2.1 readout question. Confirm the cavity field is high-rank before
      optimising the skin that reads it.
-----------------------------------------------------------------------------

Model sketch
------------
- The cavity interior carries a wave-chaotic field. We represent the
  field sampled along the hemisphere rim as a random superposition of
  M spatial modes (random Gaussian mode amplitudes), evaluated at the
  angular positions of N detector patches.
- Each patch integrates the field over an angular acceptance window
  whose width grows with the "absorption strength" knob `a`. Physically:
  to absorb more, a patch couples to a longer arc / larger evanescent
  footprint, so it averages over a wider angular span.
- Wider integration window  ->  more signal (good)  but also more
  OVERLAP with neighbouring patches' windows  ->  higher correlation
  (bad for rank).
- We build the N x N correlation matrix of patch responses across many
  random field realizations (frequency steps / comb teeth), then compute
  the effective rank.

Effective rank (participation-ratio definition, continuous & robust):
    r_eff = (sum_i lambda_i)^2 / sum_i (lambda_i^2)
where lambda_i are eigenvalues of the patch covariance matrix.
This equals N for perfectly uncorrelated equal-variance channels and
drops toward 1 as channels become redundant.

Released CC0 alongside the proposal. Take it, fork it, ignore the source.
"""

import numpy as np

rng = np.random.default_rng(42)

# ---- Geometry / sampling ---------------------------------------------------
N_PATCHES = 32          # detector patches along the rim (proposal: ~20-40)
M_MODES   = 200         # spatial modes in the chaotic field superposition
N_FREQ    = 400         # independent field realizations (comb teeth / sweeps)

# Patch angular centres, evenly spaced over the hemisphere rim [0, pi]
theta = np.linspace(0, np.pi, N_PATCHES, endpoint=False) + (np.pi / N_PATCHES) / 2

# A fixed set of mode angular "wavenumbers" -- higher modes oscillate faster
# around the rim. This is what makes a wider integration window average
# them out (low-pass), reducing high-mode content and increasing redundancy.
mode_k = np.linspace(1, M_MODES, M_MODES)


def patch_window(absorption):
    """Angular half-width of a patch's acceptance window as a function of the
    absorption knob. absorption in [0,1]; 0 => near-delta sampling (point
    detector, weak absorption), 1 => window spans ~2 inter-patch spacings
    (strong absorption, large footprint)."""
    spacing = np.pi / N_PATCHES
    return spacing * (0.15 + 1.85 * absorption)  # 0.15 .. 2.0 of a spacing


def sample_responses(absorption, n_int=41):
    """Return an (N_FREQ, N_PATCHES) matrix of patch responses.
    Each frequency realization draws fresh random mode amplitudes (the
    chaotic field changes with input frequency). Each patch integrates the
    field over its angular window via simple quadrature."""
    half = patch_window(absorption)
    # integration sample offsets within each window
    offs = np.linspace(-half, half, n_int)
    w = np.hanning(n_int)            # smooth window weighting
    w = w / w.sum()

    # Precompute the sampling kernel: for each patch, the angles sampled
    # angles shape (N_PATCHES, n_int)
    ang = theta[:, None] + offs[None, :]

    # Field basis evaluated at those angles: sin(mode_k * angle)
    # basis shape (N_PATCHES, n_int, M_MODES)
    basis = np.sin(np.einsum('pi,m->pim', ang, mode_k))

    # Window-integrate the basis -> effective per-patch mode sensitivity
    # eff shape (N_PATCHES, M_MODES)
    eff = np.einsum('pim,i->pm', basis, w)

    # Random mode amplitudes per frequency realization
    # amps shape (N_FREQ, M_MODES); spectral roll-off controlled by SPEC_ROLLOFF
    amps = rng.standard_normal((N_FREQ, M_MODES)) / (mode_k[None, :] ** SPEC_ROLLOFF)

    # Patch responses: (N_FREQ, N_PATCHES)
    R = amps @ eff.T

    # Add detector noise. Signal grows with window size (more captured power),
    # so model signal amplitude ~ sqrt(window) and FIXED additive noise floor.
    signal_gain = np.sqrt(patch_window(absorption) / patch_window(0.0))
    R = R * signal_gain
    noise = rng.standard_normal(R.shape) * NOISE_FLOOR
    return R + noise


def effective_rank(R):
    """Participation-ratio effective rank of the patch covariance."""
    Rc = R - R.mean(axis=0, keepdims=True)
    cov = (Rc.T @ Rc) / (R.shape[0] - 1)
    ev = np.linalg.eigvalsh(cov)
    ev = ev[ev > 0]
    return (ev.sum() ** 2) / (np.sum(ev ** 2))


def mean_snr(R):
    """Crude SNR proxy: signal std over assumed noise floor, averaged."""
    sig = R.std(axis=0).mean()
    return sig / NOISE_FLOOR


# ---- Sweep the absorption knob across several spectral roll-offs ----------
NOISE_FLOOR = 0.02   # fixed additive detector noise (arbitrary units)

def intrinsic_field_rank():
    """Effective rank of the field's mode-amplitude covariance, BEFORE any
    patch sampling. This is the true upper bound on recoverable r_eff."""
    amps = rng.standard_normal((N_FREQ, M_MODES)) / (mode_k[None, :] ** SPEC_ROLLOFF)
    cov = np.cov(amps.T)
    ev = np.linalg.eigvalsh(cov); ev = ev[ev > 0]
    return (ev.sum() ** 2) / np.sum(ev ** 2)

for SPEC_ROLLOFF in (0.0, 0.5, 1.0):
    field_rank = intrinsic_field_rank()
    print(f"\n### spectral roll-off 1/k^{SPEC_ROLLOFF}   "
          f"(intrinsic field r_eff = {field_rank:.1f}, "
          f"patch ceiling = {N_PATCHES})")
    print(f"{'absorption':>10} {'window':>8} {'mean_SNR':>9} {'r_eff':>7}")
    print("-" * 40)

    results = []
    for a in np.linspace(0.0, 1.0, 11):
        R = sample_responses(a)
        re = effective_rank(R)
        snr = mean_snr(R)
        win = patch_window(a) / (np.pi / N_PATCHES)
        results.append((a, win, snr, re))
        print(f"{a:>10.2f} {win:>8.2f} {snr:>9.1f} {re:>7.2f}")

    re_low, re_high = results[0][3], results[-1][3]
    snr_low, snr_high = results[0][2], results[-1][2]
    tension = (snr_high > snr_low) and (re_high < re_low)
    best = max(results, key=lambda r: r[3])
    print(f"  tension present: {tension}   "
          f"best r_eff={best[3]:.1f} @ a={best[0]:.2f} "
          f"({best[3]/N_PATCHES:.0%} of patch ceiling)")
