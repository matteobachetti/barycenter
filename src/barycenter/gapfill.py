"""Filling long gaps in a spacecraft orbit file with a fitted orbit.

A cubic spline through the tabulated positions is excellent between samples (about 2 cm
at Fermi's 30 s cadence) and useless across a gap of tens of minutes, where it is wrong
by kilometres: a cubic polynomial cannot follow a 95 minute orbit. The Fermi LAT, for
instance, is switched off in the South Atlantic Anomaly and its spacecraft file simply
has no rows there.

:class:`GapFilledInterpolator` keeps the spline everywhere except inside such a gap and
there evaluates a *physical orbit* fitted to the samples either side of it: a position
and velocity at the start of a window, integrated numerically under point-mass gravity
plus the Earth's flattening (J2). Measured on a real day of Fermi GBM positions thinned
to 30 s (``tools/benchmarks/bench_gapfill.py``), the worst position error inside a 5 to
40 minute gap is of the order of 100 m, about 0.4 us of light time, independent of the
gap length. See "The error budget of a position" in the technical details.

Limits, none of which the benchmark has tested beyond the first:

* Only a low, near-circular Earth orbit was measured. The force model is Earth gravity
  with J2 only, so a highly elliptical orbit (where the Moon and Sun matter) is out of
  scope, and so is anything not Earth-centred.
* The positions must be in a frame whose z axis is the Earth's rotation axis and that
  does not rotate (J2000 equatorial); that is how every supported mission writes them.
  An Earth-fixed frame would be silently wrong.
* Drag and the higher gravity terms are not modelled; they are most of the 100 m left.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .native import spacecraft_interpolator

__all__ = [
    "GM",
    "GapFilledInterpolator",
    "MIN_GAP_S",
    "find_gaps",
    "fit_j2_orbit",
    "j2_acceleration",
]

#: Earth's gravitational parameter, m^3 / s^2 (EGM2008 / IERS 2010).
GM = 3.986004418e14
#: The Earth's second zonal harmonic, and its equatorial radius in metres.
J2 = 1.08262668e-3
RE = 6378137.0

#: Gaps shorter than this are left to the spline. Measured on real data, a spline across a
#: 120 s gap is off by about 20 m and across 180 s by about 70 m, while the fitted orbit
#: sits at 45-60 m whatever the gap: the two cross at about 150 s (five Fermi samples).
MIN_GAP_S = 150.0

#: The fit uses this many orbital periods of samples on each side of the gap. Half an
#: orbit was the best of 0.5, 1 and 2 in the benchmark: the unmodelled forces make a
#: longer window worse, not better.
WINDOW_ORBITS = 0.5

#: Weight of a position sample in the fit, metres. Only the relative weight matters.
_SIGMA_M = 10.0


def j2_acceleration(r):
    """Acceleration in m/s^2 at geocentric position ``r`` (m): point mass plus J2."""
    r = np.asarray(r, dtype=np.float64)
    rn = np.linalg.norm(r)
    z2 = (r[2] / rn) ** 2
    factor = 1.5 * J2 * GM * RE**2 / rn**5
    return -GM * r / rn**3 + factor * r * np.array([5 * z2 - 1, 5 * z2 - 1, 5 * z2 - 3])


def find_gaps(met, min_gap=MIN_GAP_S):
    """The holes between consecutive samples that are longer than ``min_gap`` seconds.

    Returns an array of shape ``(n, 2)`` holding, for each gap, the time of the last
    sample before it and of the first sample after it.
    """
    met = np.unique(np.asarray(met, dtype=np.float64))
    holes = np.nonzero(np.diff(met) > min_gap)[0]
    return np.stack([met[holes], met[holes + 1]], axis=1)


def fit_j2_orbit(met, position):
    """Fit an orbit to tabulated positions.

    Parameters
    ----------
    met : ndarray, shape (N,)
        Sorted sample times in seconds.
    position : ndarray, shape (N, 3)
        Geocentric positions in metres (J2000 equatorial).

    Returns
    -------
    predict : callable
        ``predict(t)`` gives positions, shape ``(len(t), 3)``, for ``t`` within the
        span of ``met``.
    rms : float
        Root mean square distance of the fit from the samples, in metres. It is a
        lower bound on the error inside a gap, not an estimate of it: the benchmark
        found it about half of the worst gap error.
    """
    t0 = met[0]
    x = met - t0

    def rhs(_, y):
        return np.concatenate([y[3:], j2_acceleration(y[:3])])

    def propagate(state, **kwargs):
        return solve_ivp(rhs, (0.0, x[-1]), state, rtol=1e-10, atol=1e-5, method="DOP853", **kwargs)

    scale = np.array([1e3] * 3 + [1.0] * 3)
    start = np.concatenate([position[0], (position[1] - position[0]) / (x[1] - x[0])])

    def residuals(s):
        return ((propagate(s * scale, t_eval=x).y[:3].T - position) / _SIGMA_M).ravel()

    fit = least_squares(residuals, start / scale, method="lm", xtol=1e-13, ftol=1e-13)
    state = fit.x * scale
    dense = propagate(state, dense_output=True, t_eval=x)
    rms = float(np.sqrt(np.mean(np.sum((dense.y[:3].T - position) ** 2, axis=1))))
    return (lambda t: dense.sol(np.asarray(t, dtype=np.float64) - t0)[:3].T), rms


def _period(met, position):
    """Orbital period from the two samples at the end of ``position``, in seconds."""
    r, v = position[-1], (position[-1] - position[-2]) / (met[-1] - met[-2])
    return 2 * np.pi * np.dot(r, r) / np.linalg.norm(np.cross(r, v))


class GapFilledInterpolator:
    """A spacecraft position function that does not guess across long gaps.

    Parameters
    ----------
    met, position, velocity
        As for :func:`barycenter.native.spacecraft_interpolator`. The velocity, if
        given, is used by the spline between samples but not by the orbit fit.
    min_gap : float
        Gaps longer than this many seconds are filled with a fitted orbit.

    Attributes
    ----------
    gaps : ndarray, shape (n, 2)
        Start and end of every gap that is filled this way.
    fit_rms : dict
        Gap index to the fit's root mean square residual in metres, for the gaps that
        have been evaluated so far.

    Notes
    -----
    Outside the gaps the answer is exactly that of the plain spline, so the existing
    results do not change. Each gap is fitted the first time it is asked about.
    """

    def __init__(self, met, position, velocity=None, min_gap=MIN_GAP_S):
        self.spline = spacecraft_interpolator(met, position, velocity)
        # The spline's knots are the samples that survived its cleaning.
        self.met = np.asarray(self.spline.x)
        self.position = self.spline(self.met)
        self.gaps = find_gaps(self.met, min_gap)
        self.fit_rms = {}
        self._fits = {}

    def _fit_gap(self, index):
        if index not in self._fits:
            before, after = self.gaps[index]
            i0 = np.searchsorted(self.met, before, side="right")  # first sample after
            window = WINDOW_ORBITS * _period(self.met[:i0], self.position[:i0])
            keep = ((self.met >= before - window) & (self.met <= before)) | (
                (self.met >= after) & (self.met <= after + window)
            )
            predict, rms = fit_j2_orbit(self.met[keep], self.position[keep])
            self._fits[index] = predict
            self.fit_rms[index] = rms
        return self._fits[index]

    def __call__(self, met):
        met = np.asarray(met, dtype=np.float64)
        scalar = met.ndim == 0
        met = np.atleast_1d(met)
        out = np.array(self.spline(met))
        for index, (before, after) in enumerate(self.gaps):
            inside = (met > before) & (met < after)
            if inside.any():
                out[inside] = self._fit_gap(index)(met[inside])
        return out[0] if scalar else out
