"""Tests for filling long gaps in an orbit file with a fitted orbit.

A cubic spline has no idea what an orbit looks like, so across a gap of tens of minutes
it is wrong by kilometres; ``GapFilledInterpolator`` fits a physical orbit there instead.
"""

import os

import numpy as np
from scipy.integrate import solve_ivp

from barycenter.gapfill import GM, GapFilledInterpolator, find_gaps, j2_acceleration
from barycenter.native import spacecraft_interpolator
from barycenter.orbit import read_orbit

curdir = os.path.abspath(os.path.dirname(__file__))
GBM_ORBIT = os.path.join(curdir, "data", "dummy_gbm_poshist.fits.gz")


def leo_orbit(times):
    """A circular-ish 500 km orbit integrated under J2, so its true shape is known."""
    r0 = 6.9e6
    v0 = np.sqrt(GM / r0)
    state = [r0, 0.0, 0.0, 0.0, v0 * np.cos(np.radians(26)), v0 * np.sin(np.radians(26))]
    sol = solve_ivp(
        lambda _, y: np.concatenate([y[3:], j2_acceleration(y[:3])]),
        (times[0], times[-1]),
        state,
        t_eval=times,
        rtol=1e-11,
        atol=1e-6,
        method="DOP853",
    )
    return sol.y[:3].T


def test_find_gaps_reports_only_holes_longer_than_the_threshold():
    """A 400 s hole is a gap and a 60 s one is not, at the default 150 s threshold."""
    met = np.concatenate([np.arange(0, 1000, 30.0), np.arange(1400, 2000, 30.0)])
    met = met[(met < 1500) | (met > 1560)]
    gaps = find_gaps(met, min_gap=150.0)
    assert gaps.shape == (1, 2)
    assert gaps[0, 1] - gaps[0, 0] > 150.0
    assert gaps[0, 0] == 990.0 and gaps[0, 1] == 1400.0


def test_times_outside_gaps_are_exactly_the_plain_spline():
    """Away from the gaps nothing changes, so existing results stay bit-identical."""
    met = np.arange(0, 12000, 30.0)
    pos = leo_orbit(met)
    keep = (met < 5000) | (met > 6200)
    filled = GapFilledInterpolator(met[keep], pos[keep])
    plain = spacecraft_interpolator(met[keep], pos[keep])
    probe = np.array([100.0, 4975.0, 6300.0, 11000.0])
    np.testing.assert_array_equal(filled(probe), plain(probe))


def test_a_gap_in_an_exact_j2_orbit_is_filled_to_a_metre():
    """With data that follow the fitted law the fit is limited only by the optimiser."""
    met = np.arange(0, 12000, 30.0)
    pos = leo_orbit(met)
    keep = (met < 5000) | (met > 6200)
    filled = GapFilledInterpolator(met[keep], pos[keep])
    probe = np.arange(5030.0, 6200.0, 10.0)
    truth = leo_orbit(np.concatenate([[0.0], probe]))[1:]
    assert np.max(np.linalg.norm(filled(probe) - truth, axis=1)) < 1.0


def test_real_gbm_gap_beats_the_spline_by_orders_of_magnitude():
    """On real data a 10 minute hole is ~4.6 km wrong with a spline; the fit keeps it
    within a few hundred metres (1 us of light time is 300 m)."""
    table = read_orbit(GBM_ORBIT)
    met = np.asarray(table["MET"].value)
    pos = np.stack([np.asarray(table[c].value) for c in "XYZ"], axis=1)
    thin = np.arange(len(met)) % 30 == 0
    start = met[0] + 1200.0
    hole = (met > start) & (met < start + 600.0)
    keep = thin & ~hole
    filled = GapFilledInterpolator(met[keep], pos[keep])
    plain = spacecraft_interpolator(met[keep], pos[keep])
    err_fit = np.linalg.norm(filled(met[hole]) - pos[hole], axis=1).max()
    err_spline = np.linalg.norm(plain(met[hole]) - pos[hole], axis=1).max()
    assert err_fit < 300.0
    assert err_spline > 10 * err_fit
    assert len(filled.gaps) == 1
