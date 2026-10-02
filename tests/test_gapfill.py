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


def test_coverage_counts_a_filled_gap_as_covered_but_not_an_unfilled_one():
    """Choosing to fill a gap is what makes its times acceptable, and nothing else."""
    from barycenter.orbit import OrbitCoverage

    met = np.concatenate([np.arange(0, 1000, 30.0), np.arange(2000, 3000, 30.0)])
    plain = OrbitCoverage.from_met(met)
    filled = OrbitCoverage.from_met(met, filled=find_gaps(met))
    assert plain.uncovered([1500.0])[0] > 100.0
    assert filled.uncovered([1500.0])[0] == 0.0
    assert filled.uncovered([5000.0])[0] > 100.0  # past the end: never filled


class TestFillOrbitGapsEndToEnd:
    """The command line, on real GBM data with ten minutes cut out of its orbit file."""

    ra, dec = "254.457625", "35.342361"

    def setup_method(self):
        self.evfile = os.path.join(curdir, "data", "dummy_gbm_evt.evt")

    def cut_orbit(self, tmp_path):
        from astropy.io import fits

        out = str(tmp_path / "gappy_poshist.fits")
        with fits.open(GBM_ORBIT) as hdul:
            t = hdul["GLAST POS HIST"].data["SCLK_UTC"]
            hole = (t > 732198700.0) & (t < 732199300.0)
            hdul["GLAST POS HIST"].data = hdul["GLAST POS HIST"].data[~hole]
            hdul.writeto(out)
        return out

    def run(self, orbfile, outfile, *extra):
        from barycenter.cli import main_barycenter

        return main_barycenter(
            [self.evfile, orbfile, "-o", outfile, "--ra", self.ra, "--dec", self.dec]
            + ["--ephem", "DE405", "--clockfile", "none", *extra]
        )

    def test_refused_by_default_with_a_hint_and_accepted_with_the_flag(self, tmp_path):
        """Without the flag the file is refused and the message names the switch; with it
        the times land within 1 us of those from the complete orbit file, and the header
        says which gap was filled."""
        import pytest
        from astropy.io import fits

        gappy = self.cut_orbit(tmp_path)
        with pytest.raises(ValueError, match="--fill-orbit-gaps"):
            self.run(gappy, str(tmp_path / "refused.evt"))

        filled = self.run(gappy, str(tmp_path / "filled.evt"), "--fill-orbit-gaps")
        whole = self.run(GBM_ORBIT, str(tmp_path / "whole.evt"))
        with fits.open(filled) as a, fits.open(whole) as b:
            diff = a["EVENTS"].data["TIME"] - b["EVENTS"].data["TIME"]
            assert np.max(np.abs(diff)) < 1e-6
            assert np.max(np.abs(diff)) > 0  # it really was filled, not read
            history = "\n".join(str(card) for card in a["EVENTS"].header["HISTORY"])
            assert "Orbit gap filled" in history

    def test_pint_engine_refuses_the_flag(self, tmp_path):
        """Gap filling lives in the native engine only; asking the other for it is loud."""
        import pytest

        with pytest.raises(ValueError, match="native"):
            self.run(GBM_ORBIT, str(tmp_path / "x.evt"), "--fill-orbit-gaps", "--engine", "pint")
