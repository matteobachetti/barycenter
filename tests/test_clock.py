"""Tests for the spacecraft clock correction.

The clock correction is applied before the barycentric one, so an error here moves every
time in the file. On NuSTAR it is about 25 ms, which is 250000 times our accuracy target.
"""

import os

import numpy as np
import pytest
from astropy.table import Table

from barycenter.clock import (
    clock_cache_dir,
    cubic_interpolation,
    interpolate_clock_function,
    nustar_clock_correction_fun,
)

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")

FINE_CLOCK = os.path.join(datadir, "dummy_fine_clk.fits")
#: A real clock file in the pre-2019 format, kept so we can prove it is refused.
OLD_CLOCK = os.path.join(datadir, "dummy_clk.fits")


def toy_table(times, offsets, slopes):
    """A clock table shaped like NU_FINE_CLOCK: offset and its derivative on a grid."""
    return Table(
        {"TIME": times, "CLOCK_OFF_CORR": offsets, "CLOCK_FREQ_CORR": slopes},
        names=("TIME", "CLOCK_OFF_CORR", "CLOCK_FREQ_CORR"),
    )


class TestInterpolation:
    def test_reproduces_the_tabulated_offsets_and_slopes(self):
        """At a tabulated time the interpolation returns the tabulated value exactly.

        That is the defining property of Hermite interpolation, and it is why the
        ``CLOCK_FREQ_CORR`` column is used rather than letting a spline guess the slope.
        """
        times = np.arange(0.0, 5000.0, 1000.0)
        offsets = 0.02 + 3e-7 * times
        slopes = np.full_like(times, 3e-7)
        table = toy_table(times, offsets, slopes)

        assert np.allclose(interpolate_clock_function(table, times), offsets, atol=1e-15)
        # A linear function with the right slopes is reproduced between the knots too.
        mid = times[:-1] + 500.0
        assert np.allclose(interpolate_clock_function(table, mid), 0.02 + 3e-7 * mid, atol=1e-15)

    def test_returns_one_value_per_requested_time(self, caplog):
        """Times outside the table get a value too, instead of raising a length mismatch.

        The old code returned a shortened array plus a validity mask that its only caller
        threw away, then built a spline from a full-length abscissa and the short
        ordinate. Any event beyond the clock file's span crashed the run.
        """
        times = np.arange(0.0, 5000.0, 1000.0)
        table = toy_table(times, 0.02 + 3e-7 * times, np.full_like(times, 3e-7))
        asked = np.array([-500.0, 0.0, 2500.0, 4000.0, 9000.0])

        with caplog.at_level("WARNING"):
            got = interpolate_clock_function(table, asked)
        assert got.shape == asked.shape
        assert np.all(np.isfinite(got))
        # Extrapolation continues the edge interval, so a linear table stays linear.
        assert np.allclose(got, 0.02 + 3e-7 * asked, atol=1e-15)
        # ... and going outside is reported, because an extrapolated clock is a guess.
        assert any("outside the clock file" in r.message for r in caplog.records)

    def test_cubic_interpolation_matches_heasoft_cubeterp(self):
        """The interpolation is the cubic through two points and their derivatives.

        Checked against a polynomial written out by hand, because this routine is a
        translation of HEASOFT's ``cubeterp`` and a sign error in it would be invisible
        in the linear case above.
        """
        # y = 1 + 2t + 3t^2 + 4t^3 on [0, 1], with its exact derivatives.
        xtab = [np.array([0.0]), np.array([1.0])]
        ytab = [np.array([1.0]), np.array([10.0])]
        yptab = [np.array([2.0]), np.array([20.0])]
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            expected = 1 + 2 * t + 3 * t**2 + 4 * t**3
            got = cubic_interpolation(np.array([t]), xtab, ytab, yptab)
            assert np.isclose(got[0], expected)


class TestClockFiles:
    def test_reads_the_fine_clock_extension(self):
        """The committed NuSTAR clock file gives a 20-30 ms correction over the test span."""
        fun = nustar_clock_correction_fun(FINE_CLOCK)
        met = np.linspace(178574700.0, 178656500.0, 50)
        corr = fun(met)
        assert corr.shape == met.shape
        assert np.all((0.019 < corr) & (corr < 0.030))
        # A scalar in, a scalar out: header keywords are corrected one at a time.
        assert np.ndim(fun(178600000.0)) == 0

    def test_an_old_clock_file_is_refused(self):
        """A pre-2019 CLOCK_CORRECT file is rejected, with a message that says why.

        Its per-interval polynomial is only good to the millisecond, so applying it
        silently would leave the output four orders of magnitude off target while looking
        like a properly clock-corrected file.
        """
        with pytest.raises(ValueError, match="NU_FINE_CLOCK"):
            nustar_clock_correction_fun(OLD_CLOCK)


def test_clock_cache_is_not_the_working_directory():
    """Downloaded clock files go to a user cache, not wherever the command was run.

    NuSTAR clock files are about 12 MB and the old code re-downloaded one into the
    current directory on every run that did not name a clock file.
    """
    cache = clock_cache_dir()
    assert os.path.isdir(cache)
    assert os.path.abspath(cache) != os.path.abspath(os.getcwd())
