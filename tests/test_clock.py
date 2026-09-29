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
    read_tdc_file,
    rxte_clock_correction_fun,
    rxte_tdc_file,
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


#: A toy tdc.dat: two blocks, each with one quadratic, and the trailing comment block
#: that is what actually terminates the file for the original C reader.
TOY_TDC = """\
       100.00000       1.5000000 -1 -1
       10.000000       2.0000000       0.0000000       5.0000000
       200.00000       2.5000000 -1 -1
       1.0000000       0.0000000       3.0000000       4.0000000
#
# Not a real clock file
"""


class TestRXTETdc:
    """The RXTE coefficient file, translated from xCC.c.

    https://heasarc.gsfc.nasa.gov/docs/xte/abc/xCC.c
    """

    def test_block_headers_set_the_day_and_the_polynomial_is_relative_to_it(self, tmp_path):
        """A row with a negative fourth number starts a block; the rest are coefficients.

        Each quadratic is in days since its *block's* day, not since the mission start, so
        reading the header rows as coefficients -- or the other way round -- would give
        answers wrong by hundreds of microseconds.
        """
        path = tmp_path / "toy.dat"
        path.write_text(TOY_TDC)
        table = read_tdc_file(str(path))
        assert len(table) == 2
        assert list(table["SUBDAY"]) == [100.0, 200.0]
        assert list(table["DAY_END"]) == [105.0, 204.0]
        assert list(table["TIMEZERO"]) == [1.5, 2.5]

        # HEXTE takes the polynomial as it stands, so this checks the arithmetic alone.
        fun = rxte_clock_correction_fun(str(path), instrument="HEXTE")
        assert np.isclose(fun(102.0 * 86400), (10.0 + 2.0 * 2) * 1e-6)
        assert np.isclose(fun(202.0 * 86400), (1.0 + 3.0 * 2**2) * 1e-6)

    def test_the_pca_gets_an_extra_16_microsecond_delay(self, tmp_path):
        """Everything that is not HEXTE is treated as the PCA, as hdaxbary does.

        The two instruments differ by exactly 16 us of detector delay, and getting that
        wrong is a 16 us error that no ephemeris or orbit problem could imitate.
        """
        path = tmp_path / "toy.dat"
        path.write_text(TOY_TDC)
        met = 102.0 * 86400
        hexte = rxte_clock_correction_fun(str(path), instrument="HEXTE")(met)
        for instrument in ("PCA", "pca", "ASM", None):
            other = rxte_clock_correction_fun(str(path), instrument=instrument)(met)
            assert np.isclose(hexte - other, 16e-6)

    def test_times_past_the_last_block_get_no_correction(self, tmp_path, caplog):
        """Beyond the file there is nothing to interpolate, so nothing is applied.

        RXTE stopped observing in January 2012 and tdc.dat ends there, so this is the
        normal behaviour for a time typed in by mistake, not an edge case; it is reported
        rather than extrapolated, because the coefficients are per-interval fits with no
        meaning outside their interval.
        """
        path = tmp_path / "toy.dat"
        path.write_text(TOY_TDC)
        fun = rxte_clock_correction_fun(str(path), instrument="HEXTE")
        with caplog.at_level("WARNING"):
            got = fun(np.array([102.0, 300.0]) * 86400)
        assert np.isclose(got[0], 14e-6)
        assert got[1] == 0.0
        assert any("past the end" in record.message for record in caplog.records)

    def test_the_bundled_file_covers_the_whole_mission(self):
        """The copy of tdc.dat shipped with the package is complete and readable.

        RXTE ran from December 1995 (mission day 729) to January 2012 (day 6578), and the
        package bundles the file rather than needing HEASOFT, so a truncated or missing
        copy has to fail here and not in the middle of someone's observation.
        """
        table = read_tdc_file(rxte_tdc_file())
        assert len(table) > 700
        assert table["DAY_END"][0] < 730
        assert table["DAY_END"][-1] > 6570
        # xCC.c walks the file and stops at the first block whose end is past the time
        # asked for, which is only a searchsorted if the ends never go backwards.
        assert np.all(np.diff(table["DAY_END"]) >= 0)
