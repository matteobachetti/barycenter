"""Swift's UTCF clock correction, and how it meets the leap-second term.

The clock file tabulates the UTC correction factor as a quadratic per fit interval, in
microseconds, and Swift's MET counts UTC seconds -- so a Swift time gets two whole-second
sized pieces of bookkeeping rather than one.  These tests pin down both, and the sign
convention above all: a 15.6 s correction applied the wrong way round is a 31 s error.
"""

import os

import numpy as np
import pytest
from astropy.io import fits

from barycenter.clock import SWIFT_CLOCK_EXTENSION, swift_clock_correction_fun
from barycenter.utils import leap_seconds_since_mjdref

curdir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
CLOCKFILE = os.path.join(curdir, "data", "dummy_swift_clk.fits")

#: Swift's MJDREF, as MJDREFI + MJDREFF: 2001-01-01 00:00:00 UTC, expressed in TT.
SWIFT_MJDREF = 51910 + 0.00074287037

#: The 2015-07-01 leap second, in Swift MET.  5294 whole days after 2001-01-01, and
#: 5294 * 86400 = 457401600 exactly -- which is itself the cleanest evidence that Swift's
#: MET is plain UTC seconds with no leap seconds counted.
LEAP_2015_MET = 457401600.0


@pytest.fixture(scope="module")
def correction():
    return swift_clock_correction_fun(CLOCKFILE)


@pytest.fixture(scope="module")
def intervals():
    """TSTART, TSTOP and the quadratic coefficients, straight out of the file."""
    table = fits.getdata(CLOCKFILE, SWIFT_CLOCK_EXTENSION)
    return {
        name: np.asarray(table[name], dtype=np.float64)
        for name in ("TSTART", "TSTOP", "C0", "C1", "C2")
    }


class TestTheCorrectionItself:
    def test_it_is_the_tabulated_polynomial_with_the_sign_flipped(self, correction, intervals):
        """The value is exactly ``-(C0 + C1 x + C2 x^2) * 1e-6`` with ``x`` in days.

        This is the whole contract, checked against the file's own numbers rather than
        against a remembered value, at the start of every interval the file contains.
        """
        starts = intervals["TSTART"]
        expected = -intervals["C0"] * 1e-6
        assert np.allclose(correction(starts), expected, rtol=0, atol=1e-12)

    def test_the_correction_is_the_utcf_and_about_minus_sixteen_seconds(self, correction):
        """In December 2015 Swift's clock was 15.56 s ahead of UTC, so the shift is negative.

        The magnitude matters as much as the sign: tens of seconds, not the microseconds a
        NuSTAR-style fine clock correction would give, so a unit slip cannot hide.
        """
        value = correction(471883955.0)
        assert -15.6 < value < -15.5

    def test_the_clock_drifts_across_an_observation_rather_than_being_a_constant(self, correction):
        """The quadratic actually varies: 342 us across this 6.4 ks exposure.

        A reader that took C0 and ignored C1 would pass every test above and still be
        3400 times the accuracy target wrong by the end of a day.
        """
        drift = correction(471890320.0) - correction(471883955.0)
        assert -400e-6 < drift < -300e-6

    def test_a_scalar_stays_scalar_and_an_array_keeps_its_shape(self, correction):
        """The correction is used both on a single TSTART and on a whole time column."""
        assert np.isscalar(correction(471883955.0)) or correction(471883955.0).ndim == 0
        assert correction(np.full((3, 4), 471883955.0)).shape == (3, 4)


class TestIntervalLookup:
    def test_each_time_uses_the_interval_that_contains_it(self, correction, intervals):
        """The row picked is the one bracketing the time, not the first or the nearest.

        The file's intervals are contiguous -- TSTOP equals the next TSTART -- so an
        off-by-one in the search would be invisible in the middle of the table and wrong
        by a second at a leap-second boundary.
        """
        starts, stops = intervals["TSTART"], intervals["TSTOP"]
        middles = 0.5 * (starts + stops)
        expected = (
            -(
                intervals["C0"]
                + intervals["C1"] * ((middles - starts) / 86400.0)
                + intervals["C2"] * ((middles - starts) / 86400.0) ** 2
            )
            * 1e-6
        )
        assert np.allclose(correction(middles), expected, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("offset", [-1.0, +1.0])
    def test_times_outside_the_tabulated_span_are_refused(self, correction, intervals, offset):
        """Off either end raises, rather than silently extrapolating a quadratic.

        This is the same choice made for an old NuSTAR file the clock file does not cover:
        a wrong answer to 100 ns is worse than no answer, and the message has to say which
        way out to go -- fetch a newer file, or run without one.
        """
        edge = intervals["TSTART"].min() if offset < 0 else intervals["TSTOP"].max()
        with pytest.raises(ValueError, match="not covered"):
            correction(edge + offset)

    def test_a_file_without_the_clock_extension_is_refused(self, tmp_path):
        """Handed some other Swift product, say the wrong CALDB file, it says so."""
        path = str(tmp_path / "nonsense.fits")
        hdu = fits.BinTableHDU.from_columns(
            [fits.Column(name="TIME", format="D", array=np.zeros(3))], name="EVENTS"
        )
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
        with pytest.raises(ValueError, match=SWIFT_CLOCK_EXTENSION):
            swift_clock_correction_fun(path)


class TestMeetingTheLeapSecondTerm:
    """The two whole-second pieces, and why they have to be applied together.

    The tabulated UTCF converts the onboard clock to *UTC*, so it steps by -1 s at each
    leap second; the leap-second term converts a UTC-seconds count to TT, so it steps by
    +1 s.  Neither is optional, and each on its own is a one-second error.
    """

    def test_the_tabulated_correction_steps_down_one_second_at_a_leap_second(
        self, correction, intervals
    ):
        """Across the 2015-07-01 boundary the UTCF drops by very nearly exactly 1 s.

        The interval boundary is not at the leap second itself but one second before it,
        at the start of the inserted second: the file puts TSTART at 457401613.791, which
        is the leap instant in onboard-clock time (457401600 + 14.791 s of UTCF) minus the
        one second being inserted.
        """
        boundary = intervals["TSTART"][
            np.searchsorted(intervals["TSTART"], LEAP_2015_MET, side="right")
        ]
        step = correction(boundary + 0.001) - correction(boundary - 0.001)
        assert np.isclose(step, -1.0, rtol=0, atol=1e-5)

    def test_the_leap_term_steps_up_one_second_at_the_same_leap_second(self):
        """The other half of the pair: +1 s as MET crosses 2015-07-01 00:00:00 UTC."""
        before = leap_seconds_since_mjdref(SWIFT_MJDREF, LEAP_2015_MET - 1.0)
        after = leap_seconds_since_mjdref(SWIFT_MJDREF, LEAP_2015_MET + 1.0)
        assert (before, after) == (3.0, 4.0)

    def test_the_leap_term_is_exactly_four_seconds_for_this_observation(self):
        """Exactly 4.0, not 4.000000063: an integer count must not be computed by subtraction.

        Taking ``(epoch.tai.mjd - epoch.utc.mjd) * 86400`` gives 31.999999937 s, because two
        MJDs of order 5e4 cannot express 32 s to better than 0.6 us.  The resulting 63 ns
        bias was two thirds of the budget and showed up as a real disagreement with
        ``barycorr``, so exactness is asserted here rather than approximate equality.
        """
        assert leap_seconds_since_mjdref(SWIFT_MJDREF, 471883955.0) == 4.0

    def test_the_two_together_move_the_time_by_about_minus_eleven_and_a_half_seconds(
        self, correction
    ):
        """-15.56 s of UTCF plus 4 s of leap seconds, which is what reaches the barycentre step."""
        met = 471883955.0
        total = correction(met) + leap_seconds_since_mjdref(SWIFT_MJDREF, met)
        assert -11.6 < total < -11.5
