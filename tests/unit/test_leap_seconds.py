"""The leap-second term for missions whose MET counts UTC seconds."""

import numpy as np
import pytest

from barycenter.utils import leap_seconds_since_mjdref

#: Swift's and Fermi's reference epoch: MJD 51910 UTC, written as MJD(TT) by adding the
#: 64.184 s that TT ran ahead of UTC on 2001-01-01.
SWIFT_MJDREF = 51910 + 0.00074287037

#: Chandra's and XMM's: MJD 50814 TT exactly, with no leap-second offset folded in.
CHANDRA_MJDREF = 50814.0


def met_of(mjdref, iso):
    """The MET of a UTC instant, treating ``mjdref`` as an MJD(TT)."""
    from astropy.time import Time

    return (Time(iso, scale="utc").tt.mjd - mjdref) * 86400.0


class TestSwiftEpoch:
    """TAI-UTC was 32 s at Swift's epoch and has stepped four times since."""

    def test_the_december_2015_observation_needs_four_seconds(self):
        """The measured value: barycorr adds exactly 4 s on the Swift test observation.

        Without it we sit 4.0003 s from barycorr; with it, 36 ns. The extra 0.3 ms is the
        barycentric correction being evaluated 4 s later, not a second discrepancy.
        """
        met = 471883955.0232034  # TSTART of observation 00037258040
        assert leap_seconds_since_mjdref(SWIFT_MJDREF, met) == pytest.approx(4.0)

    @pytest.mark.parametrize(
        "iso,expected",
        [
            ("2001-06-01", 0.0),  # before the first step after the epoch
            ("2006-06-01", 1.0),  # 2006-01-01 took TAI-UTC to 33
            ("2010-06-01", 2.0),  # 2009-01-01 to 34
            ("2013-06-01", 3.0),  # 2012-07-01 to 35
            ("2016-06-01", 4.0),  # 2015-07-01 to 36
            ("2020-06-01", 5.0),  # 2017-01-01 to 37
        ],
    )
    def test_each_step_since_the_epoch(self, iso, expected):
        """One second per leap second inserted between the epoch and the observation."""
        met = met_of(SWIFT_MJDREF, iso)
        assert leap_seconds_since_mjdref(SWIFT_MJDREF, met) == pytest.approx(expected)

    def test_the_step_lands_on_the_right_side_of_the_boundary(self):
        """A minute either side of a leap second gets 3 s and 4 s respectively.

        Rounding the observation to whole days, which would be the cheap way to do this,
        would put up to 64 s of data on the wrong side -- silently, and by a whole second.
        """
        before = met_of(SWIFT_MJDREF, "2015-06-30T23:59:00")
        after = met_of(SWIFT_MJDREF, "2015-07-01T00:01:00")
        assert leap_seconds_since_mjdref(SWIFT_MJDREF, before) == pytest.approx(3.0)
        assert leap_seconds_since_mjdref(SWIFT_MJDREF, after) == pytest.approx(4.0)


class TestShapeAndEpochIndependence:
    def test_an_array_comes_back_with_its_own_shape(self):
        """Event times arrive as arrays, and a file may straddle a leap second."""
        mets = np.array([met_of(SWIFT_MJDREF, s) for s in ("2013-01-01", "2016-01-01")])
        got = leap_seconds_since_mjdref(SWIFT_MJDREF, mets)
        assert got.shape == mets.shape
        assert np.allclose(got, [3.0, 4.0])

    def test_a_scalar_comes_back_as_a_scalar(self):
        """Header keywords are corrected one at a time, not as arrays."""
        got = leap_seconds_since_mjdref(SWIFT_MJDREF, met_of(SWIFT_MJDREF, "2016-01-01"))
        assert np.ndim(got) == 0

    def test_a_tt_epoch_mission_at_the_same_date_gets_a_different_number(self):
        """The answer counts from the file's own epoch, not from any fixed date.

        Chandra's epoch is 1998, three leap seconds earlier than Swift's, so the same
        calendar date owes three more seconds -- which is exactly why this must never be
        applied to a mission whose MET already counts TT seconds.
        """
        iso = "2016-01-01"
        swift = leap_seconds_since_mjdref(SWIFT_MJDREF, met_of(SWIFT_MJDREF, iso))
        chandra = leap_seconds_since_mjdref(CHANDRA_MJDREF, met_of(CHANDRA_MJDREF, iso))
        assert swift == pytest.approx(4.0)
        assert chandra == pytest.approx(5.0)


class TestEpochAfterTheLastLeapSecond:
    """A reference epoch later than every tabulated step owes nothing, not an error."""

    #: An epoch after 2017-01-01, the most recent leap second. SVOM's is 2024, so this is
    #: the shape of any mission launched since -- not a hypothetical.
    RECENT_MJDREF = 60000.0

    def test_a_recent_epoch_owes_nothing(self):
        """No step falls after the epoch, so the leap-second table is empty here.

        The empty table used to be indexed anyway -- ``np.where`` evaluates both of its
        branches -- and raised IndexError instead of returning zero.
        """
        assert leap_seconds_since_mjdref(self.RECENT_MJDREF, 1000.0) == pytest.approx(0.0)

    def test_a_recent_epoch_owes_nothing_for_an_array_either(self):
        """Event times arrive as arrays, and must survive the same empty table."""
        got = leap_seconds_since_mjdref(self.RECENT_MJDREF, np.array([0.0, 1e6, 1e8]))
        assert np.allclose(got, 0.0)
