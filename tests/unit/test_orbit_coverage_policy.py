"""What happens to times the orbit file cannot place.

The rule has three branches, and which one a time takes depends on whether it is a good
time: a time outside every GTI is dropped, a time inside one is an error, and the
``TSTART``/``TSTOP`` keywords -- which routinely hold the *requested* range rather than
the observed one -- are moved to the edge of the data instead. These tests fix all three,
using synthetic files so they cost nothing to run.
"""

import numpy as np
import pytest
from astropy.io import fits

from barycenter.core import (
    COVERAGE_TOLERANCE_S,
    clamp_uncovered_keyword,
    enforce_orbit_coverage,
    good_time_span,
    gti_intervals,
    inside_gti,
)
from barycenter.orbit import OrbitCoverage

#: An orbit tabulated every 30 s from 1000 to 4000, like a Fermi spacecraft file.
COVERAGE = OrbitCoverage.from_met(np.arange(1000.0, 4030.0, 30.0))


def make_hdul(times, gtis=((1000.0, 4000.0),), tstart=1000.0, tstop=4000.0):
    """An event list with a GTI extension, the shape every real file has."""
    primary = fits.PrimaryHDU()
    events = fits.BinTableHDU.from_columns(
        [fits.Column(name="TIME", format="D", array=np.asarray(times, dtype=float))],
        name="EVENTS",
    )
    events.header["TSTART"], events.header["TSTOP"] = tstart, tstop
    hdus = [primary, events]
    if gtis is not None:
        gti = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="START", format="D", array=np.array([g[0] for g in gtis])),
                fits.Column(name="STOP", format="D", array=np.array([g[1] for g in gtis])),
            ],
            name="GTI",
        )
        hdus.append(gti)
    return fits.HDUList(hdus)


class TestFindingTheGoodTimes:
    """Before anything can be judged, the GTIs have to be found and read."""

    def test_the_gti_extension_is_recognised(self):
        """START and STOP without a TIME column is what makes an extension a GTI."""
        start, stop = gti_intervals(make_hdul([1500.0], gtis=((1100.0, 1200.0),)))
        assert start.tolist() == [1100.0] and stop.tolist() == [1200.0]

    def test_a_file_without_gtis_reports_none(self):
        """A missing GTI extension is not an empty one; the caller has to tell them apart."""
        assert gti_intervals(make_hdul([1500.0], gtis=None)) is None

    def test_times_are_tested_against_every_interval(self):
        """An event is in a good time if it is in any GTI, not only the first."""
        gti = gti_intervals(make_hdul([0.0], gtis=((1100.0, 1200.0), (1300.0, 1400.0))))
        assert inside_gti([1150.0, 1250.0, 1350.0], gti).tolist() == [True, False, True]

    def test_without_gtis_every_time_counts_as_good(self):
        """Silence about which times are good is not a claim that none of them are."""
        assert np.all(inside_gti([1.0, 2.0], None))


class TestSpanToClampTo:
    """Where an out-of-range keyword gets pulled back to."""

    def test_the_gtis_give_the_span(self):
        """The GTIs are the file's own statement of where its data begins and ends."""
        assert good_time_span(make_hdul([1500.0], gtis=((1100.0, 1200.0), (1300.0, 1400.0)))) == (
            1100.0,
            1400.0,
        )

    def test_without_gtis_the_events_give_the_span(self):
        """A file with no GTIs still knows where its events are, which is the same intent."""
        assert good_time_span(make_hdul([1500.0, 1900.0], gtis=None)) == (1500.0, 1900.0)

    def test_a_file_with_neither_has_no_span(self):
        """Nothing to clamp to, which the caller has to handle rather than invent a number."""
        assert good_time_span(fits.HDUList([fits.PrimaryHDU()])) is None


class TestKeywordClamping:
    """TSTART and TSTOP are moved to the data; nothing else is touched."""

    def test_a_tstart_before_the_orbit_file_is_moved_to_the_data(self):
        """The Fermi case: TSTART is the requested window, which predates the spacecraft file."""
        assert clamp_uncovered_keyword(0.0, "TSTART", COVERAGE, (1100.0, 3900.0)) == 1100.0

    def test_a_tstop_after_the_orbit_file_is_moved_to_the_data(self):
        """The same at the other end, where TSTOP goes to the last good time rather than the first."""
        assert clamp_uncovered_keyword(9999.0, "TSTOP", COVERAGE, (1100.0, 3900.0)) == 3900.0

    def test_a_covered_keyword_is_left_alone(self):
        """The common case must be untouched, or every file would have its span rewritten."""
        assert clamp_uncovered_keyword(2000.0, "TSTART", COVERAGE, (1100.0, 3900.0)) == 2000.0

    def test_a_keyword_just_outside_is_tolerated(self):
        """Orbit files end raggedly; a miss inside the tolerance is not worth rewriting a header for."""
        just_outside = 4030.0 + COVERAGE_TOLERANCE_S - 1.0
        assert clamp_uncovered_keyword(just_outside, "TSTOP", COVERAGE, (1100.0, 3900.0)) == (
            just_outside
        )

    def test_other_keywords_are_never_clamped(self):
        """Only these two hold a requested range; moving any other would be silent corruption."""
        assert clamp_uncovered_keyword(0.0, "TIME", COVERAGE, (1100.0, 3900.0)) == 0.0

    def test_the_move_is_logged_loudly(self, caplog):
        """Rewriting a header keyword must never be silent, whatever the reason."""
        clamp_uncovered_keyword(0.0, "TSTART", COVERAGE, (1100.0, 3900.0))
        assert any(r.levelname == "WARNING" for r in caplog.records)


class TestRowsOutsideTheOrbitFile:
    """Junk is dropped, real data is refused, and covered files are untouched."""

    def test_a_covered_file_loses_nothing(self):
        """The overwhelmingly common case: no rows dropped, no error, no warning."""
        hdul = make_hdul([1500.0, 2500.0, 3500.0])
        assert enforce_orbit_coverage(hdul, COVERAGE) == 0
        assert len(hdul["EVENTS"].data) == 3

    def test_an_uncovered_event_outside_every_gti_is_dropped(self):
        """It is junk the orbit file also cannot place, so it cannot be barycentred at all."""
        hdul = make_hdul([1500.0, 9000.0], gtis=((1400.0, 1600.0),))
        assert enforce_orbit_coverage(hdul, COVERAGE) == 1
        assert np.asarray(hdul["EVENTS"].data["TIME"]).tolist() == [1500.0]

    def test_dropping_rows_is_logged_loudly(self, caplog):
        """Events leaving the file silently would be the worst outcome of the whole check."""
        enforce_orbit_coverage(make_hdul([1500.0, 9000.0], gtis=((1400.0, 1600.0),)), COVERAGE)
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_an_uncovered_event_inside_a_gti_is_an_error(self):
        """Real data the orbit file cannot place: there is no honest time to write for it."""
        hdul = make_hdul([1500.0, 9000.0], gtis=((1400.0, 9500.0),))
        with pytest.raises(ValueError, match=r"EVENTS/TIME: 1 times inside the good"):
            enforce_orbit_coverage(hdul, COVERAGE)

    def test_an_event_just_outside_is_tolerated(self):
        """Within the tolerance the spline is still coasting, which is what it always does."""
        # The last sample is at 4000, coverage reaches one 30 s cadence past it, and the
        # tolerance is the rest. The GTI has to end there too, or it is itself uncovered.
        just_outside = 4000.0 + COVERAGE.cadence + COVERAGE_TOLERANCE_S - 1.0
        hdul = make_hdul([just_outside], gtis=((1000.0, just_outside),))
        assert enforce_orbit_coverage(hdul, COVERAGE) == 0

    def test_without_gtis_an_uncovered_event_is_an_error(self):
        """With nothing saying the event is bad, it is good data, so it is refused not dropped."""
        with pytest.raises(ValueError, match="outside the orbit file"):
            enforce_orbit_coverage(make_hdul([1500.0, 9000.0], gtis=None), COVERAGE)

    def test_an_uncovered_gti_boundary_is_an_error(self):
        """A GTI says the time is good by definition, so it can never be dropped as junk."""
        with pytest.raises(ValueError, match="outside the orbit file"):
            enforce_orbit_coverage(make_hdul([1500.0], gtis=((1400.0, 9000.0),)), COVERAGE)

    def test_the_shift_is_applied_before_testing(self):
        """The orbit file is asked about the clock-corrected time, so that is what must be tested."""
        hdul = make_hdul([3900.0], gtis=((1000.0, 9000.0),))
        with pytest.raises(ValueError, match="outside the orbit file"):
            enforce_orbit_coverage(hdul, COVERAGE, clock_fun=lambda t: np.full_like(t, 5000.0))
