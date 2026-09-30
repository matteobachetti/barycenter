"""Choosing an interpolation grid, and working out how wide it has to be.

These are the two decisions ``apply_barycenter_correction`` makes before it builds the
correction. They are separate functions so they can be tested without a hundred-thousand
event file, which is the whole point of the threshold.
"""

import numpy as np
import pytest
from astropy.io import fits

from barycenter.core import (
    AUTO_GRID_DT_S,
    AUTO_GRID_EVENTS,
    MET_RANGE_PAD_S,
    grid_spacing_for,
    met_range_for_file,
    pre_bary_shift,
)


class TestGridSpacingFor:
    """Small files stay exact; big files get a grid; the caller can always override."""

    def test_a_small_file_is_evaluated_exactly(self):
        """Below the threshold there is nothing to buy, so no approximation is made."""
        assert grid_spacing_for(AUTO_GRID_EVENTS) is None
        assert grid_spacing_for(0) is None

    def test_a_large_file_gets_the_default_grid(self):
        """Above the threshold the 48x speed-up is taken."""
        assert grid_spacing_for(AUTO_GRID_EVENTS + 1) == AUTO_GRID_DT_S

    def test_an_explicit_spacing_wins_either_way(self):
        """``dt`` given by the user is used whatever the file's size."""
        assert grid_spacing_for(10, dt=2.5) == 2.5
        assert grid_spacing_for(10_000_000, dt=1.0) == 1.0

    @pytest.mark.parametrize("dt", [0, 0.0, -1])
    def test_zero_or_negative_forces_the_exact_path(self, dt):
        """The escape hatch: no grid at all, however large the file."""
        assert grid_spacing_for(10_000_000, dt=dt) is None


class TestMetRangeForFile:
    """The grid has to cover the times the correction is actually asked for."""

    @staticmethod
    def hdul(tstart, tstop, extra=None):
        primary = fits.PrimaryHDU()
        events = fits.BinTableHDU.from_columns(
            [fits.Column(name="TIME", format="D", array=np.array([tstart, tstop]))], name="EVENTS"
        )
        events.header["TSTART"], events.header["TSTOP"] = tstart, tstop
        hdus = [primary, events]
        if extra is not None:
            gti = fits.BinTableHDU.from_columns(
                [fits.Column(name="START", format="D", array=np.array([extra[0]]))], name="GTI"
            )
            gti.header["TSTART"], gti.header["TSTOP"] = extra
            hdus.append(gti)
        return fits.HDUList(hdus)

    def test_the_span_is_the_headers_plus_the_pad(self):
        """With no clock or leap term the range is just TSTART/TSTOP, padded."""
        start, stop = met_range_for_file(self.hdul(1000.0, 2000.0))
        assert start == pytest.approx(1000.0 - MET_RANGE_PAD_S)
        assert stop == pytest.approx(2000.0 + MET_RANGE_PAD_S)

    def test_every_extension_is_considered(self):
        """A GTI extension reaching past the events widens the range, not narrows it."""
        start, stop = met_range_for_file(self.hdul(1000.0, 2000.0, extra=(900.0, 2500.0)))
        assert start == pytest.approx(900.0 - MET_RANGE_PAD_S)
        assert stop == pytest.approx(2500.0 + MET_RANGE_PAD_S)

    def test_a_file_with_no_tstart_cannot_be_clipped(self):
        """``None`` means "do not clip": there is nothing to clip to."""
        bare = fits.HDUList([fits.PrimaryHDU()])
        assert met_range_for_file(bare) is None

    def test_the_clock_and_leap_shifts_move_the_range(self):
        """The range follows the clock-corrected times, which is where bary_fun is called.

        On Swift the UTCF plus leap seconds come to nearly 20 s. A grid clipped to the
        raw span would leave every event outside it, which is the bug this guards.
        """
        shift = 19.56
        start, stop = met_range_for_file(
            self.hdul(1000.0, 2000.0),
            clock_fun=lambda t: np.full_like(np.asarray(t, dtype=float), -shift),
            leap_fun=lambda t: np.zeros_like(np.asarray(t, dtype=float)),
        )
        assert start == pytest.approx(1000.0 - MET_RANGE_PAD_S - shift)
        assert stop == pytest.approx(2000.0 + MET_RANGE_PAD_S - shift)

    def test_timezero_is_folded_in(self):
        """TIMEZERO is added to the times before anything else, so the range moves too."""
        start, _ = met_range_for_file(self.hdul(1000.0, 2000.0), timezero=50.0)
        assert start == pytest.approx(1000.0 - MET_RANGE_PAD_S + 50.0)


class TestPreBaryShift:
    """The arithmetic ``correct_times`` does before it calls the correction."""

    def test_both_terms_are_evaluated_on_the_raw_times(self):
        """Not on each other's output: that ordering is worth 214 ns on Swift.

        The shift is ``t + timezero + leap(t + timezero) + clock(t + timezero)``, and the
        two terms see the same argument. Feeding the clock the leap-shifted time instead
        is the mistake this pins down.
        """
        seen = []

        def clock(t):
            seen.append(np.asarray(t, dtype=float).copy())
            return np.zeros_like(np.asarray(t, dtype=float))

        out = pre_bary_shift(
            [100.0, 200.0], timezero=1.0, clock_fun=clock, leap_fun=lambda t: t * 0 + 4.0
        )
        assert np.allclose(seen[0], [101.0, 201.0]), "the clock saw the leap-shifted time"
        assert np.allclose(out, [105.0, 205.0])

    def test_no_terms_is_just_timezero(self):
        """With neither term the only change is TIMEZERO."""
        assert np.allclose(pre_bary_shift([10.0, 20.0], timezero=2.0), [12.0, 22.0])
