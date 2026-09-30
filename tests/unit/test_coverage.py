"""Telling a spacecraft position that was measured from one that was guessed.

An interpolating spline answers every time it is asked about, so nothing downstream can
tell an interpolation between two samples from an extrapolation off the end of the file.
:class:`~barycenter.orbit.OrbitCoverage` is what draws that line, and these tests pin
down where it falls -- particularly that it does *not* fall between two ordinary samples,
which is the failure mode that would reject a third of a healthy Fermi file.
"""

import numpy as np
import pytest

from barycenter.orbit import OrbitCoverage


class TestCadence:
    """The allowance is read off the file, not assumed."""

    def test_the_cadence_is_the_sampling_interval(self):
        """A uniformly sampled file reports the interval it was sampled at."""
        assert OrbitCoverage.from_met(np.arange(0.0, 300.0, 30.0)).cadence == 30.0

    def test_a_few_gaps_do_not_move_the_cadence(self):
        """The median ignores the gaps, so a file that is mostly regular reads as regular."""
        met = np.concatenate([np.arange(0.0, 3000.0, 30.0), np.arange(9000.0, 12000.0, 30.0)])
        assert OrbitCoverage.from_met(met).cadence == 30.0


class TestUncovered:
    """Zero wherever the file has something to say, positive wherever it does not."""

    def test_a_time_between_two_samples_is_covered(self):
        """The ordinary case: interpolation is not extrapolation, so nothing is reported."""
        cov = OrbitCoverage.from_met(np.arange(0.0, 300.0, 30.0))
        assert np.all(cov.uncovered(np.arange(0.0, 270.0, 0.5)) == 0.0)

    def test_one_sampling_interval_past_the_end_is_covered(self):
        """Orbit files routinely stop a sample short of the last event; that is not an error."""
        cov = OrbitCoverage.from_met(np.arange(0.0, 300.0, 30.0))
        assert cov.uncovered([300.0]) == 0.0  # last sample is 270, so this is one interval on

    def test_past_the_end_is_reported_less_the_allowance(self):
        """Beyond the allowance the shortfall is reported as seconds, to compare to a tolerance."""
        cov = OrbitCoverage.from_met(np.arange(0.0, 300.0, 30.0))
        assert cov.uncovered([400.0]) == pytest.approx(100.0)  # 400 - 270 - 30

    def test_before_the_start_is_reported_too(self):
        """The leading edge matters: a FITS TSTART can predate the first spacecraft sample."""
        cov = OrbitCoverage.from_met(np.arange(1000.0, 1300.0, 30.0))
        assert cov.uncovered([0.0]) == pytest.approx(970.0)

    def test_the_middle_of_an_interior_gap_is_uncovered(self):
        """A file missing a chunk is guessing there, even though the spline coasts through it."""
        met = np.concatenate([np.arange(0.0, 300.0, 30.0), np.arange(2000.0, 2300.0, 30.0)])
        cov = OrbitCoverage.from_met(met)
        assert cov.uncovered([1135.0]) == pytest.approx(835.0)

    def test_the_edges_of_an_interior_gap_are_covered(self):
        """Just inside a gap the spline is still coasting one interval, which is what it always does."""
        met = np.concatenate([np.arange(0.0, 300.0, 30.0), np.arange(2000.0, 2300.0, 30.0)])
        cov = OrbitCoverage.from_met(met)
        assert np.all(cov.uncovered([275.0, 290.0, 1995.0, 1975.0]) == 0.0)


class TestDegenerateTables:
    """A table too short to have a cadence still has to answer."""

    def test_an_empty_table_covers_nothing(self):
        """Nothing is known, so every time is infinitely uncovered rather than silently fine."""
        assert np.all(np.isinf(OrbitCoverage.from_met([]).uncovered([0.0, 1.0])))

    def test_a_single_sample_covers_only_itself(self):
        """With no second sample there is no cadence, so the allowance is zero."""
        cov = OrbitCoverage.from_met([100.0])
        assert cov.uncovered([100.0]) == 0.0
        assert cov.uncovered([130.0]) == pytest.approx(30.0)

    def test_duplicate_samples_do_not_collapse_the_cadence(self):
        """Repeated rows are common where two orbit files overlap; they must not read as zero spacing."""
        cov = OrbitCoverage.from_met([0.0, 0.0, 30.0, 30.0, 60.0])
        assert cov.cadence == 30.0


class TestShape:
    """The caller indexes event arrays with the result, so the shape has to survive."""

    def test_the_shape_of_the_input_is_preserved(self):
        """A per-event array in gives a per-event array out, ready to use as a mask."""
        cov = OrbitCoverage.from_met(np.arange(0.0, 300.0, 30.0))
        assert cov.uncovered(np.zeros((3, 4))).shape == (3, 4)
