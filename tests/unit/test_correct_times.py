"""The order the clock, leap-second and barycentric corrections are applied in."""

import numpy as np

from barycenter.core import correct_times


def bary(t):
    """A stand-in barycentric correction with a deliberately non-zero slope.

    A constant one would make every ordering of the three terms give the same answer, so
    it would test nothing.
    """
    return 100.0 + 1e-4 * (np.asarray(t, dtype=float) - 1000.0)


class TestOrdering:
    def test_with_nothing_to_add_the_barycentric_term_is_evaluated_on_the_raw_time(self):
        """The plain case, and the baseline the others are compared against."""
        assert correct_times(1000.0, bary) == 1100.0

    def test_the_clock_correction_goes_first(self):
        """barycorr evaluates the barycentric correction at the clock-corrected time.

        Adding the two independently instead was measured at +1146 ns against barycorr on
        NuSTAR, so this is the assertion that pins the order down.
        """
        got = correct_times(1000.0, bary, clock_fun=lambda t: np.full_like(t, 10.0))
        assert got == 1000.0 + 10.0 + bary(1010.0)

    def test_the_leap_term_also_goes_first(self):
        """A leap-second term shifts the time the barycentric correction is evaluated at."""
        got = correct_times(1000.0, bary, leap_fun=lambda t: np.full_like(t, 4.0))
        assert got == 1000.0 + 4.0 + bary(1004.0)

    def test_both_terms_are_evaluated_on_the_raw_time(self):
        """The clock polynomial is not evaluated at the leap-shifted time.

        On Swift the clock correction drifts 4616 us/day, so evaluating it 4 s later moves
        every event 214 ns -- above the 100 ns target, and not what barycorr does.
        """
        seen = []

        def clock(t):
            seen.append(np.asarray(t, dtype=float).copy())
            return np.full_like(np.asarray(t, dtype=float), -15.0)

        got = correct_times(1000.0, bary, clock_fun=clock, leap_fun=lambda t: np.full_like(t, 4.0))
        assert np.all(seen[0] == 1000.0)
        assert got == 1000.0 - 15.0 + 4.0 + bary(989.0)

    def test_an_array_keeps_its_shape(self):
        """Event times arrive as arrays and must come back as arrays of the same length."""
        times = np.array([1000.0, 2000.0, 3000.0])
        got = correct_times(times, bary, leap_fun=lambda t: np.full_like(t, 4.0))
        assert got.shape == times.shape
