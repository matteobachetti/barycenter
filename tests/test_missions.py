"""Tests for the mission registry.

The registry is the single place that knows anything mission-specific, so the tests here
are mostly about it staying consistent: a wrong entry would be a whole mission silently
mishandled, and there is no data file that would catch it.
"""

import os

import numpy as np
import pytest

from barycenter.missions import MISSIONS, mission_for

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")


class TestLookup:
    def test_matching_is_by_substring(self):
        """``TELESCOP`` is written inconsistently: XTE/RXTE, NuSTAR/NUSTAR, AXAF/CHANDRA."""
        assert mission_for("RXTE") is MISSIONS["rxte"]
        assert mission_for("XTE") is MISSIONS["rxte"]
        assert mission_for("NuSTAR") is MISSIONS["nustar"]
        assert mission_for("NUSTAR") is MISSIONS["nustar"]
        assert mission_for("AXAF") is MISSIONS["chandra"]
        assert mission_for("nicer") is MISSIONS["nicer"]

    def test_unknown_mission_says_how_to_add_one(self):
        """The error message is the documentation: it names the registry to extend."""
        with pytest.raises(ValueError, match="MISSIONS"):
            mission_for("EINSTEIN PROBE")


class TestConsistency:
    """Cheap invariants, so a typo in a new entry fails here rather than in the field."""

    def test_keys_match_names(self):
        """A mismatch would make error messages and log lines name the wrong mission."""
        for key, mission in MISSIONS.items():
            assert mission.name == key

    def test_aliases_are_lower_case_and_unambiguous(self):
        """Matching lower-cases the keyword, so an upper-case alias would never match.

        And no alias may be a substring of another mission's, or which one you get would
        depend on dictionary order.
        """
        for mission in MISSIONS.values():
            for alias in mission.telescop:
                assert alias == alias.lower()
                assert mission_for(alias) is mission

    def test_official_tools_are_ones_we_wrap(self):
        """``official`` names a branch in official.py; anything else raises at run time."""
        for mission in MISSIONS.values():
            assert mission.official in (None, "barycorr", "timeconv")
            if mission.official_ephem is not None:
                assert mission.official is not None

    def test_native_support_means_an_orbit_spec(self):
        """The two must agree, since one is derived from the other."""
        for mission in MISSIONS.values():
            assert mission.has_native_support == (mission.orbit is not None)

    def test_the_missions_with_clocks_are_the_ones_we_implement(self):
        """NuSTAR, RXTE and Swift, and nothing else claims a correction it cannot make."""
        with_clocks = {name for name, m in MISSIONS.items() if m.clock is not None}
        assert with_clocks == {"nustar", "rxte", "swift"}

    def test_only_swift_counts_its_met_in_utc_seconds(self):
        """``met_is_utc`` is opt-in, and setting it wrongly is a whole-second error.

        Swift's MET is UTC seconds since 2001-01-01, so it owes the leap seconds since
        then.  Every other mission here counts TT seconds and owes nothing; Fermi shares
        Swift's MJDREF and may belong in this set, which is why the set is asserted
        exactly rather than just checking Swift is in it.  See docs/known_issues.md.
        """
        counting_utc = {name for name, m in MISSIONS.items() if m.met_is_utc}
        assert counting_utc == {"swift"}


class TestBuilders:
    def test_the_rxte_builder_returns_a_correction_its_file_and_its_accuracy(self):
        """A clock entry has a fixed contract: ``(function, path, accuracy)``."""
        function, path, accuracy = MISSIONS["rxte"].clock(None, "PCA")
        assert path.endswith("tdc.dat")
        assert 17e-6 < function(537723471.0) < 18e-6
        # A constant of the mission, so the span it is asked about is irrelevant.
        assert accuracy(0.0, 1.0) == accuracy(5e8, 6e8) == 5e-6

    def test_the_nustar_builder_uses_the_file_it_is_given(self):
        """Given a clock file, no CALDB lookup happens -- which is what keeps CI offline."""
        clockfile = os.path.join(datadir, "dummy_fine_clk.fits")
        function, path, accuracy = MISSIONS["nustar"].clock(clockfile, None)
        assert path == clockfile
        assert np.all(function(np.array([178574700.0, 178656500.0])) > 0.019)
        # NuSTAR's accuracy is read from the file's own CLOCK_ERR_CORR column, so unlike
        # RXTE's it depends on the span asked about.
        assert 1.2e-4 < accuracy(178574700.0, 178656500.0) < 1.4e-4

    @pytest.mark.parametrize("name", ["nustar", "rxte", "swift"])
    def test_every_builder_returns_the_same_three_things(self, name):
        """One contract for all of them, so core.py needs no per-mission special case."""
        clockfiles = {
            "nustar": os.path.join(datadir, "dummy_fine_clk.fits"),
            "swift": os.path.join(datadir, "dummy_swift_clk.fits"),
            "rxte": None,
        }
        result = MISSIONS[name].clock(clockfiles[name], None)
        assert len(result) == 3
        function, path, accuracy = result
        assert callable(function) and callable(accuracy)
        assert isinstance(path, str)
        assert 0.0 < accuracy(0.0, 1e9) < 1.0
