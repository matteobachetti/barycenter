"""Finding the newest clock file in the CALDB, without going to the network."""

import os

import pytest

from barycenter import clock


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Point the clock cache at a temporary directory instead of the user's."""
    monkeypatch.setattr(clock, "clock_cache_dir", lambda: str(tmp_path))
    return tmp_path


class TestClockSourceRegistry:
    """Every mission's URL and filename pattern, which have to agree with each other."""

    def test_every_source_has_a_url_and_a_pattern(self):
        """A half-filled entry would fail only when the network was reached."""
        for name, source in clock.CLOCK_CALDB.items():
            assert source.url.startswith("https://"), name
            assert source.url.endswith("/"), name
            assert "*" in source.pattern, name

    def test_an_unlisted_mission_is_refused_by_name(self):
        """RXTE's coefficients are bundled, not fetched, so asking for them is a mistake."""
        with pytest.raises(ValueError, match="rxte"):
            clock.get_latest_clock_file("rxte")


class TestPickingTheNewest:
    @staticmethod
    def listing(*names):
        return ["https://example.invalid/caldb/" + n for n in names]

    def test_the_highest_version_wins(self, cache, monkeypatch):
        """Files are versioned in the name, so the last in sorted order is the newest."""
        monkeypatch.setattr(
            clock,
            "get_remote_directory_listing",
            lambda url: self.listing(
                "swclockcor20041120v172.fits",
                "swclockcor20041120v174.fits",
                "swclockcor20041120v173.fits",
            ),
        )
        got = []
        monkeypatch.setattr(
            "urllib.request.urlretrieve", lambda url, fname: got.append(url) or (fname, None)
        )
        result = clock.get_latest_clock_file("swift")
        assert os.path.basename(result) == "swclockcor20041120v174.fits"
        assert got == ["https://example.invalid/caldb/swclockcor20041120v174.fits"]

    def test_the_pattern_excludes_what_is_not_a_clock_file(self, cache, monkeypatch):
        """The Swift index also lists `swco.dat` and the column headings of the HTML table.

        Sorting the raw listing would pick one of those, so the pattern is what makes the
        choice right rather than merely last.
        """
        monkeypatch.setattr(
            clock,
            "get_remote_directory_listing",
            lambda url: self.listing(
                "swclockcor20041120v174.fits", "swco.dat", "Size", "Parent%20Directory"
            ),
        )
        monkeypatch.setattr("urllib.request.urlretrieve", lambda url, fname: (fname, None))
        assert os.path.basename(clock.get_latest_clock_file("swift")).startswith("swclockcor")

    def test_a_cached_file_is_not_downloaded_again(self, cache, monkeypatch):
        """Clock files are megabytes, and there is no reason to fetch one per run."""
        (cache / "nuCclock20100101v230.fits").write_text("not really a FITS file")
        monkeypatch.setattr(
            clock,
            "get_remote_directory_listing",
            lambda url: self.listing("nuCclock20100101v230.fits"),
        )

        def refuse(url, fname):
            raise AssertionError("should not have downloaded anything")

        monkeypatch.setattr("urllib.request.urlretrieve", refuse)
        assert clock.get_latest_clock_file("nustar") == str(cache / "nuCclock20100101v230.fits")


class TestOfflineFallback:
    def test_the_newest_cached_file_is_used_when_the_index_is_unreachable(self, cache, monkeypatch):
        """A network failure must not stop a run that already has a usable clock file."""
        for version in ("v230", "v231"):
            (cache / f"nuCclock20100101{version}.fits").write_text("x")

        def explode(url):
            raise OSError("no route to host")

        monkeypatch.setattr(clock, "get_remote_directory_listing", explode)
        with pytest.warns(UserWarning, match="Could not read"):
            result = clock.get_latest_clock_file("nustar")
        assert os.path.basename(result) == "nuCclock20100101v231.fits"

    def test_the_fallback_only_considers_this_mission_s_files(self, cache, monkeypatch):
        """A cached NuSTAR file must not be offered as a Swift one, or vice versa.

        The pattern is what separates them: both missions cache into the same directory,
        and handing back the wrong mission's file would be a silent tens-of-seconds error.
        """
        (cache / "nuCclock20100101v230.fits").write_text("x")

        def explode(url):
            raise OSError("no route to host")

        monkeypatch.setattr(clock, "get_remote_directory_listing", explode)
        with pytest.warns(UserWarning), pytest.raises(FileNotFoundError, match="--clockfile"):
            clock.get_latest_clock_file("swift")


class TestHeasoftDataDirs:
    """A bare clock-file name resolved the way HEASOFT's own tasks resolve it.

    ``barycorr`` looks a named Swift clock file up in ``$TIMING_DIR`` and then
    ``$LHEA_DATA`` before treating it as a path (its Swift branch, lines 347-351), so a
    name that works there has to work here.
    """

    def test_timing_dir_is_searched_first(self, tmp_path, monkeypatch):
        """With the file in both directories, $TIMING_DIR wins, as in barycorr."""
        first, second = tmp_path / "timing", tmp_path / "lhea"
        first.mkdir()
        second.mkdir()
        (first / "clk.fits").write_bytes(b"first")
        (second / "clk.fits").write_bytes(b"second")
        monkeypatch.setenv("TIMING_DIR", str(first))
        monkeypatch.setenv("LHEA_DATA", str(second))
        assert clock.in_heasoft_data_dirs("clk.fits") == str(first / "clk.fits")

    def test_lhea_data_is_the_fallback(self, tmp_path, monkeypatch):
        """A file only in $LHEA_DATA is still found."""
        directory = tmp_path / "lhea"
        directory.mkdir()
        (directory / "clk.fits").write_bytes(b"x")
        monkeypatch.setenv("TIMING_DIR", str(tmp_path / "nowhere"))
        monkeypatch.setenv("LHEA_DATA", str(directory))
        assert clock.in_heasoft_data_dirs("clk.fits") == str(directory / "clk.fits")

    def test_a_name_in_neither_is_none(self, tmp_path, monkeypatch):
        """Nothing found means None, so the caller can raise its own error."""
        monkeypatch.setenv("TIMING_DIR", str(tmp_path))
        monkeypatch.delenv("LHEA_DATA", raising=False)
        assert clock.in_heasoft_data_dirs("clk.fits") is None

    def test_a_named_clock_file_is_looked_up_there(self, tmp_path, monkeypatch):
        """clock_correction_fun resolves a bare name through the HEASOFT directories.

        The file here is the real Swift test clock file under a different name, so the
        lookup is followed by an actual read rather than stopping at the path.
        """
        import shutil

        directory = tmp_path / "timing"
        directory.mkdir()
        source = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        shutil.copy(os.path.join(source, "dummy_swift_clk.fits"), directory / "named_clk.fits")
        monkeypatch.setenv("TIMING_DIR", str(directory))
        monkeypatch.delenv("LHEA_DATA", raising=False)
        _, used, _ = clock.clock_correction_fun("swift", "named_clk.fits")
        assert used == str(directory / "named_clk.fits")

    def test_a_name_nowhere_says_where_it_looked(self, tmp_path, monkeypatch):
        """The error names both environment variables rather than just the file."""
        monkeypatch.setenv("TIMING_DIR", str(tmp_path))
        monkeypatch.delenv("LHEA_DATA", raising=False)
        with pytest.raises(FileNotFoundError, match=r"TIMING_DIR.*LHEA_DATA"):
            clock.clock_correction_fun("swift", "missing_clk.fits")
