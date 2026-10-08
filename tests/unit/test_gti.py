"""Merging good time intervals from a separate file into an event file."""

import numpy as np
import pytest
from astropy.io import fits

from barycenter.gti import (
    add_gti_extension,
    intersect_gtis,
    main_apply_gti,
    read_gtis,
    union_gtis,
)

MJDREFI, MJDREFF = 57754, 0.000800740741


def gti_hdu(name, intervals, mjdreff=MJDREFF):
    intervals = np.asarray(intervals, dtype=float)
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="START", format="D", array=intervals[:, 0]),
            fits.Column(name="STOP", format="D", array=intervals[:, 1]),
        ],
        name=name,
    )
    hdu.header["MJDREFI"], hdu.header["MJDREFF"] = MJDREFI, mjdreff
    return hdu


@pytest.fixture
def gti_file(tmp_path):
    """A GTI file in the SVOM layout: several extensions, one table that is not a GTI."""
    group = fits.BinTableHDU.from_columns(
        [fits.Column(name="GTI_NAME", format="10A", array=["GTICAL-STA"])], name="GTI-GRP"
    )
    hdus = [
        fits.PrimaryHDU(),
        group,
        gti_hdu("GTICAL-STA", [[0, 100], [200, 300]]),
        gti_hdu("GTICAL-NSA", [[50, 250]]),
        gti_hdu("GTICAL-PEO", [[0, 80], [220, 300]]),
    ]
    path = tmp_path / "gti.fits"
    fits.HDUList(hdus).writeto(path)
    return str(path)


@pytest.fixture
def event_file(tmp_path):
    times = np.array([10.0, 60.0, 90.0, 150.0, 210.0, 240.0, 290.0])
    events = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="TIME", format="D", array=times),
            fits.Column(name="PI", format="I", array=np.arange(times.size)),
        ],
        name="EVENTS",
    )
    for key, value in [
        ("TELESCOP", "SVOM"),
        ("TIMESYS", "TT"),
        ("TIMEREF", "LOCAL"),
        ("TIMEUNIT", "s"),
        ("MJDREFI", MJDREFI),
        ("MJDREFF", MJDREFF),
    ]:
        events.header[key] = value
    path = tmp_path / "events.fits"
    fits.HDUList([fits.PrimaryHDU(), events]).writeto(path)
    return str(path)


class TestSetOperations:
    def test_intersection_keeps_only_time_good_in_every_list(self):
        """Overlaps survive, everything outside any one list is cut, edges are exact."""
        result = intersect_gtis([[[0, 100], [200, 300]], [[50, 250]]])
        assert np.array_equal(result, [[50, 100], [200, 250]])

    def test_union_merges_overlapping_and_touching_intervals(self):
        """Overlapping or touching intervals become one; disjoint ones stay apart."""
        result = union_gtis([[[0, 10], [20, 30]], [[5, 20]], [[40, 50]]])
        assert np.array_equal(result, [[0, 30], [40, 50]])

    def test_an_empty_intersection_is_an_empty_array_not_an_error(self):
        """Disjoint lists intersect to nothing, with the (0, 2) shape callers expect."""
        assert intersect_gtis([[[0, 10]], [[20, 30]]]).shape == (0, 2)


class TestReading:
    def test_extensions_are_read_by_name_whatever_the_case(self, gti_file):
        """Names are matched case-insensitively, as FITS extension names are."""
        gtis = read_gtis(gti_file, ["gtical-sta", "GTICAL-NSA"])
        assert [len(g) for g in gtis] == [2, 1]

    def test_alternatives_take_the_first_one_present(self, gti_file):
        """``A|B|C`` uses the first extension the file has, so one recipe fits all files."""
        (gti,) = read_gtis(gti_file, ["GTICAL-NEO|GTICAL-PEO|GTICAL-TEO"])
        assert np.array_equal(gti, [[0, 80], [220, 300]])

    def test_a_missing_extension_is_an_error_naming_what_is_there(self, gti_file):
        """A typo must not silently drop a constraint; the message lists the choices."""
        with pytest.raises(KeyError, match="GTICAL-NSA"):
            read_gtis(gti_file, ["GTICAL-XXX"])


class TestAddingToEvents:
    def test_the_merged_gti_is_written_with_the_events_timing_keywords(
        self, event_file, gti_file, tmp_path
    ):
        """The new GTI extension is a binary table carrying MJDREF and TIMESYS.

        Without them a barycentred GTI would be on an unknown time scale.
        """
        out = str(tmp_path / "out.fits")
        add_gti_extension(event_file, gti_file, ["GTICAL-STA", "GTICAL-NSA"], outfile=out)
        with fits.open(out) as hdul:
            gti = hdul["GTI"]
            assert isinstance(gti, fits.BinTableHDU)
            assert np.array_equal(gti.data["START"], [50, 200])
            assert np.array_equal(gti.data["STOP"], [100, 250])
            assert gti.header["MJDREFI"] == MJDREFI
            assert gti.header["TIMESYS"] == "TT"
            assert len(hdul["EVENTS"].data) == 7

    def test_filtering_drops_the_events_outside(self, event_file, gti_file, tmp_path):
        """With ``filter_events`` only events inside the merged GTI are kept."""
        out = str(tmp_path / "out.fits")
        add_gti_extension(
            event_file, gti_file, ["GTICAL-STA", "GTICAL-NSA"], outfile=out, filter_events=True
        )
        with fits.open(out) as hdul:
            assert np.array_equal(hdul["EVENTS"].data["TIME"], [60, 90, 210, 240])

    def test_a_different_mjdref_is_refused(self, event_file, tmp_path):
        """GTIs counted from another epoch would be off by the epoch difference."""
        other = tmp_path / "other.fits"
        fits.HDUList([fits.PrimaryHDU(), gti_hdu("GTI2", [[0, 1]], mjdreff=0.5)]).writeto(other)
        with pytest.raises(ValueError, match="MJDREF"):
            add_gti_extension(event_file, str(other), ["GTI2"], outfile=str(tmp_path / "o.fits"))

    def test_an_existing_gti_extension_is_not_overwritten(self, event_file, gti_file, tmp_path):
        """Applying twice must not silently replace the GTIs the file already has."""
        out = str(tmp_path / "out.fits")
        add_gti_extension(event_file, gti_file, ["GTICAL-STA"], outfile=out)
        with pytest.raises(ValueError, match="already has"):
            add_gti_extension(out, gti_file, ["GTICAL-NSA"], outfile=str(tmp_path / "again.fits"))

    def test_command_line(self, event_file, gti_file, tmp_path):
        """The command intersects the listed extensions and returns the output name."""
        out = str(tmp_path / "cli.fits")
        args = [event_file, gti_file, "-e", "GTICAL-STA,GTICAL-NSA,GTICAL-NEO|GTICAL-PEO"]
        assert main_apply_gti(args + ["-o", out]) == out
        with fits.open(out) as hdul:
            assert np.array_equal(hdul["GTI"].data["START"], [50, 220])
            assert np.array_equal(hdul["GTI"].data["STOP"], [80, 250])

    def test_command_line_without_extensions_lists_them(self, event_file, gti_file, capsys):
        """With no ``-e`` the command only lists the GTI extensions and their exposure."""
        assert main_apply_gti([event_file, gti_file]) is None
        printed = capsys.readouterr().out
        assert "GTICAL-PEO" in printed and "GTI-GRP" not in printed
