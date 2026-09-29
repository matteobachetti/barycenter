"""Tests for the mission-agnostic orbit file reader.

Every mission's orbit file says the same thing in a different dialect. These tests
check that the dialect is translated correctly (units above all, where a factor of 1000
is a 20 ms error) and that the rows a spline cannot use are thrown away.
"""

import os

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits

from barycenter.orbit import ORBIT_SPECS, OrbitSpec, read_orbit, spec_for_mission

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")

NUSTAR_ORBIT = os.path.join(datadir, "dummy_orb.fits.gz")


def write_fporbit(path, met, pos, vel=None, telescope="NICER", extname="ORBIT"):
    """Write a minimal FPorbit-shaped file: three scalar position columns, metres."""
    columns = [fits.Column(name="TIME", format="D", array=met)]
    for name, values in zip("XYZ", np.asarray(pos).T):
        columns.append(fits.Column(name=name, format="D", array=values))
    if vel is not None:
        for name, values in zip(("Vx", "Vy", "Vz"), np.asarray(vel).T):
            columns.append(fits.Column(name=name, format="D", array=values))
    hdu = fits.BinTableHDU.from_columns(columns, name=extname)
    hdu.header["TELESCOP"] = telescope
    hdu.header["MJDREFI"] = 56658
    hdu.header["MJDREFF"] = 0.000777592592592593
    hdu.header["TIMESYS"] = "TT"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)
    return str(path)


def circular_orbit(met, radius=7.0e6, period=5800.0):
    """A toy circular orbit and its exact velocity, in metres and metres per second."""
    omega = 2 * np.pi / period
    phase = omega * met
    pos = radius * np.column_stack([np.cos(phase), np.sin(phase), np.zeros_like(phase)])
    vel = radius * omega * np.column_stack([-np.sin(phase), np.cos(phase), np.zeros_like(phase)])
    return pos, vel


class TestRealFile:
    def test_nustar_kilometres_become_metres(self):
        """NuSTAR is the only mission tabulating kilometres; the spec converts it.

        A missed factor of 1000 here puts the spacecraft 7000 times too far out and is
        worth about 20 ms of Roemer delay, so this is the single most valuable check in
        the file.
        """
        table = read_orbit(NUSTAR_ORBIT)
        assert table["X"].unit == u.m and table["Vx"].unit == u.m / u.s
        radius = np.linalg.norm(np.column_stack([table[c] for c in "XYZ"]), axis=1)
        assert np.all((6.8e6 < radius) & (radius < 7.3e6))
        speed = np.linalg.norm(np.column_stack([table[c] for c in ("Vx", "Vy", "Vz")]), axis=1)
        assert np.all((7.0e3 < speed) & (speed < 8.0e3))

    def test_met_and_mjd_tt_describe_the_same_instants(self):
        """The two time columns must not drift apart: each engine reads a different one.

        ``MJD_TT`` is PINT's contract and ``MET`` is the native engine's, so if they
        disagree the two engines silently use different spacecraft positions.
        """
        table = read_orbit(NUSTAR_ORBIT)
        expected = np.longdouble(table.meta["mjdref"]) + np.longdouble(table["MET"].value) / 86400
        assert np.max(np.abs(table["MJD_TT"].value - expected)) * 86400 < 1e-6

    def test_telescope_and_mjdref_reach_the_metadata(self):
        """Downstream code takes MJDREF and the mission name from the table, not the file."""
        table = read_orbit(NUSTAR_ORBIT)
        assert "nustar" in str(table.meta["telescope"]).lower()
        assert 55196 < table.meta["mjdref"] < 55198


class TestSpecLookup:
    def test_matching_is_by_substring(self):
        """``TELESCOP`` is written inconsistently: XTE/RXTE, NuSTAR/NUSTAR."""
        assert spec_for_mission("RXTE") is ORBIT_SPECS["xte"]
        assert spec_for_mission("NuSTAR") is ORBIT_SPECS["nustar"]
        assert spec_for_mission("nicer") is ORBIT_SPECS["nicer"]

    def test_unknown_mission_says_how_to_add_one(self):
        """The error is the documentation: it names the registry to extend."""
        with pytest.raises(ValueError, match="ORBIT_SPECS"):
            spec_for_mission("EINSTEIN PROBE")


class TestCleaning:
    def test_out_of_order_duplicate_and_zero_rows_are_dropped(self, tmp_path):
        """Real orbit files contain all three, and a spline cannot use any of them.

        A repeated time makes the interpolation undefined, and an all-zero placeholder
        position puts the spacecraft at the centre of the Earth: a 6400 km error.
        """
        met = np.array([0.0, 30.0, 30.0, 10.0, 20.0, 40.0])
        pos, vel = circular_orbit(met)
        pos[met == 0.0] = 0.0  # a placeholder row
        table = read_orbit(write_fporbit(tmp_path / "orb.fits", met, pos, vel))
        assert np.allclose(table["MET"].value, [10.0, 20.0, 30.0, 40.0])

    def test_missing_velocity_is_obtained_by_differentiating(self, tmp_path):
        """A file with no velocity columns is still usable, with a warning.

        The velocity only enters the microsecond-sized topocentric term, so a numerical
        derivative is good enough -- but it has to be flagged, because the tabulated
        velocity is always better.
        """
        met = np.arange(0.0, 600.0, 30.0)
        pos, vel = circular_orbit(met)
        table = read_orbit(write_fporbit(tmp_path / "novel.fits", met, pos))
        got = np.column_stack([table["Vx"], table["Vy"]])
        # np.gradient on a 30 s grid recovers the true velocity to about 0.1 per cent
        # in the interior. At the first and last row it is one-sided and several per
        # cent out -- one more reason to prefer the tabulated velocity.
        assert np.allclose(got[1:-1], vel[1:-1, :2], rtol=2e-3)


class TestSeveralFiles:
    def test_files_are_stacked_and_sorted(self, tmp_path):
        """A stack of daily orbit files may be passed straight through, in any order."""
        met = np.arange(0.0, 1200.0, 30.0)
        pos, vel = circular_orbit(met)
        first = write_fporbit(tmp_path / "a.fits", met[:20], pos[:20], vel[:20])
        second = write_fporbit(tmp_path / "b.fits", met[20:], pos[20:], vel[20:])
        table = read_orbit([second, first])
        assert np.allclose(table["MET"].value, met)

    def test_at_metafile_lists_the_files(self, tmp_path):
        """HEASOFT's ``@file`` convention: one file name per line."""
        met = np.arange(0.0, 600.0, 30.0)
        pos, vel = circular_orbit(met)
        first = write_fporbit(tmp_path / "a.fits", met[:10], pos[:10], vel[:10])
        second = write_fporbit(tmp_path / "b.fits", met[10:], pos[10:], vel[10:])
        meta = tmp_path / "files.txt"
        # The trailing newline must not become an empty file name.
        meta.write_text(f"{first}\n{second}\n")
        assert np.allclose(read_orbit(f"@{meta}")["MET"].value, met)


def test_an_explicit_spec_overrides_the_telescope_keyword(tmp_path):
    """Passing a spec is how an unregistered or mislabelled mission is read."""
    met = np.arange(0.0, 300.0, 30.0)
    pos, vel = circular_orbit(met)
    fname = write_fporbit(tmp_path / "odd.fits", met, pos, vel, telescope="SOMETHING NEW")
    spec = OrbitSpec(pos=("X", "Y", "Z"), vel=("Vx", "Vy", "Vz"))
    assert len(read_orbit(fname, spec=spec)) == len(met)
