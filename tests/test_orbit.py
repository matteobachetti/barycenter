"""Tests for the mission-agnostic orbit file reader.

Every mission's orbit file says the same thing in a different dialect. These tests
check that the dialect is translated correctly (units above all, where a factor of 1000
is a 20 ms error) and that the rows a spline cannot use are thrown away.
"""

import os

import astropy.units as u
import numpy as np
from astropy.io import fits

from barycenter.orbit import OrbitSpec, read_orbit

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")

NUSTAR_ORBIT = os.path.join(datadir, "dummy_orb.fits.gz")
XMM_ORBIT = os.path.join(datadir, "dummy_xmm_orb.fits.gz")
CHANDRA_ORBIT = os.path.join(datadir, "dummy_chandra_orb.fits.gz")
SWIFT_ORBIT = os.path.join(datadir, "dummy_swift_orb.fits.gz")
FERMI_ORBIT = os.path.join(datadir, "dummy_fermi_orb.fits.gz")
GBM_ORBIT = os.path.join(datadir, "dummy_gbm_poshist.fits.gz")
SVOM_ORBIT = os.path.join(datadir, "dummy_svom_orb.fits.gz")


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

    def test_xmm_takes_the_equatorial_triple_and_not_the_ecliptic_one(self):
        """XMM's orbit file offers two position triples; only GEI is the right one.

        ``GEI_X/Y/Z`` is geocentric equatorial, which is the frame the ephemeris and the
        source direction are in. ``GSE_X/Y/Z`` is the same vector rotated into the
        Earth-Sun frame, so it has the same length and looks equally plausible -- and
        using it is a 160 ms error. Checking the components, not the magnitude, is the
        only way to tell the two apart.
        """
        table = read_orbit(XMM_ORBIT)
        with fits.open(XMM_ORBIT) as hdul:
            gei = np.column_stack([hdul["ORBIT"].data[f"GEI_{c}"] for c in "XYZ"])
            gse = np.column_stack([hdul["ORBIT"].data[f"GSE_{c}"] for c in "XYZ"])
        ours = np.column_stack([table[c].value for c in "XYZ"])
        assert np.allclose(ours, gei * 1000.0, rtol=0, atol=1e-6)
        assert not np.allclose(ours, gse * 1000.0, rtol=0, atol=1.0)

    def test_chandras_mixed_case_columns_are_read(self):
        """Chandra's ORBITEPHEM extension is read, velocities and all.

        It spells its columns ``Time``, ``X``, ``Vx``, so a case-sensitive lookup would
        find no time column at all -- and if it somehow got past that, no velocity, and
        would differentiate the position instead of using the tabulated one.
        """
        table = read_orbit(CHANDRA_ORBIT)
        with fits.open(CHANDRA_ORBIT) as hdul:
            orbit = hdul["ORBITEPHEM"].data
            assert np.allclose(table["MET"].value, orbit["Time"])
            assert np.allclose(table["X"].value, orbit["X"])
            assert np.array_equal(table["Vx"].value, orbit["Vx"])
        # Metres already, so nothing should have been scaled: 86000-113600 km.
        radius = np.hypot(np.hypot(table["X"], table["Y"]), table["Z"])
        assert np.all((radius > 8.0e7 * u.m) & (radius < 1.2e8 * u.m))

    def test_swifts_prefilter_vectors_are_read_and_converted_from_km(self):
        """Swift's PREFILTER gives POSITION and VELOCITY as vector columns in kilometres.

        The units are the point: Swift tabulates km and km/s where NICER tabulates metres,
        so a missing conversion would put the spacecraft 1000 times too far out and cost
        about 20 ms.  A 585 km orbit is unmistakable at either scale.
        """
        table = read_orbit(SWIFT_ORBIT)
        with fits.open(SWIFT_ORBIT) as hdul:
            orbit = hdul["PREFILTER"].data
            assert np.allclose(table["MET"].value, orbit["TIME"])
            # Column 0 of each vector, scaled by 1000: km in the file, metres in the table.
            assert table["X"].unit == u.m and table["Vx"].unit == u.m / u.s
            assert np.allclose(table["X"].value, orbit["POSITION"][:, 0] * 1000.0)
            assert np.allclose(table["Vx"].value, orbit["VELOCITY"][:, 0] * 1000.0)
        radius = np.hypot(np.hypot(table["X"].value, table["Y"].value), table["Z"].value)
        assert np.all((radius > 6.9e6) & (radius < 7.0e6))

    def test_gbms_position_history_is_read_with_its_own_velocities(self, caplog):
        """A GBM ``poshist`` says TELESCOP=GLAST like a LAT file, and must still be read.

        Its layout is nothing like the LAT spacecraft file's: extension ``GLAST POS
        HIST``, time in ``SCLK_UTC``, scalar ``POS_*`` and ``VEL_*`` columns in metres.
        The velocities are tabulated, so nothing may be differentiated.
        """
        with caplog.at_level("WARNING"):
            table = read_orbit(GBM_ORBIT)
        assert not any("differentiating" in r.message for r in caplog.records)
        with fits.open(GBM_ORBIT) as hdul:
            orbit = hdul["GLAST POS HIST"].data
            assert np.array_equal(table["MET"].value, orbit["SCLK_UTC"])
            assert np.array_equal(table["X"].value, orbit["POS_X"])
            assert np.array_equal(table["Vz"].value, orbit["VEL_Z"])
        # Fermi flies about 530 km up, so metres were not mistaken for kilometres.
        radius = np.hypot(np.hypot(table["X"].value, table["Y"].value), table["Z"].value)
        assert np.all((radius > 6.85e6) & (radius < 6.95e6))

    def test_svoms_float32_vectors_are_read_as_metres(self, caplog):
        """SVOM's ``SVO-ORB-CNV`` holds float32 ``POSITION``/``VELOCITY`` vectors in metres.

        Same column names as Swift and NuSTAR, which tabulate kilometres: reading SVOM's
        as kilometres would put it a thousand Earth radii out. Nothing may be
        differentiated, and the extension name must be the one the spec expects.
        """
        with caplog.at_level("WARNING"):
            table = read_orbit(SVOM_ORBIT)
        assert not caplog.records
        with fits.open(SVOM_ORBIT) as hdul:
            orbit = hdul["SVO-ORB-CNV"].data
            assert np.array_equal(table["MET"].value, orbit["TIME"])
            assert np.array_equal(table["X"].value, orbit["POSITION"][:, 0])
            assert np.array_equal(table["Vz"].value, orbit["VELOCITY"][:, 2])
        radius = np.hypot(np.hypot(table["X"].value, table["Y"].value), table["Z"].value)
        assert np.all((radius > 6.95e6) & (radius < 7.05e6))

    def test_the_lat_spacecraft_file_is_still_read_from_sc_data(self):
        """Adding GBM's layout must not change which one a LAT file is read with."""
        table = read_orbit(FERMI_ORBIT)
        with fits.open(FERMI_ORBIT) as hdul:
            assert np.array_equal(table["MET"].value, hdul["SC_DATA"].data["START"])


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

    def test_columns_are_found_whatever_case_they_are_written_in(self, tmp_path):
        """A file spelling its columns in lower case is read, velocities included.

        Chandra writes ``Time`` and ``Vx``; a case-sensitive lookup would find no
        velocity and quietly differentiate the position instead, which is a degradation
        rather than an error and so would never be noticed.
        """
        met = np.arange(0.0, 600.0, 30.0)
        pos, vel = circular_orbit(met)
        fname = write_fporbit(tmp_path / "lower.fits", met, pos, vel)
        with fits.open(fname, mode="update") as hdul:
            for column in hdul["ORBIT"].columns:
                column.name = column.name.lower()
        table = read_orbit(fname)
        assert np.allclose(table["MET"].value, met)
        # Exactly the tabulated velocity, not a numerical derivative of the position.
        assert np.array_equal(table["Vx"].value, vel[:, 0])


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
