import os

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

from barycenter import main_barycenter
from barycenter.core import get_coordinates_from_fits_header

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")

#: True where numpy's longdouble is wider than float64 (x86 and most Linux).
#: On Apple Silicon and Windows it is plain float64, and the PINT engine, which
#: subtracts two absolute MJDs, is then quantised at ~1.1 us.
HAS_EXTENDED_PRECISION = np.finfo(np.longdouble).eps < np.finfo(np.float64).eps

#: How close we must get to the official tool, in seconds. The science target is 100 ns,
#: and the default (native) engine reaches it on every platform: what is left is 0, 1 or
#: 2 units in the last place of the reference file's float64 times, 29.8 ns each.
TOLERANCE_S = 1e-7

#: The PINT engine needs a looser one. It sits at +116 ns mean because PINT's Shapiro
#: delay carries an extra 2T*ln(r/AU) annual term that axBary omits, and without an
#: 80-bit longdouble it is quantised at ~1.1 us on top of that, because it subtracts two
#: absolute MJDs. See docs/technical_details.md.
PINT_TOLERANCE_S = 2e-7 if HAS_EXTENDED_PRECISION else 2e-6

#: Coordinates the reference was generated with.  dummy_evt.evt has
#: RA_OBJ=294.91067 and RA_NOM=294.9107; the two differ by 0.1 arcsec, which is
#: 172 us of Roemer delay, so the test must not let the code pick for itself.
REF_RA, REF_DEC = "294.9107", "21.58308"


def assert_times_agree(ours, reference, tolerance=TOLERANCE_S):
    """Assert two arrays of absolute times agree, allowing for float64 granularity.

    A reference file stores times as float64 seconds since MJDREF, so it cannot express a
    difference finer than one unit in the last place: 29.8 ns for NuSTAR's 1.8e8 s and
    119.2 ns for RXTE's 5.4e8 s. Asserting a flat 100 ns would therefore be asserting
    something the reference file is physically unable to record, so the quantisation step
    is added to the tolerance -- and the mean, which averages that noise away, is
    reported in the failure message.
    """
    ulp = np.spacing(np.max(np.abs(reference)))
    diff = np.asarray(ours) - np.asarray(reference)
    assert np.max(np.abs(diff)) < tolerance + ulp, (
        f"max |difference| = {np.max(np.abs(diff)) * 1e9:.1f} ns "
        f"(mean {diff.mean() * 1e9:+.1f} ns, std {diff.std() * 1e9:.1f} ns, "
        f"allowed {(tolerance + ulp) * 1e9:.1f} ns)"
    )


class TestExecution(object):
    @classmethod
    def setup_class(cls):
        cls.orbfile = os.path.join(datadir, "dummy_orb.fits.gz")
        cls.parfile = os.path.join(datadir, "dummy_par.par")
        cls.evfile = os.path.join(datadir, "dummy_evt.evt")
        cls.clkfile = os.path.join(datadir, "dummy_clk.fits")
        cls.fine_clkfile = os.path.join(datadir, "dummy_fine_clk.fits")
        cls.bary_evfile = os.path.join(datadir, "dummy_evt_bary_DE440_noclk.evt.gz")
        cls.bary_clk_evfile = os.path.join(datadir, "dummy_evt_bary_DE440_clk.evt.gz")

    def test_agrees_with_barycorr_with_the_clock_correction(self, tmp_path):
        """With the clock correction on, we still match barycorr to better than 100 ns.

        The clock correction is 25 ms here, and it has to be applied *before* the
        barycentric one -- barycorr evaluates the barycentric correction, and the
        spacecraft position, at the clock-corrected time. Computing
        ``t + clock(t) + bary(t)`` instead was measured at +1146 ns mean and 1878 ns peak
        against this same reference, so this test is what pins the order down.
        """
        outfile = str(tmp_path / "clk.evt")
        main_barycenter(
            [
                self.evfile,
                self.orbfile,
                "-o",
                outfile,
                "--ra",
                REF_RA,
                "--dec",
                REF_DEC,
                "--ephem",
                "DE440",
                "-c",
                self.fine_clkfile,
            ]
        )
        with fits.open(outfile) as hdul, fits.open(self.bary_clk_evfile) as ref:
            assert_times_agree(hdul[1].data["TIME"], ref[1].data["TIME"])
            assert hdul[1].header["CLOCKAPP"] is True

    def test_the_clock_correction_moves_the_times_by_milliseconds(self, tmp_path):
        """Sanity check that the clock file is actually being used.

        Without this, a clock correction silently evaluating to zero would make the test
        above pass for the wrong reason.
        """
        with_clk = str(tmp_path / "with.evt")
        without = str(tmp_path / "without.evt")
        common = [self.evfile, self.orbfile, "--ra", REF_RA, "--dec", REF_DEC, "--ephem", "DE440"]
        main_barycenter([*common, "-o", with_clk, "-c", self.fine_clkfile])
        main_barycenter([*common, "-o", without, "--clockfile", "none"])

        with fits.open(with_clk) as a, fits.open(without) as b:
            shift = a[1].data["TIME"] - b[1].data["TIME"]
            assert np.all((0.019 < shift) & (shift < 0.030)), (shift.min(), shift.max())
            assert b[1].header["CLOCKAPP"] is False

    def test_an_old_format_clock_file_is_refused(self, tmp_path):
        """A pre-2019 NuSTAR clock file stops the run rather than degrading it silently.

        Its polynomial correction is only good to the millisecond, so a file produced
        with it would look clock-corrected while being 10000 times off target.
        """
        with pytest.raises(ValueError, match="NU_FINE_CLOCK"):
            main_barycenter(
                [
                    self.evfile,
                    self.orbfile,
                    "-o",
                    str(tmp_path / "old.evt"),
                    "--ra",
                    REF_RA,
                    "--dec",
                    REF_DEC,
                    "-c",
                    self.clkfile,
                ]
            )

    def test_agrees_with_barycorr(self, tmp_path):
        """Our barycentred times match HEASOFT barycorr to better than 100 ns.

        The reference was made with DE440, ICRS, explicit coordinates and no
        clock correction, and the run below pins all four the same way: anything
        left over is a difference in the solar-system delays themselves.
        """
        outfile = str(tmp_path / "out.evt")
        assert (
            main_barycenter(
                [
                    self.evfile,
                    self.orbfile,
                    "-o",
                    outfile,
                    "--ra",
                    REF_RA,
                    "--dec",
                    REF_DEC,
                    "--ephem",
                    "DE440",
                    "--clockfile",
                    "none",
                ]
            )
            == outfile
        )

        with fits.open(outfile) as hdul, fits.open(self.bary_evfile) as ref:
            assert_times_agree(hdul[1].data["TIME"], ref[1].data["TIME"])
            assert np.isclose(hdul[1].header["RA_OBJ"], ref[1].header["RA_OBJ"])
            assert np.isclose(hdul[1].header["DEC_OBJ"], ref[1].header["DEC_OBJ"])

    def test_pint_engine_also_agrees_with_barycorr(self, tmp_path):
        """--engine pint still reproduces barycorr, to its own looser tolerance.

        The PINT path is kept as an independent second opinion, so it has to keep
        working; its offset is the Shapiro convention, not a mistake.
        """
        outfile = str(tmp_path / "pint.evt")
        main_barycenter(
            [
                self.evfile,
                self.orbfile,
                "-o",
                outfile,
                "--ra",
                REF_RA,
                "--dec",
                REF_DEC,
                "--ephem",
                "DE440",
                "--clockfile",
                "none",
                "--engine",
                "pint",
            ]
        )
        with fits.open(outfile) as hdul, fits.open(self.bary_evfile) as ref:
            assert_times_agree(
                hdul[1].data["TIME"], ref[1].data["TIME"], tolerance=PINT_TOLERANCE_S
            )

    def test_the_two_engines_agree(self, tmp_path):
        """Two independent implementations of the same physics, on the same file.

        They are not identical -- the Shapiro convention differs by ~100 ns, and PINT
        loses another microsecond where there is no extended precision -- but a
        disagreement beyond that would mean one of them has a bug.
        """
        tolerance = 2e-7 if HAS_EXTENDED_PRECISION else 2e-6
        files = {}
        for engine in ("native", "pint"):
            files[engine] = str(tmp_path / f"{engine}.evt")
            main_barycenter(
                [
                    self.evfile,
                    self.orbfile,
                    "-o",
                    files[engine],
                    "--ra",
                    REF_RA,
                    "--dec",
                    REF_DEC,
                    "--ephem",
                    "DE440",
                    "--clockfile",
                    "none",
                    "--engine",
                    engine,
                ]
            )
        with fits.open(files["native"]) as a, fits.open(files["pint"]) as b:
            diff = a[1].data["TIME"] - b[1].data["TIME"]
        assert np.max(np.abs(diff)) < tolerance, f"max {np.max(np.abs(diff)) * 1e9:.1f} ns"

    def test_gtis_corrected_like_the_events(self, tmp_path):
        """The GTIs get the same correction as the events that fall inside them.

        A GTI left on the spacecraft clock while the events move to the
        barycentre silently truncates the data by up to ~500 s.
        """
        outfile = str(tmp_path / "gti.evt")
        main_barycenter(
            [
                self.evfile,
                self.orbfile,
                "-o",
                outfile,
                "--ra",
                REF_RA,
                "--dec",
                REF_DEC,
                "--clockfile",
                "none",
            ]
        )
        with fits.open(outfile) as hdul, fits.open(self.bary_evfile) as ref:
            assert np.allclose(
                hdul["GTI"].data["START"], ref["GTI"].data["START"], rtol=0, atol=TOLERANCE_S
            )
            assert np.allclose(
                hdul["GTI"].data["STOP"], ref["GTI"].data["STOP"], rtol=0, atol=TOLERANCE_S
            )

    def test_a_lower_case_time_column_is_still_corrected(self, tmp_path):
        """A file spelling its time column ``time`` gets the same correction as ``TIME``.

        FITS column names are case-insensitive, and Chandra uses that freedom. A
        case-sensitive lookup skips such a column while still correcting the capitalised
        START/STOP of the GTI beside it, so the failure mode is not an error but a file
        whose events and good-time intervals are on different time scales.
        """
        lower = str(tmp_path / "lower.evt")
        with fits.open(self.evfile) as hdul:
            hdul[1].columns["TIME"].name = "time"
            assert "START" in hdul["GTI"].data.names, "the GTI must stay capitalised"
            hdul.writeto(lower)

        outfile = str(tmp_path / "lower_bary.evt")
        main_barycenter(
            [
                lower,
                self.orbfile,
                "-o",
                outfile,
                "--ra",
                REF_RA,
                "--dec",
                REF_DEC,
                "--clockfile",
                "none",
            ]
        )
        with fits.open(outfile) as hdul, fits.open(self.bary_evfile) as ref:
            assert_times_agree(hdul[1].data["time"], ref[1].data["TIME"])
            assert_times_agree(hdul["GTI"].data["START"], ref["GTI"].data["START"])

    def test_several_orbit_files(self, tmp_path):
        """Passing a list of orbit files works, and repeated entries are dropped.

        The same file given twice must give exactly the answer it gives once:
        this exercises the multi-file branch of load_orbit and its
        de-duplication, which the spline interpolation depends on.
        """
        one = str(tmp_path / "one.evt")
        two = str(tmp_path / "two.evt")
        main_barycenter([self.evfile, self.orbfile, "-o", one, "--clockfile", "none"])
        main_barycenter([self.evfile, self.orbfile, self.orbfile, "-o", two, "--clockfile", "none"])

        with fits.open(one) as h1, fits.open(two) as h2:
            assert np.array_equal(h1[1].data["TIME"], h2[1].data["TIME"])

    def test_overwrite(self, tmp_path):
        """An existing output file is refused unless --overwrite is given."""
        outfile = str(tmp_path / "over.evt")
        main_barycenter([self.evfile, self.orbfile, "-o", outfile, "--clockfile", "none"])

        with pytest.raises(Exception, match="already exists"):
            main_barycenter(
                [
                    self.evfile,
                    self.orbfile,
                    "-p",
                    self.parfile,
                    "-o",
                    outfile,
                    "--clockfile",
                    "none",
                ]
            )
        main_barycenter(
            [
                self.evfile,
                self.orbfile,
                "-p",
                self.parfile,
                "-o",
                outfile,
                "--clockfile",
                "none",
                "--overwrite",
            ]
        )

    def test_barycorr_slim(self, tmp_path):
        """--only-columns keeps TIME plus the named columns and drops the rest."""
        outfile = str(tmp_path / "slim.evt")
        main_barycenter(
            [
                self.evfile,
                self.orbfile,
                "-p",
                self.parfile,
                "-o",
                outfile,
                "--clockfile",
                "none",
                "--only-columns",
                "PI,PRIOR",
            ]
        )
        assert os.path.exists(outfile)

        with fits.open(outfile) as hdul:
            assert "PRIOR" in hdul[1].data.names
            assert "TIME" in hdul[1].data.names
            assert "NUMRISE" not in hdul[1].data.names

    @pytest.mark.remote_data
    @pytest.mark.parametrize("prefix", ["s3://nasa-heasarc/", "https://heasarc.gsfc.nasa.gov/FTP/"])
    def test_barycorr_remote(self, prefix, tmp_path):
        coord = SkyCoord.from_name("M82 X-2")
        ra, dec = coord.ra.deg, coord.dec.deg
        infile = f"{prefix}nustar/data/obs/07/3/30702012003/event_cl/nu30702012003A06_cl.evt.gz"
        orbfile = f"{prefix}nustar/data/obs/07/3/30702012003/event_cl/nu30702012003A.attorb.gz"

        outfile = str(tmp_path / "remote.evt")
        main_barycenter([infile, orbfile, "-o", outfile, "--ra", str(ra), "--dec", str(dec)])
        with fits.open(outfile) as hdul:
            assert np.isclose(hdul[1].header["RA_OBJ"], ra)
            assert np.isclose(hdul[1].header["DEC_OBJ"], dec)
            assert "bary" in hdul[1].header.comments["RA_OBJ"].lower()


class TestRXTE:
    """RXTE, whose clock correction comes from an ASCII coefficient file, not FITS.

    The dataset is a PCA observation of PSR B1509-58 with its FPorbit file, decimated to
    404 events spanning the full hour so the orbit interpolation is exercised over a
    whole RXTE orbit.
    """

    @classmethod
    def setup_class(cls):
        cls.evfile = os.path.join(datadir, "dummy_xte_evt.evt")
        cls.orbfile = os.path.join(datadir, "dummy_xte_orb.fits.gz")
        cls.bary_noclk = os.path.join(datadir, "dummy_xte_bary_DE440_noclk.evt.gz")
        cls.bary_clk = os.path.join(datadir, "dummy_xte_bary_DE440_clk.evt.gz")
        cls.ra, cls.dec = "228.481995", "-59.136002"

    def run(self, outfile, *extra):
        return main_barycenter(
            [
                self.evfile,
                self.orbfile,
                "-o",
                outfile,
                "--ra",
                self.ra,
                "--dec",
                self.dec,
                "--ephem",
                "DE440",
                *extra,
            ]
        )

    def test_agrees_with_barycorr_without_the_clock_correction(self, tmp_path):
        """The solar-system delays alone match barycorr on RXTE.

        The same test as for NuSTAR, on a mission whose orbit file has three scalar
        position columns instead of one vector column, and whose times are three times
        larger -- so this is also what caught the spurious TIMEPIXR half-bin shift, which
        moved every PCA event 477 ns.
        """
        outfile = str(tmp_path / "noclk.evt")
        assert self.run(outfile, "--clockfile", "none") == outfile
        with fits.open(outfile) as hdul, fits.open(self.bary_noclk) as ref:
            assert_times_agree(hdul[1].data["TIME"], ref[1].data["TIME"])
            assert hdul[1].header["CLOCKAPP"] is False

    def test_agrees_with_barycorr_with_the_clock_correction(self, tmp_path):
        """With tdc.dat applied we still match barycorr, which applies it too.

        No clock file is named: barycorr ignores its own ``clockfile`` parameter for RXTE
        and reads ``tdc.dat``, and so do we, from the copy bundled with the package.
        """
        outfile = str(tmp_path / "clk.evt")
        self.run(outfile)
        with fits.open(outfile) as hdul, fits.open(self.bary_clk) as ref:
            assert_times_agree(hdul[1].data["TIME"], ref[1].data["TIME"])
            assert hdul[1].header["CLOCKAPP"] is True

    def test_the_clock_correction_matches_the_constant_barycorr_froze(self):
        """Our reading of tdc.dat reproduces the number HEASOFT computed, to a few ns.

        barycorr evaluates the RXTE correction once, at the middle of the observation, and
        folds that single number into TIMEZERO, so the difference between the two
        references *is* that constant, and comparing our coefficient file reader against
        it directly is far sharper than comparing output files: the corrected times are
        5.4e8 s, where one float64 step is 119 ns, so a 17 us difference between two of
        them is only known to within that step.

        This is what pins down the PCA's extra 16 us detector delay, which the two tests
        above -- each using one reference only -- could not tell from a bad ephemeris.
        """
        from barycenter.clock import rxte_clock_correction_fun

        with fits.open(self.bary_clk) as clk, fits.open(self.bary_noclk) as noclk:
            barycorr_constant = np.mean(clk[1].data["TIME"] - noclk[1].data["TIME"])
        header = fits.getheader(self.evfile, 1)
        middle = (header["TSTART"] + header["TSTOP"]) / 2

        ours = rxte_clock_correction_fun(instrument=header["INSTRUME"])(middle)
        # Tens of microseconds, and the right tens: not zero, and not the 33 us that
        # forgetting the detector delay would give.
        assert 17e-6 < barycorr_constant < 18e-6
        assert abs(ours - barycorr_constant) < 5e-9, (
            f"ours {ours * 1e6:.4f} us vs barycorr {barycorr_constant * 1e6:.4f} us"
        )

    def test_a_clock_file_passed_for_rxte_is_refused_politely(self, tmp_path):
        """Naming a clock file for RXTE warns and uses tdc.dat, rather than failing.

        It is what HEASOFT does with the parameter, and a pipeline that passes
        ``--clockfile`` for every mission should not break on this one.
        """
        outfile = str(tmp_path / "warned.evt")
        with pytest.warns(UserWarning, match="tdc.dat"):
            self.run(outfile, "-c", os.path.join(datadir, "dummy_fine_clk.fits"))
        with fits.open(outfile) as hdul, fits.open(self.bary_clk) as ref:
            assert_times_agree(hdul[1].data["TIME"], ref[1].data["TIME"])


class TestXMM:
    """XMM-Newton, the mission HEASOFT cannot do at all.

    ``barycorr`` refuses an XMM file outright ("Invalid Observatory/Spacecraft position
    vector"), and SAS ``barycen`` needs a full SAS installation and an ingested ODF, so
    this is the first mission where the native engine is not a convenience but the only
    practical route. The reference was made with SAS 22.1 ``barycen`` and DE430; see
    ``tools/make_test_data.py``.

    The dataset is EPIC-pn observation 0112290201 (M82), decimated to 401 events spanning
    the full 7.6 h, with its PPS orbit file thinned to one sample every 10 s.
    """

    @classmethod
    def setup_class(cls):
        cls.evfile = os.path.join(datadir, "dummy_xmm_evt.evt")
        cls.orbfile = os.path.join(datadir, "dummy_xmm_orb.fits.gz")
        cls.reference = os.path.join(datadir, "dummy_xmm_bary_DE430.evt.gz")
        # M82 X-2, which matches none of the file's own RA_/DEC_ keywords: reading a
        # keyword instead of these would be a 100 us error rather than no error at all.
        cls.ra, cls.dec = "148.96267", "69.67931"

    def run(self, outfile, *extra):
        return main_barycenter(
            [
                self.evfile,
                self.orbfile,
                "-o",
                outfile,
                "--ra",
                self.ra,
                "--dec",
                self.dec,
                # Not the default: DE405 would be 1.7 us out here and DE200 1.8 ms out.
                "--ephem",
                "DE430",
                *extra,
            ]
        )

    def test_agrees_with_barycen(self, tmp_path):
        """Our barycentred times match SAS barycen to better than 100 ns on XMM.

        The orbit comes from the PPS ``ORBTSR`` product while barycen reads the ODF's
        ``ROS.ASC``, so passing this also says the two describe the same orbit -- and
        since XMM's is a 48 h eccentric orbit reaching 110000 km, the spacecraft term is
        0.37 s here rather than the 20 ms of a low Earth orbit.
        """
        outfile = str(tmp_path / "xmm.evt")
        assert self.run(outfile) == outfile
        with fits.open(outfile) as hdul, fits.open(self.reference) as ref:
            assert_times_agree(hdul["EVENTS"].data["TIME"], ref["EVENTS"].data["TIME"])
            assert hdul["EVENTS"].header["TIMESYS"] == "TDB"
            assert hdul["EVENTS"].header["TIMEREF"] == "SOLARSYSTEM"

    def test_gtis_agree_with_barycen(self, tmp_path):
        """The GTI boundaries match too, not just the events.

        ``barycen`` corrects its GTI tables when ``processgtis=yes``, so the reference has
        a corrected copy to compare against -- which no other mission's reference gives
        us, because ``barycorr`` folds its correction into ``TIMEZERO`` instead.
        """
        outfile = str(tmp_path / "xmm.evt")
        self.run(outfile)
        with fits.open(outfile) as hdul, fits.open(self.reference) as ref:
            for column in ("START", "STOP"):
                assert_times_agree(hdul["STDGTI01"].data[column], ref["STDGTI01"].data[column])

    def test_tstart_and_tstop_agree_to_barycens_keyword_precision(self, tmp_path):
        """TSTART and TSTOP agree to a microsecond, which is all the reference can say.

        SAS writes a floating-point keyword with 15 significant digits, and at XMM's
        1.06e8 s that leaves six decimals -- so the reference's ``TSTART`` is quantised at
        1 us, a hundred times coarser than the binary ``TIME`` column beside it. The two
        keywords come out 373 ns and -179 ns from ours, both inside half a step, so
        asserting 100 ns here would be asserting something the file cannot record.
        """
        outfile = str(tmp_path / "xmm.evt")
        self.run(outfile)
        with fits.open(outfile) as hdul, fits.open(self.reference) as ref:
            for keyword in ("TSTART", "TSTOP"):
                assert abs(hdul["EVENTS"].header[keyword] - ref["EVENTS"].header[keyword]) < 1e-6

    def test_no_clock_correction_is_applied(self, tmp_path):
        """XMM needs none: the time correlation is applied when the ODF is ingested.

        So ``CLOCKAPP`` comes out false without ``--clockfile none`` having to be asked
        for, and the times still match a reference that had no clock correction either.
        """
        outfile = str(tmp_path / "xmm.evt")
        self.run(outfile)
        with fits.open(outfile) as hdul:
            assert hdul["EVENTS"].header["CLOCKAPP"] is False


class TestChandra:
    """Chandra, the other mission HEASOFT cannot do, and the only one with two references.

    ``barycorr`` fails on a Chandra orbit file -- "no bracketing sample found", then
    "Invalid Observatory/Spacecraft position vector", on a file that brackets the time
    comfortably -- because ``hdaxbary`` carries orbit readers for RXTE, NICER, Swift and
    NuSTAR only. The mission's own tool is CIAO ``axbary``.

    ``axbary`` picks the ephemeris from the reference frame and can reach only two
    combinations, so both are committed: ``refframe=ICRS`` reads DE405 and
    ``refframe=FK5`` reads DE200. That pair is the point of this class -- it is the only
    place where the ephemeris/frame pairing is checked against a real tool rather than
    against ourselves.

    The dataset is ACIS-S observation 10026 (M82), decimated to 401 events spanning the
    full 5.6 h, with the orbit ephemeris beside it at its own 300 s sampling.
    """

    @classmethod
    def setup_class(cls):
        cls.evfile = os.path.join(datadir, "dummy_chandra_evt.evt")
        cls.orbfile = os.path.join(datadir, "dummy_chandra_orb.fits.gz")
        cls.de405 = os.path.join(datadir, "dummy_chandra_bary_DE405.evt.gz")
        cls.de200 = os.path.join(datadir, "dummy_chandra_bary_DE200.evt.gz")
        # RA_TARG/DEC_TARG of the observation, passed explicitly like everywhere else:
        # RA_NOM is 324 arcsec away, which is 0.8 s of Roemer delay.
        cls.ra, cls.dec = "148.959167", "69.679722"

    def run(self, outfile, ephem):
        return main_barycenter(
            [
                self.evfile,
                self.orbfile,
                "-o",
                outfile,
                "--ra",
                self.ra,
                "--dec",
                self.dec,
                "--ephem",
                ephem,
                "--clockfile",
                "none",
            ]
        )

    def test_agrees_with_axbary(self, tmp_path):
        """Our DE405 times match CIAO axbary to better than 100 ns on Chandra.

        Chandra's is a 63 h orbit reaching 113000 km, so the spacecraft term is much
        larger than a low-Earth-orbit mission's -- and the event times are in a column
        called ``time``, which is what makes this file worth having.
        """
        outfile = str(tmp_path / "chandra.evt")
        assert self.run(outfile, "DE405") == outfile
        with fits.open(outfile) as hdul, fits.open(self.de405) as ref:
            assert_times_agree(hdul["EVENTS"].data["time"], ref["EVENTS"].data["time"])
            assert hdul["EVENTS"].header["TIMESYS"] == "TDB"
            assert hdul["EVENTS"].header["TIMEREF"] == "SOLARSYSTEM"

    def test_de200_agrees_too_which_checks_the_frame_pairing(self, tmp_path):
        """DE200 matches the FK5 reference, confirming the ephemeris/frame pairing.

        DE200 is referred to FK5 and DE405 onwards to ICRS, and the code pairs them
        automatically. Getting it wrong is not subtle -- DE405 read in FK5 is +11.2 us
        and DE200 read in ICRS -11.1 us -- but no other reference can catch it, because
        every other official tool here was run in one frame only.
        """
        outfile = str(tmp_path / "chandra200.evt")
        self.run(outfile, "DE200")
        with fits.open(outfile) as hdul, fits.open(self.de200) as ref:
            assert_times_agree(hdul["EVENTS"].data["time"], ref["EVENTS"].data["time"])

    def test_the_two_references_really_are_different_ephemerides(self):
        """The DE405 and DE200 references differ by the 1.8 ms DE200 is known to cost.

        Without this, two references accidentally made with the same settings would make
        the test above pass while checking nothing.
        """
        with fits.open(self.de405) as a, fits.open(self.de200) as b:
            diff = a["EVENTS"].data["time"] - b["EVENTS"].data["time"]
        assert np.allclose(diff, -1.8003e-3, rtol=0, atol=1e-6)

    def test_gtis_agree_with_axbary(self, tmp_path):
        """The GTI boundaries match the reference as well as the events.

        On this file the GTI columns are ``START``/``STOP`` in capitals while the events
        are ``time`` in lower case, so a case-sensitive column lookup passes this test
        and fails the one above -- which is exactly the failure this pair is here for.
        """
        outfile = str(tmp_path / "chandra.evt")
        self.run(outfile, "DE405")
        with fits.open(outfile) as hdul, fits.open(self.de405) as ref:
            for column in ("START", "STOP"):
                assert_times_agree(hdul["GTI"].data[column], ref["GTI"].data[column])

    def test_the_lower_case_time_column_actually_moved(self, tmp_path):
        """The events are shifted by the ~52 s the correction is worth, not left alone.

        A column that is not found is not an error: it is simply not corrected. Asserting
        agreement with the reference would catch that, but only as a mysterious 52 s
        disagreement, so this says plainly what went wrong. The 0.6 s window is the
        Roemer delay's own drift across the 5.6 h exposure.
        """
        outfile = str(tmp_path / "chandra.evt")
        self.run(outfile, "DE405")
        with fits.open(outfile) as hdul, fits.open(self.evfile) as orig:
            shift = hdul["EVENTS"].data["time"] - orig["EVENTS"].data["time"]
        assert np.all(np.abs(shift + 52.46) < 0.6)


class TestSwift:
    """Swift, the only mission here whose MET counts UTC seconds.

    Two references, differing only in whether ``barycorr`` was given the clock file, and
    that pair is the point of the class. Swift's clock correction is the UTC correction
    factor, tens of seconds rather than the microseconds of a NuSTAR fine clock file, and
    it is *not* the only whole-second term: MJDREFF = 0.00074287037 is 64.184 s, which is
    TT - UTC at 2001-01-01, so the MET is UTC seconds since then and owes the leap seconds
    accumulated since -- 4 s for a December 2015 observation. ``barycorr`` adds those
    whatever ``clockfile`` says, so both references carry them and the pair shows that we
    do too.

    The dataset is XRT photon-counting observation 00037258040 of Mrk 421, decimated to
    492 events over its 6.4 ks, with the prefilter beside it at 2 s sampling.
    """

    @classmethod
    def setup_class(cls):
        cls.evfile = os.path.join(datadir, "dummy_swift_evt.evt")
        cls.orbfile = os.path.join(datadir, "dummy_swift_orb.fits.gz")
        cls.clockfile = os.path.join(datadir, "dummy_swift_clk.fits")
        cls.noclk = os.path.join(datadir, "dummy_swift_bary_DE440_noclk.evt.gz")
        cls.clk = os.path.join(datadir, "dummy_swift_bary_DE440_clk.evt.gz")
        # The target position, passed explicitly as everywhere else in these tests.
        cls.ra, cls.dec = "182.635833", "39.405833"

    def run(self, outfile, clockfile):
        return main_barycenter(
            [
                self.evfile,
                self.orbfile,
                "-o",
                outfile,
                "--ra",
                self.ra,
                "--dec",
                self.dec,
                "--ephem",
                "DE440",
                "--clockfile",
                clockfile,
            ]
        )

    def test_agrees_with_barycorr_without_a_clock_file(self, tmp_path):
        """With ``--clockfile none`` we match barycorr, which means the 4 s is still there.

        This is the test that would fail loudest if the leap-second term were tied to the
        clock correction: the reference has it and a naive reading of the MET does not, so
        the disagreement would be 4 s rather than 40 ns.
        """
        outfile = str(tmp_path / "swift_noclk.evt")
        assert self.run(outfile, "none") == outfile
        with fits.open(outfile) as hdul, fits.open(self.noclk) as ref:
            assert_times_agree(hdul["EVENTS"].data["TIME"], ref["EVENTS"].data["TIME"])
            assert hdul["EVENTS"].header["TIMESYS"] == "TDB"
            assert hdul["EVENTS"].header["CLOCKAPP"] is False

    def test_agrees_with_barycorr_with_the_clock_file(self, tmp_path):
        """The UTCF is read and applied to the same 100 ns, and CLOCKAPP says so."""
        outfile = str(tmp_path / "swift_clk.evt")
        self.run(outfile, self.clockfile)
        with fits.open(outfile) as hdul, fits.open(self.clk) as ref:
            assert_times_agree(hdul["EVENTS"].data["TIME"], ref["EVENTS"].data["TIME"])
            assert hdul["EVENTS"].header["CLOCKAPP"] is True

    def test_gtis_agree_with_barycorr(self, tmp_path):
        """The GTI boundaries get the clock and leap-second terms as well as the events."""
        outfile = str(tmp_path / "swift_clk.evt")
        self.run(outfile, self.clockfile)
        with fits.open(outfile) as hdul, fits.open(self.clk) as ref:
            for column in ("START", "STOP"):
                assert_times_agree(hdul["GTI"].data[column], ref["GTI"].data[column])

    def test_the_two_references_differ_by_the_utcf(self):
        """The references really were made with and without the clock file: 15.56 s apart.

        Two references accidentally made with the same settings would let both tests above
        pass while checking nothing -- and on Swift the difference is large enough to see
        from across the room, which is exactly why it must not be left unasserted.
        """
        with fits.open(self.clk) as a, fits.open(self.noclk) as b:
            diff = a["EVENTS"].data["TIME"] - b["EVENTS"].data["TIME"]
        assert np.all((diff > -15.56) & (diff < -15.55))

    def test_the_leap_seconds_survive_switching_the_clock_file_off(self, tmp_path):
        """Both runs are shifted by the +4 s, so no flag can silently lose a whole second.

        Without the clock file the shift is +64.26 to +64.76 s and with it +48.70 to
        +49.20 s; the 0.5 s span in each is the Roemer delay's own drift across the 6.4 ks
        exposure, and the two windows differ by the UTCF and by nothing else. Had the
        leap-second term been tied to the clock correction, the first window would sit near
        +60.5 s instead -- a whole second out, four times over.
        """
        with fits.open(self.evfile) as orig:
            before = np.array(orig["EVENTS"].data["TIME"])
        shifts = {}
        for tag, clockfile in (("none", "none"), ("file", self.clockfile)):
            outfile = str(tmp_path / f"swift_{tag}.evt")
            self.run(outfile, clockfile)
            with fits.open(outfile) as hdul:
                shifts[tag] = np.array(hdul["EVENTS"].data["TIME"]) - before
        assert np.all((shifts["none"] > 64.2) & (shifts["none"] < 64.8))
        assert np.all((shifts["file"] > 48.6) & (shifts["file"] < 49.3))

    def test_a_clock_file_that_does_not_cover_the_events_is_refused(self, tmp_path):
        """Rather than extrapolating a quadratic, it says the times are not covered.

        The same choice as an old NuSTAR file: the correction is 15 s, so extrapolating it
        off the end of the table is not a small error. Here the file is truncated to the
        intervals before the observation, which is what an out-of-date CALDB looks like.
        """
        from barycenter.clock import SWIFT_CLOCK_EXTENSION

        stale = str(tmp_path / "stale_clk.fits")
        with fits.open(self.clockfile) as hdul:
            table = hdul[SWIFT_CLOCK_EXTENSION]
            keep = np.asarray(table.data["TSTOP"]) < 460000000.0
            trimmed = fits.BinTableHDU(
                data=table.data[keep], header=table.header, name=SWIFT_CLOCK_EXTENSION
            )
            fits.HDUList([hdul[0].copy(), trimmed]).writeto(stale)
        with pytest.raises(ValueError, match="not covered"):
            self.run(str(tmp_path / "swift_stale.evt"), stale)


class TestCoordinateKeywords:
    """The header keywords we take the source position from, and HEASOFT's different order."""

    @staticmethod
    def header(**keywords):
        return fits.Header(keywords)

    def test_the_target_position_wins_over_the_pointing(self):
        """RA_OBJ is preferred even when the pointing keywords HEASOFT prefers are present."""
        hdr = self.header(RA_OBJ=294.91067, DEC_OBJ=21.58308, RA_NOM=294.9107, DEC_NOM=21.58308)
        assert get_coordinates_from_fits_header(hdr) == ("RA_OBJ", "DEC_OBJ")

    def test_chandras_target_keywords_are_used(self):
        """RA_TARG/DEC_TARG is taken when present: Chandra writes no RA_OBJ at all.

        Its RA_NOM can sit arcminutes from the target -- 324 arcsec on the ACIS test
        file, worth 0.8 s of Roemer delay -- so falling through to the pointing would be
        a gross error, not a rounding one.
        """
        hdr = self.header(RA_TARG=148.959167, DEC_TARG=69.679722, RA_NOM=148.87, DEC_NOM=69.6)
        assert get_coordinates_from_fits_header(hdr) == ("RA_TARG", "DEC_TARG")

    @pytest.mark.parametrize(
        "ra_key,dec_key",
        [("RA_TARG", "DEC_TARG"), ("RA_NOM", "DEC_NOM"), ("RA_PNT", "DEC_PNT"), ("RA", "DEC")],
    )
    def test_falls_back_down_the_chain(self, ra_key, dec_key):
        """With no RA_OBJ, each remaining pair in turn is used, down to plain RA/DEC."""
        hdr = self.header(**{ra_key: 294.9107, dec_key: 21.58308})
        assert get_coordinates_from_fits_header(hdr) == (ra_key, dec_key)

    def test_says_so_when_heasoft_would_have_chosen_differently(self, caplog):
        """A material disagreement with barycorr's choice is logged, quantified as a delay."""
        hdr = self.header(RA_OBJ=294.91067, DEC_OBJ=21.58308, RA_NOM=294.9107, DEC_NOM=21.58308)
        get_coordinates_from_fits_header(hdr)
        assert "RA_NOM" in caplog.text
        assert "us of Roemer delay" in caplog.text

    def test_stays_quiet_when_the_keywords_agree(self, caplog):
        """No warning when the target and pointing positions are the same to under a ns."""
        hdr = self.header(RA_OBJ=294.9107, DEC_OBJ=21.58308, RA_NOM=294.9107, DEC_NOM=21.58308)
        get_coordinates_from_fits_header(hdr)
        assert caplog.text == ""

    def test_a_header_with_no_position_says_what_it_looked_for(self):
        """The error names every keyword pair tried, and points at --ra/--dec."""
        with pytest.raises(ValueError, match=r"RA_OBJ/DEC_OBJ.*--ra"):
            get_coordinates_from_fits_header(self.header(TELESCOP="NUSTAR"))
