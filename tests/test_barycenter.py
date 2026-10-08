import os

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits

from barycenter import main_barycenter
from barycenter.core import get_coordinates_from_fits_header
from barycenter.utils import high_precision_mjdref

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

    @pytest.mark.parametrize("clockfile", ["none", "file"])
    def test_utcfinit_is_removed_from_every_hdu(self, tmp_path, clockfile):
        """UTCFINIT does not survive into a TDB file, as barycorr also removes it.

        It means "the UTC correction factor at TSTART", and after barycentring neither
        half of that sentence is true any more: TSTART has moved and TIMESYS is TDB.
        Worse, the UTCF has been folded into the times, so anyone who applied the
        keyword as it stands would move a Swift event another 15.56 s. HEASOFT
        ``barycorr`` walks every HDU deleting it (v1.7, "remove the UTCFINIT keyword
        since leap seconds have now been adjusted for"), and both committed references
        have it gone -- including the one made with ``clockfile=NONE``.
        """
        outfile = str(tmp_path / "swift.evt")
        self.run(outfile, self.clockfile if clockfile == "file" else "none")
        with fits.open(self.evfile) as orig:
            assert "UTCFINIT" in orig[1].header, "the input was supposed to carry it"
        reference = self.clk if clockfile == "file" else self.noclk
        with fits.open(outfile) as hdul, fits.open(reference) as ref:
            for hdu in hdul:
                assert "UTCFINIT" not in hdu.header, f"left in {hdu.name}"
            assert all("UTCFINIT" not in hdu.header for hdu in ref)

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


#: ``(label, event file, orbit file, reference, ra, dec, ephem, extra arguments)`` for
#: every mission with a committed reference, so one fixture can barycentre all of them
#: once and several tests can then read the headers.
DERIVED_CASES = [
    (
        "nustar",
        "dummy_evt.evt",
        "dummy_orb.fits.gz",
        "dummy_evt_bary_DE440_noclk.evt.gz",
        REF_RA,
        REF_DEC,
        "DE440",
        ("--clockfile", "none"),
    ),
    (
        "rxte",
        "dummy_xte_evt.evt",
        "dummy_xte_orb.fits.gz",
        "dummy_xte_bary_DE440_noclk.evt.gz",
        "228.481995",
        "-59.136002",
        "DE440",
        ("--clockfile", "none"),
    ),
    (
        "swift",
        "dummy_swift_evt.evt",
        "dummy_swift_orb.fits.gz",
        "dummy_swift_bary_DE440_noclk.evt.gz",
        "182.635833",
        "39.405833",
        "DE440",
        ("--clockfile", "none"),
    ),
    (
        "chandra",
        "dummy_chandra_evt.evt",
        "dummy_chandra_orb.fits.gz",
        "dummy_chandra_bary_DE405.evt.gz",
        "148.959167",
        "69.679722",
        "DE405",
        ("--clockfile", "none"),
    ),
    (
        "xmm",
        "dummy_xmm_evt.evt",
        "dummy_xmm_orb.fits.gz",
        "dummy_xmm_bary_DE430.evt.gz",
        "148.96267",
        "69.67931",
        "DE430",
        (),
    ),
]

#: The missions whose official tool recomputes the date keywords from the corrected
#: TSTART: HEASOFT ``barycorr`` and CIAO ``axbary``. SAS ``barycen`` shifts the existing
#: string instead, so XMM is handled by its own test.
RECOMPUTING_MISSIONS = ["nustar", "rxte", "swift", "chandra"]


@pytest.fixture(scope="module")
def corrected(tmp_path_factory):
    """Barycentre every mission's test file once; yields ``{label: (ours, reference)}``."""
    outdir = tmp_path_factory.mktemp("derived")
    out = {}
    for label, evt, orb, ref, ra, dec, ephem, extra in DERIVED_CASES:
        outfile = str(outdir / f"{label}.evt")
        main_barycenter(
            [
                os.path.join(datadir, evt),
                os.path.join(datadir, orb),
                "-o",
                outfile,
                "--ra",
                ra,
                "--dec",
                dec,
                "--ephem",
                ephem,
                *extra,
            ]
        )
        out[label] = (outfile, os.path.join(datadir, ref))
    return out


class TestDerivedKeywords:
    """The keywords that are not times but are computed from TSTART and TSTOP.

    Every official tool rewrites some of these and none rewrites all of them, so there
    is no single reference to copy: SAS ``barycen`` updates ``TELAPSE`` where HEASOFT
    ``barycorr`` leaves it 3.4 s stale, and CIAO ``axbary`` updates ``MJD-OBS`` where
    ``barycorr`` does not. What they do agree on is that a file must not leave claiming a
    duration, or a calendar date, that its own corrected ``TSTART`` and ``TSTOP``
    contradict.
    """

    @staticmethod
    def truncate(date):
        """A FITS date string cut back to whole seconds, as the official tools write it."""
        return date.split(".")[0]

    @pytest.mark.parametrize("label", [c[0] for c in DERIVED_CASES])
    def test_telapse_is_the_corrected_span(self, corrected, label):
        """TELAPSE equals the corrected TSTOP minus the corrected TSTART, in every HDU.

        It moves by as much as the two ends' corrections differ -- 3.4 s on the NuSTAR
        test file, 1.6 s on the XMM one -- so leaving it alone makes the file contradict
        itself.
        """
        ours, _ = corrected[label]
        with fits.open(ours) as hdul:
            seen = 0
            for hdu in hdul:
                hdr = hdu.header
                if "TELAPSE" not in hdr:
                    continue
                seen += 1
                assert hdr["TELAPSE"] == pytest.approx(hdr["TSTOP"] - hdr["TSTART"], abs=1e-6)
            if label in ("nustar", "xmm"):
                assert seen > 0, "the test file was supposed to carry TELAPSE"

    def test_barycorr_leaves_telapse_stale_where_we_do_not(self, corrected):
        """The NuSTAR reference itself disagrees with its own TSTART and TSTOP.

        This is the evidence for not copying ``barycorr`` here: its output keeps the
        uncorrected TELAPSE while moving TSTART by 309.5 s and TSTOP by 306.1 s, so the
        keyword is 3.4 s away from the span it claims to describe.
        """
        ours, reference = corrected["nustar"]
        with fits.open(reference) as ref:
            hdr = ref[1].header
            stale = hdr["TELAPSE"] - (hdr["TSTOP"] - hdr["TSTART"])
        assert abs(stale) > 3.0
        with fits.open(ours) as hdul:
            assert hdul[1].header["TELAPSE"] == pytest.approx(
                hdul[1].header["TSTOP"] - hdul[1].header["TSTART"], abs=1e-6
            )

    def test_telapse_matches_barycen(self, corrected):
        """On XMM, where the official tool does update TELAPSE, we land on its value.

        To a microsecond, which is all SAS's 15-significant-digit keyword can record at
        XMM's 1.06e8 s -- the same limit as the TSTART/TSTOP comparison above.
        """
        ours, reference = corrected["xmm"]
        with fits.open(ours) as hdul, fits.open(reference) as ref:
            assert hdul["EVENTS"].header["TELAPSE"] == pytest.approx(
                ref["EVENTS"].header["TELAPSE"], abs=1e-6
            )

    @pytest.mark.parametrize("label", RECOMPUTING_MISSIONS)
    def test_dates_match_the_official_tool_to_the_second(self, corrected, label):
        """DATE-OBS and DATE-END reproduce barycorr's and axbary's strings exactly.

        Both recompute the date from ``MJDREF + TSTART/86400`` rather than shifting the
        string the file came in with, and both truncate to whole seconds. We keep the
        milliseconds, so the comparison is against our string cut back the same way --
        which still pins the formula down completely.
        """
        ours, reference = corrected[label]
        with fits.open(ours) as hdul, fits.open(reference) as ref:
            checked = 0
            for keyword in ("DATE-OBS", "DATE-END"):
                if keyword not in ref[1].header:
                    continue
                checked += 1
                assert self.truncate(hdul[1].header[keyword]) == ref[1].header[keyword]
            assert checked == 2

    def test_mjd_obs_matches_axbary(self, corrected):
        """MJD-OBS is recomputed too, and matches the one tool that also updates it.

        ``axbary`` writes ``MJDREF + TSTART/86400``; ``barycorr`` leaves MJD-OBS at its
        uncorrected value, which on the NuSTAR reference is 3.6 ms wrong.
        """
        ours, reference = corrected["chandra"]
        with fits.open(ours) as hdul, fits.open(reference) as ref:
            assert hdul[1].header["MJD-OBS"] == pytest.approx(ref[1].header["MJD-OBS"], abs=1e-11)

    def test_xmm_dates_differ_from_barycen_by_xmms_own_utc_offset(self, corrected):
        """On XMM we deliberately disagree with barycen, and by a knowable amount.

        XMM writes DATE-OBS in UTC while counting MET in TT seconds since MJDREF, so the
        two were already 63 s apart on the way in. ``barycen`` shifts the string by
        however far TSTART moved and so keeps that inconsistency; we recompute, as the
        other two tools do, which makes the date the date of the time the file actually
        records now that TIMESYS is TDB. The difference is the file's own UTC-to-TT
        offset and nothing else, which is what this pins down.
        """
        from astropy.time import Time

        ours, reference = corrected["xmm"]
        raw_name = os.path.join(datadir, "dummy_xmm_evt.evt")
        with fits.open(ours) as hdul, fits.open(reference) as ref, fits.open(raw_name) as raw:
            for keyword, met_keyword in (("DATE-OBS", "TSTART"), ("DATE-END", "TSTOP")):
                # How far the input's own date string sat from the date of its own MET.
                # The string is truncated to whole seconds, so this has to be taken per
                # keyword rather than once for the file.
                offset = (
                    Time(raw[1].header[keyword], scale="tai").mjd
                    - (raw[1].header["MJDREF"] + raw[1].header[met_keyword] / 86400)
                ) * 86400
                assert -64.2 < offset < -62.0, f"{keyword}: {offset:.3f} s"
                theirs = Time(ref[1].header[keyword], scale="tai")
                mine = Time(hdul[1].header[keyword], scale="tai")
                assert (theirs - mine).sec == pytest.approx(offset, abs=0.001)

    def test_nothing_is_invented_where_the_file_had_nothing(self, corrected):
        """A keyword the input never carried is not added on the way out.

        The Chandra GTI extension has no MJD-OBS and the file has no TELAPSE at all;
        writing either would be making metadata up.
        """
        ours, _ = corrected["chandra"]
        with fits.open(ours) as hdul:
            assert "MJD-OBS" not in hdul["GTI"].header
            assert all("TELAPSE" not in hdu.header for hdu in hdul)

    def test_the_exposure_keywords_are_left_alone(self, corrected):
        """ONTIME, LIVETIME and EXPOSURE are untouched, as every official tool leaves them.

        They are sums of good-time interval lengths, not differences of the file's ends,
        and the corrections at the two edges of one interval differ by microseconds.
        """
        ours, reference = corrected["nustar"]
        with (
            fits.open(ours) as hdul,
            fits.open(os.path.join(datadir, "dummy_evt.evt")) as raw,
            fits.open(reference) as ref,
        ):
            for keyword in ("ONTIME", "LIVETIME", "EXPOSURE"):
                assert hdul[1].header[keyword] == raw[1].header[keyword]
                assert ref[1].header[keyword] == raw[1].header[keyword]


class TestClockAccuracyKeyword:
    """``TIERABSO``, the absolute accuracy of the clock correction.

    ``hdaxbary`` rewrites it on every run that applies a clock correction, and leaves it
    alone otherwise. It is the one keyword here whose value is not derivable from the
    times, so each mission's number needs its own justification: Swift's and RXTE's are
    constants of the correction, NuSTAR's comes out of the clock file's own
    ``CLOCK_ERR_CORR`` column.
    """

    @staticmethod
    def run(mission, clockfile, outfile):
        args = {
            "nustar": ("dummy_evt.evt", "dummy_orb.fits.gz", REF_RA, REF_DEC),
            "swift": ("dummy_swift_evt.evt", "dummy_swift_orb.fits.gz", "182.635833", "39.405833"),
            "rxte": ("dummy_xte_evt.evt", "dummy_xte_orb.fits.gz", "228.481995", "-59.136002"),
        }[mission]
        evt, orb, ra, dec = args
        argv = [
            os.path.join(datadir, evt),
            os.path.join(datadir, orb),
            "-o",
            outfile,
            "--ra",
            ra,
            "--dec",
            dec,
            "--ephem",
            "DE440",
        ]
        # ``None`` means "say nothing about clocks", which is how RXTE gets its bundled
        # coefficients applied: naming a file there is refused.
        if clockfile is not None:
            argv += ["--clockfile", clockfile]
        return main_barycenter(argv)

    @pytest.mark.parametrize(
        "mission, clockfile, reference",
        [
            ("swift", "dummy_swift_clk.fits", "dummy_swift_bary_DE440_clk.evt.gz"),
            ("rxte", None, "dummy_xte_bary_DE440_clk.evt.gz"),
        ],
    )
    def test_the_constant_accuracies_match_hdaxbary_exactly(
        self, tmp_path, mission, clockfile, reference
    ):
        """On Swift and RXTE the value is a constant, and it is HEASOFT's constant.

        10 us for Swift once the UTCF is applied, 5 us for RXTE once ``tdc.dat`` is.
        Both are read off the committed references, which is the only evidence for them
        there is -- the recipe lives in ``hdaxbary``'s C source, which the distributed
        HEASOFT source tarballs do not include. RXTE is run with no ``--clockfile`` at
        all, since its coefficients are bundled and naming a file is refused.
        """
        outfile = str(tmp_path / f"{mission}.evt")
        given = clockfile if clockfile is None else os.path.join(datadir, clockfile)
        self.run(mission, given, outfile)
        with fits.open(outfile) as hdul, fits.open(os.path.join(datadir, reference)) as ref:
            assert hdul[1].header["TIERABSO"] == ref[1].header["TIERABSO"]

    def test_nustars_accuracy_comes_from_the_clock_file(self, tmp_path):
        """NuSTAR's is measured, and is the worst value over the observation.

        ``hdaxbary`` writes 122.9 us, which is ``CLOCK_ERR_CORR`` near ``TSTOP``; we
        write the maximum over the span, 131.8 us, because ``TIERABSO`` describes the
        whole file with one number. The two are the same quantity read two ways and
        differ by 7 per cent, so this asserts ours is in the column's range, is no
        smaller than HEASOFT's, and is within 10 per cent of it -- not that it is equal,
        which it deliberately is not.
        """
        outfile = str(tmp_path / "nustar.evt")
        self.run("nustar", os.path.join(datadir, "dummy_fine_clk.fits"), outfile)
        column = fits.getdata(os.path.join(datadir, "dummy_fine_clk.fits"), 1)["CLOCK_ERR_CORR"]
        theirs = fits.getheader(os.path.join(datadir, "dummy_evt_bary_DE440_clk.evt.gz"), 1)[
            "TIERABSO"
        ]
        with fits.open(outfile) as hdul:
            ours = hdul[1].header["TIERABSO"]
            assert np.min(column) <= ours <= np.max(column)
            assert ours >= theirs
            assert abs(ours - theirs) / theirs < 0.10, f"{ours:.6g} vs {theirs:.6g}"
            # And in every extension, as hdaxbary writes it.
            assert all(hdu.header["TIERABSO"] == ours for hdu in hdul)

    @pytest.mark.parametrize("mission", ["nustar", "swift"])
    def test_no_clock_correction_leaves_the_keyword_alone(self, tmp_path, mission):
        """With ``--clockfile none`` the keyword is not written, and not invented.

        HEASOFT does the same for NuSTAR, whose ``clockfile=NONE`` reference has no
        ``TIERABSO`` at all. For Swift it instead writes 100 s, a constant we do not
        copy: our Swift times still carry the leap-second term, so the figure that would
        describe them is the size of the UTCF we were told not to read -- 15.56 s here,
        not 100 s -- and it is exactly the thing this run cannot know.
        """
        outfile = str(tmp_path / f"{mission}_noclk.evt")
        self.run(mission, "none", outfile)
        source = {"nustar": "dummy_evt.evt", "swift": "dummy_swift_evt.evt"}[mission]
        before = fits.getheader(os.path.join(datadir, source), 1).get("TIERABSO")
        with fits.open(outfile) as hdul:
            assert hdul[1].header.get("TIERABSO") == before


class TestInterpolationGrid:
    """Running with ``dt`` set must not cost accuracy that matters.

    The grid is a pure speed optimisation -- 48x at a million events -- so the test is
    that it still reproduces HEASOFT ``barycorr``, and that the two paths agree with each
    other far inside the 100 ns target.
    """

    evfile = os.path.join(datadir, "dummy_evt.evt")
    orbfile = os.path.join(datadir, "dummy_orb.fits.gz")
    reference = os.path.join(datadir, "dummy_evt_bary_DE440_noclk.evt.gz")

    def run(self, outfile, *extra):
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
                *extra,
            ]
        )
        return outfile

    def test_a_gridded_run_still_matches_barycorr(self, tmp_path):
        """``--dt 5`` reproduces the reference just as the exact path does."""
        out = self.run(str(tmp_path / "grid.evt"), "--dt", "5")
        with fits.open(out) as hdul, fits.open(self.reference) as ref:
            assert_times_agree(hdul[1].data["TIME"], ref[1].data["TIME"])

    def test_the_grid_and_the_exact_path_land_on_the_same_stored_times(self, tmp_path):
        """The two paths differ by the interpolation error and nothing else.

        The correction itself agrees to 1.6 ns at ``dt=5`` -- asserted directly in
        ``tests/test_native.py``. Here the comparison is between *stored* times, which
        are absolute float64 seconds since MJDREF and so quantised at 29.8 ns for this
        file, so that step is what the difference is allowed to be. In practice every
        event but one comes out bit-identical.
        """
        exact = self.run(str(tmp_path / "exact.evt"), "--dt", "0")
        grid = self.run(str(tmp_path / "grid2.evt"), "--dt", "5")
        with fits.open(exact) as a, fits.open(grid) as b:
            ours, theirs = np.asarray(b[1].data["TIME"]), np.asarray(a[1].data["TIME"])
        # One storage step (29.8 ns, added by the helper) plus the 1.6 ns of
        # interpolation error the grid actually costs.
        assert_times_agree(ours, theirs, tolerance=2e-9)
        # And the two are not merely within tolerance: they are the same number almost
        # everywhere, which is what shows the grid is not drifting.
        assert np.count_nonzero(ours != theirs) < 0.02 * len(ours)

    def test_a_small_file_is_not_gridded_by_default(self, tmp_path, caplog):
        """Below the threshold the default run is the exact one, and says so.

        This is what keeps every reference comparison in this file on the
        unapproximated code: the committed test files all have a few hundred events.
        """
        with caplog.at_level("INFO"):
            self.run(str(tmp_path / "default.evt"))
        assert any("at every event" in r.message for r in caplog.records)
        assert not any("grid" in r.message for r in caplog.records)


class TestFermi:
    """Fermi LAT, the only mission here whose spacecraft file has no velocity column.

    ``gtbary`` is the mission's own tool, and unlike ``barycorr`` on XMM or Chandra it
    works -- so this reference is a straight cross-check rather than the only route. What
    makes the file worth having is the orbit reader: the position lives in ``SC_DATA`` as
    a single ``SC_POSITION`` vector, timed by a ``START`` column rather than a ``TIME``
    one, and there is no ``SC_VELOCITY`` beside it, so the velocity is differentiated.
    That is the one committed file exercising that fallback end to end.

    The dataset is the simulated pulsar from the Fermi ScienceTools tutorial, decimated
    to 413 events spanning the full week, with all 70 GTIs and the spacecraft file at its
    native 30 s sampling -- which is not decimated on purpose; see ``trim_fermi_inputs``.
    """

    @classmethod
    def setup_class(cls):
        cls.evfile = os.path.join(datadir, "dummy_fermi_evt.evt")
        cls.orbfile = os.path.join(datadir, "dummy_fermi_orb.fits.gz")
        cls.reference = os.path.join(datadir, "dummy_fermi_bary_DE405.evt.gz")
        # The simulated pulsar's position, passed explicitly like everywhere else here.
        cls.ra, cls.dec = "111.11", "22.22"

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
                "DE405",
                "--clockfile",
                "none",
                *extra,
            ]
        )

    def test_agrees_with_gtbary(self, tmp_path):
        """Our times match Fermi gtbary to one unit in the last place of its own file.

        DE405 because that and DE200 are the only ephemerides gtbary offers, and DE405 is
        the one paired with ICRS.
        """
        outfile = str(tmp_path / "fermi.evt")
        assert self.run(outfile) == outfile
        with fits.open(outfile) as hdul, fits.open(self.reference) as ref:
            assert_times_agree(hdul["EVENTS"].data["TIME"], ref["EVENTS"].data["TIME"])
            assert hdul["EVENTS"].header["TIMESYS"] == "TDB"
            assert hdul["EVENTS"].header["TIMEREF"] == "SOLARSYSTEM"

    def test_gtis_agree_with_gtbary(self, tmp_path):
        """All 70 GTI boundaries move with the events, not just the TIME column."""
        outfile = str(tmp_path / "fermi.evt")
        self.run(outfile)
        with fits.open(outfile) as hdul, fits.open(self.reference) as ref:
            assert len(hdul["GTI"].data) == 70
            assert_times_agree(hdul["GTI"].data["START"], ref["GTI"].data["START"])
            assert_times_agree(hdul["GTI"].data["STOP"], ref["GTI"].data["STOP"])

    def test_tstart_and_tstop_agree_to_gtbarys_keyword_precision(self, tmp_path):
        """TSTART and TSTOP agree to a microsecond, which is all a FITS keyword can say.

        Both are comfortably inside the spacecraft file here, so they are corrected
        rather than clamped -- which is the case the clamping must not disturb.
        """
        outfile = str(tmp_path / "fermi.evt")
        self.run(outfile)
        with fits.open(outfile) as hdul, fits.open(self.reference) as ref:
            for keyword in ("TSTART", "TSTOP"):
                assert abs(hdul["EVENTS"].header[keyword] - ref["EVENTS"].header[keyword]) < 1e-6

    def test_the_velocity_is_differentiated_and_says_so(self, tmp_path, caplog):
        """No SC_VELOCITY column means the position is differentiated, which must be announced.

        Quietly differentiating would be a silent accuracy loss; the reference above shows
        the loss is nil at this sampling, but only because the file is not decimated.
        """
        with caplog.at_level("WARNING"):
            self.run(str(tmp_path / "fermi.evt"))
        assert any("differentiating the position" in r.message for r in caplog.records)

    def test_no_clock_correction_is_applied(self, tmp_path):
        """Fermi has no clock file here, so CLOCKAPP is false and the reference had none either."""
        outfile = str(tmp_path / "fermi.evt")
        self.run(outfile)
        with fits.open(outfile) as hdul:
            assert hdul["EVENTS"].header["CLOCKAPP"] is False

    def test_the_fortran_style_mjdref_is_read_and_the_dates_follow(self, tmp_path, caplog):
        """This file writes MJDREFF as ``7.428703703703703D-4``, a string, not a number.

        Without parsing that, ``MJDREF`` is unreadable and ``DATE-OBS`` silently keeps its
        pre-barycentring value -- leaving the output claiming a start seven minutes before
        its own ``TSTART``. The date lands 66.2 s from gtbary's, which is TT - UTC at this
        epoch and the documented convention difference, not an error; see
        ``update_derived_keywords``.
        """
        outfile = str(tmp_path / "fermi.evt")
        with caplog.at_level("WARNING"):
            self.run(outfile)
        assert not any("no MJDREF" in r.message for r in caplog.records)

        with fits.open(self.evfile) as before, fits.open(outfile) as after:
            assert after["EVENTS"].header["DATE-OBS"] != before["EVENTS"].header["DATE-OBS"]
            # The recomputed date is the date of the corrected TSTART, to the second.
            start = Time(after["EVENTS"].header["DATE-OBS"], format="isot", scale="tt")
            mjdref = high_precision_mjdref(after["EVENTS"].header)
            expected = float(mjdref + after["EVENTS"].header["TSTART"] / 86400)
            assert start.tt.mjd == pytest.approx(expected, abs=1.2e-5)  # ~1 s

    def test_a_tstart_before_the_spacecraft_file_is_clamped(self, tmp_path, caplog):
        """The real Fermi quirk: TSTART is the requested window, which can predate the data.

        A LAT extraction sets TSTART to the start of the window that was asked for, while
        the spacecraft file begins whenever the data does -- which is why gtbary refuses
        such a file outright. Here that is reproduced by moving TSTART an hour earlier:
        the events and GTIs must come out untouched, and only TSTART may move, to the
        first GTI rather than to an extrapolated position.
        """
        shifted = str(tmp_path / "shifted.evt")
        with fits.open(self.evfile) as hdul:
            for hdu in hdul:
                if "TSTART" in hdu.header:
                    hdu.header["TSTART"] -= 3600.0
            hdul.writeto(shifted)

        outfile = str(tmp_path / "clamped.evt")
        with caplog.at_level("WARNING"):
            main_barycenter(
                [shifted, self.orbfile, "-o", outfile]
                + ["--ra", self.ra, "--dec", self.dec, "--ephem", "DE405", "--clockfile", "none"]
            )
        assert any("outside the orbit file" in r.message for r in caplog.records)

        with fits.open(outfile) as hdul, fits.open(self.reference) as ref:
            # The data itself is untouched: only the keyword was out of range.
            assert len(hdul["EVENTS"].data) == len(ref["EVENTS"].data)
            assert_times_agree(hdul["EVENTS"].data["TIME"], ref["EVENTS"].data["TIME"])
            assert_times_agree(hdul["GTI"].data["START"], ref["GTI"].data["START"])
            # And TSTART landed on the corrected first GTI, not an hour before it.
            assert hdul["EVENTS"].header["TSTART"] == pytest.approx(
                hdul["GTI"].data["START"].min(), abs=1e-6
            )


def poshist_in_lat_layout(poshist, outfile):
    """Rewrite a GBM position history as a LAT spacecraft file holding the same numbers.

    ``SC_DATA`` with ``START``/``STOP`` and the position and velocity as vector columns:
    the layout :class:`TestFermi` already pins against ``gtbary``.
    """
    with fits.open(poshist) as hdul:
        data, header = hdul["GLAST POS HIST"].data, hdul["GLAST POS HIST"].header
        met = data["SCLK_UTC"]
        columns = [
            fits.Column(name="START", format="D", array=met),
            fits.Column(name="STOP", format="D", array=met + 1.0),
            fits.Column(
                name="SC_POSITION",
                format="3D",
                array=np.column_stack([data["POS_X"], data["POS_Y"], data["POS_Z"]]),
            ),
            fits.Column(
                name="SC_VELOCITY",
                format="3D",
                array=np.column_stack([data["VEL_X"], data["VEL_Y"], data["VEL_Z"]]),
            ),
        ]
        sc_data = fits.BinTableHDU.from_columns(columns, name="SC_DATA")
        for keyword in ("TELESCOP", "TIMESYS", "TIMEUNIT", "MJDREFI", "MJDREFF", "TSTART", "TSTOP"):
            sc_data.header[keyword] = header[keyword]
        fits.HDUList([fits.PrimaryHDU(header=hdul[0].header), sc_data]).writeto(outfile)
    return str(outfile)


class TestFermiGBM:
    """Fermi GBM, read from its own position history rather than the LAT spacecraft file.

    The LAT file cannot stand in for it: the LAT is off in the South Atlantic Anomaly,
    and its spacecraft file has gaps there -- nine a day, 3.3 hours in all on 2024-03-15
    -- where GBM is already taking data. So GBM needs its own ``poshist`` file, which
    shares ``TELESCOP=GLAST`` with the LAT one and nothing else.

    ``gtbary`` refuses GBM event files, so there is no official reference. The test
    instead shows that the ``poshist`` route gives exactly the times the LAT-layout route
    gives for the same positions -- and that route is the one :class:`TestFermi` checks
    against ``gtbary``. The data are real: 401 events from one hour of NaI 0, with the
    position history at its native 1 s sampling. See ``trim_gbm_inputs``.
    """

    @classmethod
    def setup_class(cls):
        cls.evfile = os.path.join(datadir, "dummy_gbm_evt.evt")
        cls.orbfile = os.path.join(datadir, "dummy_gbm_poshist.fits.gz")
        # Her X-1. GBM event files carry no source position, so it must always be given.
        cls.ra, cls.dec = "254.457625", "35.342361"

    def run(self, orbfile, outfile):
        return main_barycenter(
            [self.evfile, orbfile, "-o", outfile, "--ra", self.ra, "--dec", self.dec]
            + ["--ephem", "DE405", "--clockfile", "none"]
        )

    def test_poshist_gives_the_times_its_lat_layout_gives(self, tmp_path):
        """The same positions in either layout give bit-identical events and GTIs."""
        lat_layout = poshist_in_lat_layout(self.orbfile, tmp_path / "sc.fits")
        ours = self.run(self.orbfile, str(tmp_path / "poshist.evt"))
        theirs = self.run(lat_layout, str(tmp_path / "lat_layout.evt"))
        with fits.open(ours) as a, fits.open(theirs) as b, fits.open(self.evfile) as raw:
            assert np.array_equal(a["EVENTS"].data["TIME"], b["EVENTS"].data["TIME"])
            assert np.array_equal(a["GTI"].data["START"], b["GTI"].data["START"])
            # And the times did move: a barycentred hour is minutes away from the input.
            assert np.all(np.abs(a["EVENTS"].data["TIME"] - raw["EVENTS"].data["TIME"]) > 1)


class TestSVOM:
    """SVOM/ECLAIRs, on a synthetic dataset in the mission's real file layout.

    No tool barycentres SVOM, so the reference is HEASOFT ``barycorr`` run on the same
    events and positions relabelled as a NICER observation, whose orbit layout ``barycorr``
    reads; ``MJDREF``, ``TIMESYS`` and every number are unchanged. See
    ``make_svom_inputs`` and ``svom_as_nicer`` in ``tools/make_test_data.py``.

    The GTIs come in a separate file, as SVOM ships them, and are merged in first with
    :func:`barycenter.gti.add_gti_extension`, so the run is the one a user would do.
    """

    @classmethod
    def setup_class(cls):
        cls.evfile = os.path.join(datadir, "dummy_svom_evt.evt")
        cls.orbfile = os.path.join(datadir, "dummy_svom_orb.fits.gz")
        cls.gtifile = os.path.join(datadir, "dummy_svom_gti.fits")
        cls.reference = os.path.join(datadir, "dummy_svom_bary_DE440.evt.gz")

    def run(self, tmp_path):
        from barycenter.gti import add_gti_extension

        with_gti = add_gti_extension(
            self.evfile,
            self.gtifile,
            ["GTICAL-STA", "GTICAL-NSA", "GTICAL-NEO|GTICAL-PEO|GTICAL-TEO"],
            outfile=str(tmp_path / "gti.evt"),
        )
        return main_barycenter(
            [with_gti, self.orbfile, "-o", str(tmp_path / "bary.evt")]
            + ["--ra", "270.0", "--dec", "-25.0", "--ephem", "DE440"]
        )

    def test_agrees_with_barycorr_events_and_gtis(self, tmp_path):
        """Events and merged GTIs both match barycorr to 100 ns, row by row.

        Row by row also shows the events keep their original, not quite sorted, order.
        No ``--clockfile``: SVOM needs no clock correction and none must be looked for.
        """
        outfile = self.run(tmp_path)
        with fits.open(outfile) as ours, fits.open(self.reference) as ref:
            assert_times_agree(ours[1].data["TIME"], ref[1].data["TIME"])
            assert_times_agree(ours["GTI"].data["START"], ref["GTI"].data["START"])
            assert_times_agree(ours["GTI"].data["STOP"], ref["GTI"].data["STOP"])
            assert ours[1].header["TIMESYS"] == "TDB"
            assert ours["GTI"].header["TIMEREF"] == "SOLARSYSTEM"
            assert np.any(np.diff(ours[1].data["TIME"]) < 0)
