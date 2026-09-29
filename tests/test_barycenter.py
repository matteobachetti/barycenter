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


class TestCoordinateKeywords:
    """The header keywords we take the source position from, and HEASOFT's different order."""

    @staticmethod
    def header(**keywords):
        return fits.Header(keywords)

    def test_the_target_position_wins_over_the_pointing(self):
        """RA_OBJ is preferred even when the pointing keywords HEASOFT prefers are present."""
        hdr = self.header(RA_OBJ=294.91067, DEC_OBJ=21.58308, RA_NOM=294.9107, DEC_NOM=21.58308)
        assert get_coordinates_from_fits_header(hdr) == ("RA_OBJ", "DEC_OBJ")

    @pytest.mark.parametrize(
        "ra_key,dec_key", [("RA_NOM", "DEC_NOM"), ("RA_PNT", "DEC_PNT"), ("RA", "DEC")]
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
