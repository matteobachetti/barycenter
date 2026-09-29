import os

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits

from barycenter import main_barycenter

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")

#: True where numpy's longdouble is wider than float64 (x86 and most Linux).
#: On Apple Silicon and Windows it is plain float64, and the PINT engine, which
#: subtracts two absolute MJDs, is then quantised at ~1.1 us.
HAS_EXTENDED_PRECISION = np.finfo(np.longdouble).eps < np.finfo(np.float64).eps

#: How close we must get to the official tool, in seconds.  The science target
#: is 100 ns; the PINT engine sits at +116 ns mean with 60 ns of spread, because
#: PINT's Shapiro delay carries an extra 2T*ln(r/AU) annual term that axBary
#: omits.  Without extended precision the spread grows to ~1.3 us peak to peak
#: for the reason above, so the check is looser there; the native engine
#: evaluates small quantities in float64 and removes the distinction.
#: See docs/technical_details.md.
TOLERANCE_S = 2e-7 if HAS_EXTENDED_PRECISION else 2e-6

#: Coordinates the reference was generated with.  dummy_evt.evt has
#: RA_OBJ=294.91067 and RA_NOM=294.9107; the two differ by 0.1 arcsec, which is
#: 172 us of Roemer delay, so the test must not let the code pick for itself.
REF_RA, REF_DEC = "294.9107", "21.58308"


class TestExecution(object):
    @classmethod
    def setup_class(cls):
        cls.orbfile = os.path.join(datadir, "dummy_orb.fits.gz")
        cls.parfile = os.path.join(datadir, "dummy_par.par")
        cls.evfile = os.path.join(datadir, "dummy_evt.evt")
        cls.clkfile = os.path.join(datadir, "dummy_clk.fits")
        cls.bary_evfile = os.path.join(datadir, "dummy_evt_bary_DE440_noclk.evt.gz")

    def test_agrees_with_barycorr(self, tmp_path):
        """Our barycentred times match HEASOFT barycorr to better than 200 ns.

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
                    "-o", outfile,
                    "--ra", REF_RA,
                    "--dec", REF_DEC,
                    "--ephem", "DE440",
                    "--clockfile", "none",
                ]
            )
            == outfile
        )

        with fits.open(outfile) as hdul, fits.open(self.bary_evfile) as ref:
            diff = hdul[1].data["TIME"] - ref[1].data["TIME"]
            assert np.max(np.abs(diff)) < TOLERANCE_S, (
                f"max |difference| = {np.max(np.abs(diff)) * 1e9:.1f} ns "
                f"(mean {diff.mean() * 1e9:+.1f} ns, std {diff.std() * 1e9:.1f} ns)"
            )
            assert np.isclose(hdul[1].header["RA_OBJ"], ref[1].header["RA_OBJ"])
            assert np.isclose(hdul[1].header["DEC_OBJ"], ref[1].header["DEC_OBJ"])

    def test_gtis_corrected_like_the_events(self, tmp_path):
        """The GTIs get the same correction as the events that fall inside them.

        A GTI left on the spacecraft clock while the events move to the
        barycentre silently truncates the data by up to ~500 s.
        """
        outfile = str(tmp_path / "gti.evt")
        main_barycenter(
            [self.evfile, self.orbfile, "-o", outfile, "--ra", REF_RA, "--dec", REF_DEC,
             "--clockfile", "none"]
        )
        with fits.open(outfile) as hdul, fits.open(self.bary_evfile) as ref:
            assert np.allclose(hdul["GTI"].data["START"], ref["GTI"].data["START"],
                               rtol=0, atol=TOLERANCE_S)
            assert np.allclose(hdul["GTI"].data["STOP"], ref["GTI"].data["STOP"],
                               rtol=0, atol=TOLERANCE_S)

    def test_several_orbit_files(self, tmp_path):
        """Passing a list of orbit files works, and repeated entries are dropped.

        The same file given twice must give exactly the answer it gives once:
        this exercises the multi-file branch of load_orbit and its
        de-duplication, which the spline interpolation depends on.
        """
        one = str(tmp_path / "one.evt")
        two = str(tmp_path / "two.evt")
        main_barycenter([self.evfile, self.orbfile, "-o", one, "--clockfile", "none"])
        main_barycenter(
            [self.evfile, self.orbfile, self.orbfile, "-o", two, "--clockfile", "none"]
        )

        with fits.open(one) as h1, fits.open(two) as h2:
            assert np.array_equal(h1[1].data["TIME"], h2[1].data["TIME"])

    def test_overwrite(self, tmp_path):
        """An existing output file is refused unless --overwrite is given."""
        outfile = str(tmp_path / "over.evt")
        main_barycenter([self.evfile, self.orbfile, "-o", outfile, "--clockfile", "none"])

        with pytest.raises(Exception, match="already exists"):
            main_barycenter(
                [self.evfile, self.orbfile, "-p", self.parfile, "-o", outfile,
                 "--clockfile", "none"]
            )
        main_barycenter(
            [self.evfile, self.orbfile, "-p", self.parfile, "-o", outfile,
             "--clockfile", "none", "--overwrite"]
        )

    def test_barycorr_slim(self, tmp_path):
        """--only-columns keeps TIME plus the named columns and drops the rest."""
        outfile = str(tmp_path / "slim.evt")
        main_barycenter(
            [self.evfile, self.orbfile, "-p", self.parfile, "-o", outfile,
             "--clockfile", "none", "--only-columns", "PI,PRIOR"]
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
        main_barycenter(
            [infile, orbfile, "-o", outfile, "--ra", str(ra), "--dec", str(dec)]
        )
        with fits.open(outfile) as hdul:
            assert np.isclose(hdul[1].header["RA_OBJ"], ra)
            assert np.isclose(hdul[1].header["DEC_OBJ"], dec)
            assert "bary" in hdul[1].header.comments["RA_OBJ"].lower()
