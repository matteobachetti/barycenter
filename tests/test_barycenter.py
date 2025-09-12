from barycenter import main_barycenter
import os
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits


curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")


class TestExecution(object):
    @classmethod
    def setup_class(self):
        self.t0, self.t1 = 57263.9, 57264.6

        self.orbfile = os.path.join(datadir, "dummy_orb.fits.gz")
        self.parfile = os.path.join(datadir, "dummy_par.par")
        self.evfile = os.path.join(datadir, "dummy_evt.evt")

    def test_barycorr_overwrite(self):
        outfile = main_barycenter([self.evfile, self.orbfile, "-p", self.parfile])
        assert os.path.exists(outfile)
        with pytest.raises(Exception, match="File bary_dummy_evt.evt already exists."):
            main_barycenter([self.evfile, self.orbfile, "-p", self.parfile, "-o", outfile])
        main_barycenter(
            [self.evfile, self.orbfile, "-p", self.parfile, "-o", outfile, "--overwrite"]
        )

        os.unlink(outfile)

    @pytest.mark.remote_data
    @pytest.mark.parametrize("prefix", ["s3://nasa-heasarc/", "https://heasarc.gsfc.nasa.gov/FTP/"])
    def test_barycorr_remote(self, prefix):
        coord = SkyCoord.from_name("M82 X-2")
        ra, dec = coord.ra.deg, coord.dec.deg
        infile = f"{prefix}nustar/data/obs/07/3/30702012003/event_cl/nu30702012003A06_cl.evt.gz"
        orbfile = f"{prefix}nustar/data/obs/07/3/30702012003/event_cl/nu30702012003A.attorb.gz"
        print("Input file:", infile)
        print("Orbit file:", orbfile)

        outfile = main_barycenter([infile, orbfile, "--ra", str(ra), "--dec", str(dec)])
        with fits.open(outfile) as hdul:
            assert np.isclose(hdul[1].header["RA_OBJ"], ra)
            assert np.isclose(hdul[1].header["DEC_OBJ"], dec)
            assert "bary" in hdul[1].header.comments["RA_OBJ"].lower()

        os.unlink(outfile)

    def test_barycorr_slim(self):
        outfile = main_barycenter(
            [self.evfile, self.orbfile, "-p", self.parfile, "--only-columns", "PI,PRIOR"]
        )
        assert os.path.exists(outfile)

        with fits.open(outfile) as hdul:
            assert "PRIOR" in hdul[1].data.names
            assert "TIME" in hdul[1].data.names
            assert "NUMRISE" not in hdul[1].data.names

        os.unlink(outfile)
