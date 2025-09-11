from barycenter import main_barycenter
import os
import numpy as np
import pytest


curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")


class TestExecution(object):
    @classmethod
    def setup_class(self):
        self.t0, self.t1 = 57263.9, 57264.6

        self.orbfile = os.path.join(datadir, "dummy_orb.fits.gz")
        self.parfile = os.path.join(datadir, "dummy_par.par")
        self.evfile = os.path.join(datadir, "dummy_evt.evt")

    @pytest.mark.remote_data
    def test_barycorr_overwrite(self):
        outfile = main_barycenter([self.evfile, self.orbfile, "-p", self.parfile])
        assert os.path.exists(outfile)
        with pytest.raises(Exception, match="File bary_dummy_evt.evt already exists."):
            main_barycenter([self.evfile, self.orbfile, "-p", self.parfile, "-o", outfile])
        main_barycenter(
            [self.evfile, self.orbfile, "-p", self.parfile, "-o", outfile, "--overwrite"]
        )

        os.unlink(outfile)
