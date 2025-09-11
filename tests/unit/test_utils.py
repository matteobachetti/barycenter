import logging as logger

import numpy as np

from barycenter.utils import fits_open_including_remote

fname = (
    "s3://nasa-heasarc/swift/data/obs/2015_12/00037258040/xrt/event/"
    "sw00037258040xwtw2st_cl.evt.gz"
)


def test_simple_loading():
    hdul = fits_open_including_remote(fname)
    assert np.isclose(hdul[1].header["MJDREFI"], 51910)
    assert hdul is not None
    hdul.close()
