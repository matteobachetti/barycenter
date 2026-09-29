import numpy as np
import pytest

from barycenter.utils import fits_open_including_remote

fname = (
    "s3://nasa-heasarc/swift/data/obs/2015_12/00037258040/xrt/event/sw00037258040xwtw2st_cl.evt.gz"
)


@pytest.mark.remote_data
def test_simple_loading():
    """An event file can be opened straight from HEASARC's S3 bucket."""
    with fits_open_including_remote(fname) as hdul:
        assert np.isclose(hdul[1].header["MJDREFI"], 51910)
