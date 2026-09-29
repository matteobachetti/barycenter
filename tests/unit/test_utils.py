import numpy as np
import pytest

from barycenter.utils import column_named, fits_open_including_remote

fname = (
    "s3://nasa-heasarc/swift/data/obs/2015_12/00037258040/xrt/event/sw00037258040xwtw2st_cl.evt.gz"
)


@pytest.mark.remote_data
def test_simple_loading():
    """An event file can be opened straight from HEASARC's S3 bucket."""
    with fits_open_including_remote(fname) as hdul:
        assert np.isclose(hdul[1].header["MJDREFI"], 51910)


class TestColumnNamed:
    """Looking a column up by name, whatever case the file wrote it in.

    FITS column names are case-insensitive by standard and missions use that freedom.
    Chandra writes ``time`` and ``Time`` where everyone else writes ``TIME``, so a
    case-sensitive lookup finds nothing -- and because the GTI extension beside it does
    use capitals, the file would come out with barycentred intervals and untouched
    events.
    """

    @staticmethod
    def table(*names):
        from astropy.io import fits

        return fits.BinTableHDU.from_columns(
            [fits.Column(name=n, format="D", array=np.zeros(3)) for n in names]
        ).data

    @pytest.mark.parametrize("written", ["TIME", "time", "Time", "tImE"])
    def test_any_spelling_is_found(self, written):
        """Each way a mission might spell the time column resolves to that column."""
        assert column_named(self.table(written, "PI"), "TIME") == written

    def test_the_name_asked_for_may_also_be_any_case(self):
        """The lookup is symmetric, so callers need not shout either."""
        assert column_named(self.table("time"), "time") == "time"

    def test_a_missing_column_gives_none(self):
        """Absent means ``None``, not an exception: some extensions simply have no times."""
        assert column_named(self.table("RAWX", "RAWY"), "TIME") is None

    def test_an_extension_with_no_table_gives_none(self):
        """Image and empty extensions have no columns at all, and must not raise."""
        assert column_named(None, "TIME") is None
