import os

import numpy as np
import pytest
from astropy.io import fits

from barycenter.utils import column_named, fits_open_including_remote, fits_open_remote

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


class TestFitsOpenRemote:
    """The fallback to anonymous access must not swallow the error it cannot handle."""

    @pytest.mark.skipif(
        os.name == "nt",
        reason="chmod(0) only sets the read-only flag on Windows; the file stays readable",
    )
    def test_a_local_permission_error_is_reported_as_itself(self, tmp_path):
        """A local unreadable file raises PermissionError, not UnboundLocalError.

        The anonymous-access retry only makes sense for a URL. When the name is a
        local path the ``except`` branch has nothing to try, and it used to fall
        through to ``return hdul`` with ``hdul`` never assigned -- turning a plain
        "you cannot read this file" into an UnboundLocalError from inside our code.

        Skipped on Windows, where there is no way to make a file unreadable this way.
        The branch under test is platform-independent, so POSIX coverage is enough.
        """
        pytest.importorskip("botocore")
        unreadable = tmp_path / "unreadable.fits"
        fits.PrimaryHDU().writeto(unreadable)
        unreadable.chmod(0o000)
        try:
            with pytest.raises(PermissionError):
                fits_open_remote(str(unreadable))
        finally:
            unreadable.chmod(0o600)


#: A CALDB index page, cut down to the shapes that matter: a parent link, a
#: subdirectory, a hidden entry, two clock files, and an anchor with no text.
CALDB_INDEX = b"""<html><head><title>Index of /caldb</title></head><body>
<h1>Index of /caldb</h1>
<table>
<tr><td><a href="/caldb/">Parent Directory</a></td></tr>
<tr><td><a href="bcf/">bcf/</a></td></tr>
<tr><td><a href=".hidden/">.hidden/</a></td></tr>
<tr><td><a href="nuCclock20100101v123.fits">nuCclock20100101v123.fits</a></td></tr>
<tr><td><a href="nuCclock20100101v124.fits">nuCclock20100101v124.fits</a></td></tr>
<tr><td><a href="icon.gif"><img src="icon.gif"></a></td></tr>
</table></body></html>"""


class TestLinkTextsInHtml:
    """Reading a directory index with the standard library rather than BeautifulSoup.

    The CALDB scrape is the default way NuSTAR and Swift clock files are found, so this
    parser has to work in a plain installation -- which is why it is not bs4.
    """

    def test_every_link_text_is_collected_in_order(self):
        """Apache indexes name each entry in the link text, and order picks the newest."""
        from barycenter.utils import link_texts_in_html

        assert link_texts_in_html(CALDB_INDEX) == [
            "Parent Directory",
            "bcf/",
            ".hidden/",
            "nuCclock20100101v123.fits",
            "nuCclock20100101v124.fits",
        ]

    def test_an_anchor_with_no_text_is_skipped(self):
        """Index pages wrap icons in bare anchors; an empty name has nothing to fetch."""
        from barycenter.utils import link_texts_in_html

        assert "" not in link_texts_in_html(CALDB_INDEX)

    def test_entities_and_nested_markup_come_out_as_text(self):
        """The text may be marked up or escaped, and the filename is what is wanted."""
        from barycenter.utils import link_texts_in_html

        html = b'<a href="x">a&amp;b</a><a href="y"><b>bold</b>ed</a>'
        assert link_texts_in_html(html) == ["a&b", "bolded"]

    def test_it_agrees_with_beautifulsoup(self):
        """The replacement must read a real index exactly as the old bs4 code did."""
        bs4 = pytest.importorskip("bs4", reason="the parity check needs the old dependency")
        from barycenter.utils import link_texts_in_html

        soup = bs4.BeautifulSoup(CALDB_INDEX, "html.parser")
        expected = [t for t in (a.get_text() for a in soup.find_all("a")) if t]
        assert link_texts_in_html(CALDB_INDEX) == expected

    def test_it_works_with_bs4_uninstalled(self, monkeypatch):
        """A plain installation has no bs4, and must still find its clock files.

        ``beautifulsoup4`` is not a dependency of the base package, so an import of it
        anywhere on the default NuSTAR or Swift path breaks ``pip install barycenter``.
        """
        import builtins

        real_import = builtins.__import__

        def refuse_bs4(name, *args, **kwargs):
            if name.split(".")[0] == "bs4":
                raise ImportError("No module named 'bs4'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse_bs4)
        monkeypatch.delitem(__import__("sys").modules, "bs4", raising=False)

        from barycenter.utils import link_texts_in_html

        assert "nuCclock20100101v124.fits" in link_texts_in_html(CALDB_INDEX)
