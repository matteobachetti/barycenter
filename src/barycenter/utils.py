import re
import logging as logger

from html.parser import HTMLParser

import numpy as np
from astropy.io import fits

__all__ = [
    "as_longdouble",
    "column_named",
    "leap_seconds_since_mjdref",
    "fits_open_including_remote",
    "fits_open_remote",
    "high_precision_keyword_read",
    "high_precision_mjdref",
    "splitext_improved",
]


def column_named(data, name):
    """The column actually called ``name``, whatever case the file wrote it in.

    FITS column names are case-insensitive by standard, and missions use that freedom:
    Chandra writes its event times in a column called ``time`` and its orbit times in
    ``Time``, where everyone else writes ``TIME``. A case-sensitive ``"TIME" in
    data.names`` therefore finds nothing on a Chandra file -- and since the ``GTI``
    extension beside it *does* spell ``START``/``STOP`` in capitals, the result is not a
    failure but a file whose good-time intervals are barycentred and whose events are
    not. Silently producing half-corrected times is the worst outcome available here, so
    every lookup of a column by name goes through this.

    Parameters
    ----------
    data : astropy.io.fits.FITS_rec or None
        Table data, or ``None`` for an image or empty extension.
    name : str
        The name to look for, in any case.

    Returns
    -------
    str or None
        The column's name as the file spells it, or ``None`` if there is no such column.
    """
    if data is None:
        return None
    wanted = str(name).upper()
    for actual in data.names:
        if actual.upper() == wanted:
            return actual
    return None


def fits_open_remote(filename, **kwargs):
    """Open a remote FITS file.

    This function attempts to open a FITS file using `astropy.io.fits.open`. If a
    `PermissionError` is raised and the filename appears to be a remote URL,
    it retries opening the file with fsspec.

    Requires the `botocore` package to be installed.

    Parameters
    ----------
    filename : str
        The path or URL to the FITS file to open. Can be a local file path or a remote URL.
    **kwargs
        Additional keyword arguments passed to `astropy.io.fits.open`.

    Returns
    -------
    hdulist : astropy.io.fits.HDUList
        The opened FITS file as an HDUList object.

    Raises
    ------
    PermissionError
        If the file cannot be opened and anonymous access is not possible or fails.

    """
    import botocore
    import botocore.exceptions

    try:
        # This will work for local files and remote files with proper permissions
        return fits.open(filename, **kwargs)
    except (PermissionError, botocore.exceptions.NoCredentialsError):
        # Anonymous access is only a thing for a URL. On a local path there is nothing
        # to retry, and the permission error is the answer -- so re-raise it rather
        # than fall through to an unassigned name, which used to turn "you cannot read
        # this file" into an UnboundLocalError pointing at this function.
        if "://" not in filename:
            raise
        logger.info(f"Permission denied for {filename}, trying with fsspec.")
        return fits.open(filename, use_fsspec=True, fsspec_kwargs={"anon": True}, **kwargs)


def fits_open_including_remote(filename, **kwargs):
    """Open a FITS file, including remote files with anonymous access if needed.

    If the filename appears to be a remote URL, it calls `fits_open_remote` to handle
    potential permission issues. Otherwise, it opens the file directly with
    `astropy.io.fits.open`.

    Parameters
    ----------
    filename : str
        The path or URL to the FITS file to open. Can be a local file path or a remote URL.
    **kwargs
        Additional keyword arguments passed to `astropy.io.fits.open`.

    Returns
    -------
    hdulist : astropy.io.fits.HDUList
        The opened FITS file as an HDUList object.

    """

    if "://" in filename:
        return fits_open_remote(filename, **kwargs)
    return fits.open(filename, **kwargs)


def slim_down_hdu_list(hdul, additional_cols=None, ext=1):
    """Reduce a FITS HDUList size by only keeping few columns in the specified extension.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        Input HDUList.

    Other Parameters
    ----------------

    additional_cols : list of str, optional
        Additional column names to keep in the output file, in addition to the "TIME" column
    ext: int or str or List
        Extension(s) to slim down. Default is 1.
    """

    data = hdul[1].data
    time_col = column_named(data, "TIME")
    if time_col is None:
        raise ValueError("Extension 1 does not contain a TIME column.")
    cols = [data.columns[time_col]]
    for col in additional_cols or []:
        actual = column_named(data, col)
        if actual is not None:
            cols.append(data.columns[actual])
    if isinstance(ext, (int, str)):
        ext = [ext]

    for e in ext:
        hdu = hdul[e]
        if hdu.data is None:
            continue
        if column_named(hdu.data, "TIME") is None:
            raise ValueError(f"Extension {e} does not contain a TIME column.")
        logger.info(f"Slimming down extension {e} to columns {[c.name for c in cols]}")
        hdul[e].data = fits.BinTableHDU.from_columns(cols).data

    return hdul


def slim_down_file(file, outfile, additional_cols=None, ext=1):
    """Reduce a FITS file size only keeping few columns in the specified extension.

    Parameters
    ----------
    file : str
        Input FITS file path
    outfile : str
        Output FITS file path.

    Other Parameters
    ----------------
    additional_cols : list of str, optional
        Additional column names to keep in the output file, in addition to the "TIME" column
    ext: int or str or List
        Extension(s) to slim down. Default is 1.
    """
    hdul = slim_down_hdu_list(
        fits_open_including_remote(file), additional_cols=additional_cols, ext=ext
    )

    hdul.writeto(outfile)


class _LinkTextParser(HTMLParser):
    """Collect the text of every ``<a>`` element, which is what a directory index lists."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.texts = []
        self._depth = 0
        self._parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._depth += 1

    def handle_endtag(self, tag):
        if tag == "a" and self._depth > 0:
            self._depth -= 1
            if self._depth == 0:
                self.texts.append("".join(self._parts))
                self._parts = []

    def handle_data(self, data):
        if self._depth > 0:
            self._parts.append(data)


def link_texts_in_html(html):
    """The text of every link on a page, in the order the page lists them.

    This is all the CALDB scrape ever wanted from BeautifulSoup, and the standard
    library's parser does it in twenty lines. Keeping bs4 would mean either a dependency
    on the base installation -- the NuSTAR and Swift clock files are fetched this way by
    default, so a plain ``pip install`` has to be able to do it -- or an install that
    fails on its own README example. Anchors with no text, which index pages use to wrap
    icons, are left out: there is no filename in them to fetch.
    """
    parser = _LinkTextParser()
    parser.feed(html.decode("utf-8", errors="replace") if isinstance(html, bytes) else html)
    parser.close()
    return [text for text in parser.texts if text]


def get_remote_directory_listing(url: str):
    """Give the list of files in the remote directory."""
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError

    url = url.replace(" ", "%20")
    req = Request(url)
    try:
        a = urlopen(req).read()
    except HTTPError:
        return None

    urls = []
    for file_name in link_texts_in_html(a):
        url_new = url + file_name
        url_new = url_new.replace(" ", "%20")
        if file_name[-1] == "/" and file_name[0] != ".":
            urls.append(url_new)
            url_new = get_remote_directory_listing(url_new)
            if url_new is None:
                continue
            urls.extend(url_new)
        else:
            urls.append(url_new)

    return urls


#: A real number written the way Fortran writes one, with ``D`` (or ``Q``) marking the
#: exponent instead of ``E``. Matched in full rather than substituted blindly, so that a
#: keyword which is not a number stays unparseable instead of being coerced into one.
FORTRAN_EXPONENT = re.compile(r"\A([+-]?(?:\d+\.?\d*|\.\d+))[DdQq]([+-]?\d+)\Z")


def as_longdouble(value):
    """A header value as a ``longdouble``, allowing for Fortran's exponent marker.

    FITS says a floating-point keyword is written with ``E`` or no exponent at all, and
    astropy hands such a card back as a number. A card whose value it cannot recognise as
    a number comes back as a *string* instead -- which is what happens to a keyword an
    older Fortran tool wrote as ``7.428703703703703D-4``. Fermi's own tutorial event file
    writes ``MJDREFF`` that way, and so do other products of that era.

    Only the exponent marker is rewritten, and only when the whole value is otherwise a
    number. Anything else is passed through untouched and fails as it did before: silently
    coercing an unrecognised keyword would be far worse than refusing it, because being
    wrong about ``MJDREF`` by a day is a half-hour error in the barycentric correction.
    """
    if isinstance(value, str):
        match = FORTRAN_EXPONENT.match(value.strip())
        if match is not None:
            value = f"{match.group(1)}E{match.group(2)}"
    return np.longdouble(value)


def high_precision_keyword_read(header, keyword):
    """Read a FITS keyword that may be split into integer and fractional halves.

    Missions write their reference epoch either as one keyword, ``MJDREF``, or as a
    pair, ``MJDREFI`` + ``MJDREFF``. The pair exists precisely because a single float64
    cannot hold an MJD to better than a microsecond, so the two parts have to be summed
    in extended precision -- summing them as float64 throws away the reason they were
    split.

    Parameters
    ----------
    header : astropy.io.fits.Header or dict
    keyword : str
        For example ``"MJDREF"`` or ``"TSTART"``.

    Returns
    -------
    numpy.longdouble or None
        ``None`` if neither the single keyword nor the pair is present.
    """
    if keyword in header:
        return as_longdouble(header[keyword])

    stem = keyword[:7] if len(keyword) == 8 else keyword
    if stem + "I" in header and stem + "F" in header:
        return as_longdouble(header[stem + "I"]) + as_longdouble(header[stem + "F"])
    return None


def high_precision_mjdref(header):
    """The mission reference epoch, MJD(TT), in extended precision.

    Raises
    ------
    ValueError
        If the header has neither ``MJDREF`` nor ``MJDREFI``/``MJDREFF``. Guessing a
        reference epoch is never the right thing to do: being wrong by a day is a
        half-hour error in the barycentric correction.
    """
    mjdref = high_precision_keyword_read(header, "MJDREF")
    if mjdref is None:
        raise ValueError("Header has no MJDREF, nor MJDREFI/MJDREFF")
    return mjdref


def leap_seconds_since_mjdref(mjdref, mets):
    """Seconds of leap-second drift between a file's reference epoch and its times.

    Most missions count their mission elapsed time in TT seconds, and ``MJDREF`` is all
    that is needed to place a time stamp: ``MJD(TT) = MJDREF + MET / 86400``. Swift does
    not. Its MET counts **UTC** seconds, which is to say it is a clock that is held back
    by one second every time a leap second is inserted, so the number of TT seconds that
    have actually elapsed since the epoch is larger than the MET by however many leap
    seconds fell in between. HEASOFT ``barycorr`` adds that difference before doing
    anything else, and it does so whether or not a clock file was given: leaving it out
    puts a December 2015 Swift observation 4 s away from ``barycorr``, which is forty
    million times the accuracy target.

    The offset is deliberately measured from the *file's own* epoch rather than from any
    fixed date, because that is what ``MJDREF`` means. Swift's ``MJDREFF`` of
    0.00074287037 is 64.184 s, which is TT - UTC on 2001-01-01; the four leap seconds
    since then are what this function returns for a 2015 observation. Applying it to a
    mission whose MET is already in TT seconds would be an error of the same size, so it
    is opt-in per mission, through :attr:`barycenter.missions.Mission.met_is_utc`.

    Parameters
    ----------
    mjdref : float
        The file's reference epoch as an MJD in TT, from :func:`high_precision_mjdref`.
    mets : float or array-like
        Mission elapsed times, in seconds.

    Returns
    -------
    float or ndarray
        Seconds to add, with the shape of ``mets``.

    Notes
    -----
    The leap-second epochs come from ERFA's own table, converted to METs once, so the
    answer is exact at the boundary rather than rounded to the nearest day. A file may
    straddle a leap second -- rarely, but the alternative to handling it is a silent
    one-second step in the middle of an observation.
    """
    import erfa
    from astropy.time import Time

    table = erfa.leap_seconds.get()
    # Before 1972 TAI-UTC was a drifting rate rather than a whole number of seconds. No
    # mission this package handles observed then, and pretending those entries are steps
    # would be wrong, so they are left out.
    table = table[table["year"] >= 1972]

    boundaries = Time(
        [f"{row['year']:04d}-{row['month']:02d}-01T00:00:00" for row in table], scale="utc"
    )
    tai_utc = np.asarray(table["tai_utc"], dtype=np.float64)

    # TAI - UTC at the reference epoch, read out of the table rather than computed as
    # ``(epoch.tai.mjd - epoch.utc.mjd) * 86400``. That difference of two MJDs of order
    # 5e4 rounds to 31.999999937 s for Swift, and the resulting 63 ns bias is most of the
    # 100 ns budget -- it was measured, as a 60 ns disagreement with ``barycorr`` that
    # went away when this was made exact. Since 1972 the quantity is a whole number of
    # seconds by definition, so it should be read, never subtracted.
    epoch_utc_mjd = Time(np.float64(mjdref), format="mjd", scale="tt").utc.mjd
    at = np.searchsorted(boundaries.mjd, epoch_utc_mjd, side="right") - 1
    if at < 0:
        raise ValueError(
            f"MJDREF {mjdref} predates 1972, when TAI-UTC stopped being a drifting rate. "
            "No mission this package handles observed then."
        )
    at_epoch = tai_utc[at]

    # Where each step falls, on the file's own clock -- and that clock counts UTC seconds,
    # which is the whole reason this function exists. ``(boundary.tt - mjdref) * 86400`` is
    # a count of *TT* seconds, so it sits ahead of the MET of the same instant by exactly
    # the leap seconds owed there, and subtracting ``offsets`` puts the boundary back into
    # MET. Leaving it out puts every step 4 s late for Swift: harmless for data taken away
    # from a leap second, but wrong by a second for the four seconds after one.
    elapsed_tt = (boundaries.tt.mjd - np.float64(mjdref)) * 86400.0
    offsets = tai_utc - at_epoch
    boundary_mets = elapsed_tt - offsets

    # Only the steps that fall after this file's epoch can contribute to it.
    after = boundary_mets > 0.0
    boundary_mets = boundary_mets[after]
    offsets = offsets[after]

    asked = np.asarray(mets, dtype=np.float64)
    index = np.searchsorted(boundary_mets, np.atleast_1d(asked), side="right")
    # Index 0 means "before the first step after the epoch", which owes nothing, so the
    # zero is prepended and ``index`` reads the table directly. Selecting it afterwards
    # with ``np.where`` would not do: that evaluates both of its branches, so an empty
    # table -- an MJDREF later than every leap second, which is every mission launched
    # since 2017 -- raised IndexError on a time that correctly owes zero.
    values = np.concatenate(([0.0], offsets))[index]
    return values.reshape(asked.shape) if asked.ndim else values[0]


def splitext_improved(path):
    """Split off a file extension, keeping a compression suffix attached to it.

    Examples
    --------
    >>> splitext_improved("a.tar.gz")
    ('a', '.tar.gz')
    >>> splitext_improved("a.tar")
    ('a', '.tar')
    >>> splitext_improved("a.f/a.tar")
    ('a.f/a', '.tar')
    >>> splitext_improved("a.a.a.f/a.tar.gz")
    ('a.a.a.f/a', '.tar.gz')
    """
    import os

    ext = ""
    dir, file = os.path.split(path)
    for zip_ext in [".tar", ".tar.gz", ".gz", ".bz2", ".zip", ".xz", ".Z"]:
        if file.endswith(zip_ext):
            file = file[: -len(zip_ext)]
            ext = zip_ext
            break

    froot, new_ext = os.path.splitext(file)

    return os.path.join(dir, froot), new_ext + ext
