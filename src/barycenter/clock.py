"""Spacecraft clock corrections.

A spacecraft's own clock drifts. Missions publish a table of the offset against a
reference, and the correction has to be applied before -- or strictly speaking,
together with -- the barycentric one.

Two missions are covered:

* **NuSTAR**, whose ``NU_FINE_CLOCK`` CALDB extension is a table of the offset and its
  derivative every 1000 s, reaching 25 ms. Files from before 2019 carry a coarser
  ``CLOCK_CORRECT`` extension instead, and are refused.
* **RXTE**, whose coefficients live in an ASCII file, ``tdc.dat``, bundled with this
  package. HEASOFT's ``barycorr`` ignores its own ``clockfile`` parameter for RXTE and
  always reads that file, so reproducing it means reading it too. The correction is tens
  of microseconds -- small, but several hundred times our accuracy target.

:func:`clock_correction_fun` is the one entry point: give it the mission and whatever the
user asked for, and it returns the correction function and the file it came from, or
``None`` for a mission that needs no correction.
"""

import glob
import logging as logger
import os
import warnings

import numpy as np
from astropy.io import fits
from astropy.table import Table
from numba import vectorize

from .utils import get_remote_directory_listing

__all__ = [
    "CLOCK_CALDB_URLS",
    "FINE_CLOCK_EXTENSION",
    "RXTE_DETECTOR_DELAY",
    "clock_cache_dir",
    "clock_correction_fun",
    "cubic_interpolation",
    "get_latest_clock_file",
    "interpolate_clock_function",
    "nustar_clock_correction_fun",
    "read_tdc_file",
    "rxte_clock_correction_fun",
    "rxte_tdc_file",
]


#: Where the CALDB keeps each mission's clock files.
CLOCK_CALDB_URLS = {
    "nustar": "https://heasarc.gsfc.nasa.gov/FTP/caldb/data/nustar/fpm/bcf/clock/",
}

#: The extension a modern NuSTAR clock file keeps its fine correction in.
FINE_CLOCK_EXTENSION = "NU_FINE_CLOCK"


def clock_cache_dir():
    """Where downloaded clock files are kept between runs.

    Under the user's cache directory, not the working directory: NuSTAR clock files are
    about 12 MB and there is no reason to fetch one per run, nor to drop it wherever the
    command happened to be invoked.
    """
    from astropy.config.paths import get_cache_dir

    path = os.path.join(get_cache_dir("barycenter"), "clock")
    os.makedirs(path, exist_ok=True)
    return path


def get_latest_clock_file(mission):
    """The newest clock correction file for a mission, downloading it if needed.

    The CALDB directory index is scraped for the highest-versioned file, which is then
    cached under :func:`clock_cache_dir`. If the network is unavailable, the newest
    already-cached file is used instead.

    Parameters
    ----------
    mission : str

    Returns
    -------
    clockfile : str
        Path to a local clock file.

    Raises
    ------
    FileNotFoundError
        If the index cannot be read *and* nothing is cached locally.
    """
    from urllib.request import urlretrieve

    mission = mission.lower()
    if mission not in CLOCK_CALDB_URLS:
        raise ValueError(f"Mission {mission} not supported for automatic clock file retrieval")

    cache = clock_cache_dir()
    pattern = os.path.join(cache, "nuCclock*.fits*")

    try:
        listing = get_remote_directory_listing(CLOCK_CALDB_URLS[mission])
        remote = sorted(f for f in listing if "nuCclock" in f)[-1]
    except Exception as exc:
        # The fallback that the original code intended but never reached, because the
        # name it returned was only ever assigned on the success path.
        warnings.warn(f"Could not read the {mission} clock file index: {exc}")
        local = sorted(glob.glob(pattern))
        if not local:
            raise FileNotFoundError(
                f"Could not reach the CALDB and no clock file is cached in {cache}. "
                "Pass one with --clockfile, or --clockfile none to skip the correction."
            ) from exc
        logger.info(f"Falling back to the newest cached clock file {local[-1]}")
        return local[-1]

    fname = os.path.join(cache, remote.split("/")[-1])
    if os.path.exists(fname):
        logger.info(f"Using cached clock file {fname}")
    else:
        logger.info(f"Retrieving {remote} into {cache}")
        urlretrieve(remote, fname)
    return fname


@vectorize("float64(float64, float64, float64, float64, float64, float64, float64)")
def _cubic_interpolation(x, xtab0, xtab1, ytab0, ytab1, yptab0, yptab1):
    """Cubic interpolation of tabular data.

    Translated from the cubeterp function in seekinterp.c,
    distributed with HEASOFT.

    Given a tabulated abcissa at two points xtab[] and a tabulated
    ordinate ytab[] (+derivative yptab[]) at the same abcissae, estimate
    the ordinate and derivative at requested point "x"

    Works for numbers or arrays for x. If x is an array,
    xtab, ytab and yptab are arrays of shape (2, x.size).
    """
    dx = x - xtab0
    # Distance between adjoining tabulated abcissae and ordinates
    xs = xtab1 - xtab0
    ys = ytab1 - ytab0

    # Rescale or pull out quantities of interest
    dx = dx / xs  # Rescale DX
    y0 = ytab0  # No rescaling of Y - start of interval
    yp0 = yptab0 * xs  # Rescale tabulated derivatives - start of interval
    yp1 = yptab1 * xs  # Rescale tabulated derivatives - end of interval

    # Compute polynomial coefficients
    a = y0
    b = yp0
    c = 3 * ys - 2 * yp0 - yp1
    d = yp0 + yp1 - 2 * ys

    # Perform cubic interpolation
    yint = a + dx * (b + dx * (c + dx * d))
    return yint


def cubic_interpolation(x, xtab, ytab, yptab):
    """Cubic interpolation of tabular data.

    Translated from the cubeterp function in seekinterp.c,
    distributed with HEASOFT.

    Given a tabulated abcissa at two points xtab[] and a tabulated
    ordinate ytab[] (+derivative yptab[]) at the same abcissae, estimate
    the ordinate and derivative at requested point "x"

    Works for numbers or arrays for x. If x is an array,
    xtab, ytab and yptab are arrays of shape (2, x.size).
    """
    return _cubic_interpolation(x, xtab[0], xtab[1], ytab[0], ytab[1], yptab[0], yptab[1])


def interpolate_clock_function(clock_table, mets):
    """The clock offset correction at arbitrary mission elapsed times.

    The table gives an offset and its time derivative on a coarse grid (1000 s for
    NuSTAR), so the interpolation is the cubic Hermite one HEASOFT uses -- a translation
    of ``cubeterp`` from ``seekinterp.c``.

    Parameters
    ----------
    clock_table : astropy.table.Table
        Columns ``TIME``, ``CLOCK_OFF_CORR``, ``CLOCK_FREQ_CORR``.
    mets : array-like
        Times to evaluate at, in seconds.

    Returns
    -------
    clock_off_corr : ndarray
        Seconds to add, one per requested time.

    Notes
    -----
    Times outside the tabulated range are extrapolated from the nearest interval, with a
    warning. The previous version returned a validity mask alongside a *shortened* array
    of corrections; its only caller discarded the mask and then built a spline from a
    full-length abscissa and a short ordinate, which raised a length mismatch for any
    event outside the clock file's span.
    """
    tab_times = np.asarray(clock_table["TIME"], dtype=np.float64)
    mets = np.asarray(mets, dtype=np.float64)

    outside = (mets < tab_times.min()) | (mets > tab_times.max())
    if np.any(outside):
        worst = max(tab_times.min() - mets.min(), mets.max() - tab_times.max())
        logger.warning(
            f"{np.count_nonzero(outside)} times fall outside the clock file, by up to "
            f"{worst:.1f} s; extrapolating from the nearest interval."
        )

    # Clamped so that a time outside the table uses the edge interval's cubic rather
    # than indexing past the end.
    tab_idxs = np.clip(np.searchsorted(tab_times, mets, side="right") - 1, 0, len(tab_times) - 2)

    clock_off_corr = np.asarray(clock_table["CLOCK_OFF_CORR"], dtype=np.float64)
    clock_freq_corr = np.asarray(clock_table["CLOCK_FREQ_CORR"], dtype=np.float64)

    xtab = [tab_times[tab_idxs], tab_times[tab_idxs + 1]]
    ytab = [clock_off_corr[tab_idxs], clock_off_corr[tab_idxs + 1]]
    yptab = [clock_freq_corr[tab_idxs], clock_freq_corr[tab_idxs + 1]]

    return cubic_interpolation(mets, xtab, ytab, yptab)


def nustar_clock_correction_fun(clockfile):
    """A function giving the NuSTAR clock correction at arbitrary times.

    Parameters
    ----------
    clockfile : str
        A NuSTAR clock file containing the ``NU_FINE_CLOCK`` extension.

    Returns
    -------
    callable
        ``fun(met)`` gives the correction in seconds, with the shape it was given.

    Raises
    ------
    ValueError
        If the file has no fine clock extension. Older files carry a ``CLOCK_CORRECT``
        extension instead, whose per-interval polynomial is only good to the millisecond
        -- four orders of magnitude worse than the fine correction and than our target --
        so it is refused rather than silently applied.

    Notes
    -----
    The correction is evaluated directly at the times asked for. It used to be Hermite
    interpolated onto a 1 s grid and then Akima interpolated from that grid onto the
    events; one interpolation is both faster and more accurate than two.
    """
    with fits.open(clockfile) as hdul:
        names = [hdu.name for hdu in hdul[1:]]
        if FINE_CLOCK_EXTENSION not in names:
            raise ValueError(
                f"{clockfile} has no {FINE_CLOCK_EXTENSION} extension (found {names}). "
                "Clock files from before 2019 carry a CLOCK_CORRECT extension, which is "
                "only accurate to the millisecond and is not supported; fetch a current "
                "one from the CALDB, or pass --clockfile none."
            )
        clocktable = Table(hdul[FINE_CLOCK_EXTENSION].data)

    logger.info(f"Read {len(clocktable)} rows from {FINE_CLOCK_EXTENSION} of {clockfile}")

    def correction(times):
        asked = np.asarray(times, dtype=np.float64)
        values = interpolate_clock_function(clocktable, np.atleast_1d(asked))
        return values.reshape(asked.shape) if asked.ndim else values[0]

    return correction


def rxte_tdc_file():
    """Where to find the RXTE fine clock coefficients.

    A copy of HEASOFT's ``refdata/tdc.dat`` ships with this package, because RXTE
    finished observing in January 2012 and the file has not changed since. ``TIMING_DIR``
    and ``LHEA_DATA`` are still honoured first, so a HEASOFT installation's own copy wins
    if there is one.
    """
    for variable in ("TIMING_DIR", "LHEA_DATA"):
        directory = os.environ.get(variable)
        if directory and os.path.exists(os.path.join(directory, "tdc.dat")):
            return os.path.join(directory, "tdc.dat")
    return os.path.join(os.path.dirname(__file__), "data", "tdc.dat")


def read_tdc_file(tdcfile=None):
    """Read the RXTE fine clock correction coefficients.

    The file is a stream of free-format rows of four numbers, in two kinds. A row whose
    fourth number is negative starts a new block: its first number is the day the block's
    times are measured from and its second is the value of ``TIMEZERO`` in that block. Any
    other row gives the three coefficients of a quadratic in days-since-that-day, valid
    until the fourth number. The trailing comment block is what ends the file, since the
    original C reader stops as soon as a row fails to parse.

    Parameters
    ----------
    tdcfile : str, optional
        Defaults to :func:`rxte_tdc_file`.

    Returns
    -------
    astropy.table.Table
        One row per coefficient set, in file order, with columns ``DAY_END`` (the absolute
        day the set is valid until), ``SUBDAY`` (the day its polynomial is measured from),
        ``TIMEZERO``, ``C0``, ``C1``, ``C2``. ``C0``..``C2`` give microseconds.

    Notes
    -----
    Translated from ``xCC.c`` by A. Rots,
    https://heasarc.gsfc.nasa.gov/docs/xte/abc/xCC.c, which is the reader HEASOFT's
    ``axBary`` uses.
    """
    tdcfile = tdcfile or rxte_tdc_file()
    subday = timezero = None
    rows = []
    with open(tdcfile) as fobj:
        for line in fobj:
            try:
                m0, m1, m2, end = (float(value) for value in line.split())
            except ValueError:
                # A comment, a blank line or anything else unparseable ends the file,
                # exactly as the fscanf loop in xCC.c does.
                break
            if end < 0:
                subday, timezero = m0, m1
            elif subday is None:
                raise ValueError(f"{tdcfile} starts with coefficients before any block header")
            else:
                rows.append((subday + end, subday, timezero, m0, m1, m2))

    if not rows:
        raise ValueError(f"No clock correction coefficients found in {tdcfile}")

    table = Table(
        rows=rows, names=("DAY_END", "SUBDAY", "TIMEZERO", "C0", "C1", "C2"), dtype=[float] * 6
    )
    table.meta["name"] = tdcfile
    return table


#: Detector-dependent delay subtracted from the RXTE fine clock correction, in seconds.
#: ``xCC.c`` quotes the correction for HEXTE and the correction minus 16 microseconds for
#: the PCA; ``hdaxbary`` tests ``INSTRUME`` against ``HEXTE`` and treats everything else
#: as the PCA, so that is the default here too.
RXTE_DETECTOR_DELAY = {"HEXTE": 0.0}
RXTE_DEFAULT_DETECTOR_DELAY = 16e-6


def rxte_clock_correction_fun(tdcfile=None, instrument=None):
    """A function giving the RXTE fine clock correction at arbitrary times.

    Parameters
    ----------
    tdcfile : str, optional
        Defaults to :func:`rxte_tdc_file`.
    instrument : str, optional
        The ``INSTRUME`` keyword. Anything other than ``HEXTE`` gets the PCA's extra
        16 microsecond delay subtracted, which is what ``hdaxbary`` does.

    Returns
    -------
    callable
        ``fun(met)`` gives the correction in seconds, with the shape it was given.

    Notes
    -----
    ``TIMEZERO`` is *not* included: the event file's header already carries it, and
    :func:`barycenter.core.correct_times` already folds that in. Only the fine correction
    is left to add, as the comment at the top of ``xCC.c`` says.

    The correction is evaluated at each time asked for. HEASOFT computes it once, at the
    middle of the observation, and folds that single number into ``TIMEZERO`` -- which
    over a long observation is a real, if small, difference: the coefficients drift by
    about 25 ns per hour.
    """
    table = read_tdc_file(tdcfile)
    delay = RXTE_DETECTOR_DELAY.get(str(instrument).upper(), RXTE_DEFAULT_DETECTOR_DELAY)
    logger.info(
        f"Read {len(table)} RXTE clock coefficient sets from {table.meta['name']}; "
        f"{instrument} detector delay {delay * 1e6:g} us"
    )

    day_end = np.asarray(table["DAY_END"], dtype=np.float64)
    subday = np.asarray(table["SUBDAY"], dtype=np.float64)
    coefficients = [np.asarray(table[name], dtype=np.float64) for name in ("C0", "C1", "C2")]

    def correction(times):
        asked = np.asarray(times, dtype=np.float64)
        days = np.atleast_1d(asked) / 86400.0
        # xCC.c walks the file and takes the first set whose end day is past the time
        # asked for. The end days are non-decreasing, so that is a searchsorted.
        index = np.searchsorted(day_end, days, side="right")
        outside = index >= len(day_end)
        if np.any(outside):
            logger.warning(
                f"{np.count_nonzero(outside)} times are past the end of "
                f"{table.meta['name']}, which ends on mission day {day_end[-1]:.2f}; "
                "no fine clock correction is available there and none is applied."
            )
            index = np.clip(index, 0, len(day_end) - 1)
        dt = days - subday[index]
        values = coefficients[0][index] + dt * (
            coefficients[1][index] + dt * coefficients[2][index]
        )
        values = values * 1e-6 - delay
        values[outside] = 0.0
        return values.reshape(asked.shape) if asked.ndim else values[0]

    return correction


def clock_correction_fun(mission, clockfile=None, instrument=None):
    """The clock correction for a mission, and the file it was read from.

    This is the only place that knows which missions have a clock correction and how each
    one is obtained, so the rest of the package stays mission-agnostic.

    Parameters
    ----------
    mission : str
        The ``TELESCOP`` keyword, in any case.
    clockfile : str, optional
        A clock file the user named. For NuSTAR the newest CALDB file is fetched when this
        is omitted; for RXTE it is ignored, because HEASOFT ignores it too.
    instrument : str, optional
        The ``INSTRUME`` keyword, needed for RXTE's detector delay.

    Returns
    -------
    clock_fun : callable or None
        ``None`` for a mission with no clock correction of its own.
    clockfile : str or None
        The file actually used, for the history record.

    Raises
    ------
    FileNotFoundError
        If a named clock file does not exist.
    """
    mission = str(mission).lower()

    if clockfile is not None and not os.path.exists(clockfile):
        raise FileNotFoundError(f"Clock file {clockfile} not found")

    if "nustar" in mission:
        if clockfile is None:
            clockfile = get_latest_clock_file("nustar")
            logger.info(f"Using latest NuSTAR clock file: {clockfile}")
        return nustar_clock_correction_fun(clockfile), clockfile

    if "xte" in mission:
        if clockfile is not None:
            warnings.warn(
                f"RXTE clock corrections come from tdc.dat, not from {clockfile}; "
                "HEASOFT barycorr ignores its clockfile parameter for RXTE too."
            )
        tdcfile = rxte_tdc_file()
        return rxte_clock_correction_fun(tdcfile, instrument=instrument), tdcfile

    if clockfile is not None:
        warnings.warn(
            f"No clock correction is implemented for mission {mission}; ignoring {clockfile}."
        )
    return None, None
