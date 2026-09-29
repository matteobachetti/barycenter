"""Spacecraft clock corrections.

A spacecraft's own clock drifts. Missions publish a table of the offset against a
reference, and the correction has to be applied before -- or strictly speaking,
together with -- the barycentric one. Only NuSTAR's is implemented so far.
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
    "clock_cache_dir",
    "cubic_interpolation",
    "get_latest_clock_file",
    "interpolate_clock_function",
    "nustar_clock_correction_fun",
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
