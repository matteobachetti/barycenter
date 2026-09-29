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
from astropy.table import Table
from numba import vectorize
from scipy.interpolate import Akima1DInterpolator

from .utils import get_remote_directory_listing

__all__ = [
    "cubic_interpolation",
    "get_latest_clock_file",
    "interpolate_clock_function",
    "nustar_clock_correction_fun",
]


def get_latest_clock_file(mission):
    """Get the latest NuSTAR clock correction file from HEASARC.

    Returns
    -------
    clockfile : str
        Path to the latest clock correction file.
    """
    from urllib.request import urlretrieve

    if mission.lower() not in ["nustar"]:
        raise ValueError(f"Mission {mission} not supported for automatic clock file retrieval")

    try:
        listing = get_remote_directory_listing(
            "https://heasarc.gsfc.nasa.gov/FTP/caldb/data/nustar/fpm/bcf/clock/"
        )

        clckfile = sorted([f for f in listing if "nuCclock" in f])[-1]

        fname = clckfile.split("/")[-1]
        if not os.path.exists(fname):
            logger.info(f"Retrieving latest clock file {clckfile}")
            urlretrieve(clckfile, fname)
        else:
            logger.info(f"Using existing local clock file {fname}")
    except Exception as e:
        warnings.warn(f"Could not retrieve latest clock file: {e}")

        clckfile = sorted(glob.glob("nuCclock*.fits"))
        if len(clckfile) == 0:
            raise FileNotFoundError("Error retrieving clock file, and no clock file found locally")
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


def interpolate_clock_function(new_clock_table, mets):
    """Interpolate clock correction table to given MET times.

    Parameters
    ----------
    new_clock_table : astropy.table.Table
        Table with columns TIME, CLOCK_OFF_CORR, CLOCK_FREQ_CORR.
    mets : array-like
        Array of MET times to interpolate to.

    Returns
    -------
    clock_off_corr : array-like
        Interpolated clock offset corrections at the given MET times.
    good_mets : array-like
        Boolean array indicating which MET times are within the tabulated range.
    """
    tab_times = new_clock_table["TIME"]
    good_mets = (mets > tab_times.min()) & (mets < tab_times.max())
    mets = mets[good_mets]
    tab_idxs = np.searchsorted(tab_times, mets, side="right") - 1

    clock_off_corr = new_clock_table["CLOCK_OFF_CORR"]
    clock_freq_corr = new_clock_table["CLOCK_FREQ_CORR"]

    x = np.array(mets)
    xtab = [tab_times[tab_idxs], tab_times[tab_idxs + 1]]
    ytab = [clock_off_corr[tab_idxs], clock_off_corr[tab_idxs + 1]]
    yptab = [clock_freq_corr[tab_idxs], clock_freq_corr[tab_idxs + 1]]

    return cubic_interpolation(x, xtab, ytab, yptab), good_mets


def nustar_clock_correction_fun(clockfile, t_start, t_stop, t_res=1.0):
    """Apply NuSTAR clock correction to times.

    Parameters
    ----------
    times : array-like
        Array of times to correct.
    clock_table : astropy.table.Table
        Table with columns TIME, CLOCK_OFF_CORR, CLOCK_FREQ_CORR.

    Returns
    -------
    corrected_times : array-like
        Array of corrected times.
    """
    unique_times = np.arange(t_start - t_res, t_stop + t_res, t_res)

    hduname = "NU_FINE_CLOCK"
    logger.info(f"Read extension {hduname}")
    clocktable = Table.read(clockfile, hdu=hduname)
    clock_corr, _ = interpolate_clock_function(clocktable, unique_times)
    clock_fun = Akima1DInterpolator(unique_times, clock_corr, extrapolate=True)

    return clock_fun
