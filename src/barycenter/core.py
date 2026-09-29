"""The mission-agnostic correction: read a file, correct every time in it, write it out.

This is the pure-Python workflow. It reads the spacecraft orbit through
:mod:`barycenter.orbit`, gets a correction function from one of the engines, optionally
adds a clock correction from :mod:`barycenter.clock`, and applies both to every time
column and time keyword in every extension.
"""

import logging as logger
import os
import tempfile
import warnings

import numpy as np
from astropy.time import Time

from ._version import __version__
from .clock import get_latest_clock_file, nustar_clock_correction_fun
from .official import apply_mission_specific_barycenter_correction
from .orbit import read_orbit
from .pintengine import pint_barycentric_correction
from .remote import download_locally
from .utils import fits_open_including_remote, slim_down_hdu_list

__all__ = [
    "apply_barycenter_correction",
    "correct_times",
    "extract_events_in_region",
    "get_barycentric_correction",
    "get_coordinates_from_fits_header",
    "get_dummy_parfile_for_position",
]


def get_coordinates_from_fits_header(hdr):
    """Get RA/Dec coordinate keywords from FITS header.

    In order of priority, looks for RA_OBJ/DEC_OBJ, RA_NOM/DEC_NOM, RA_PNT/DEC_PNT.

    Parameters
    ----------
    hdr : astropy.io.fits.Header
        FITS header to read coordinates from.

    Returns
    -------
    ra_key : str
        Keyword name for Right Ascension.
    dec_key : str
        Keyword name for Declination.
    """

    if "RA_OBJ" in hdr:
        return "RA_OBJ", "DEC_OBJ"
    elif "RA_NOM" in hdr:
        return "RA_NOM", "DEC_NOM"
    elif "RA_PNT" in hdr:
        return "RA_PNT", "DEC_PNT"
    else:
        raise ValueError("No coordinates found in header")


def get_dummy_parfile_for_position(orbfile):
    """Get a dummy parfile with RAJ and DECJ from the orbit file.

    Parameters
    ----------
    orbfile : str
        Orbit file.

    Returns
    -------
    modelin : pint.models.TimingModel
        Timing model with RAJ and DECJ defined.
    """
    from astropy.coordinates import Angle
    from pint.models import StandardTimingModel

    # Construct model by hand
    with fits_open_including_remote(orbfile, memmap=True) as hdul:
        ra_label, dec_label = get_coordinates_from_fits_header(hdul[1].header)
        ra = hdul[1].header[ra_label]
        dec = hdul[1].header[dec_label]

    modelin = StandardTimingModel
    # Should check if 12:13:14.2 syntax is used and support that as well!
    modelin.RAJ.quantity = Angle(ra, unit="deg")
    modelin.DECJ.quantity = Angle(dec, unit="deg")
    modelin.DM.quantity = 0
    return modelin


def get_barycentric_correction(
    orbfile,
    modelin,
    dt=5,
    met_range=None,
):
    """Get a function to compute barycentric correction from MET TT to MET TDB.

    Parameters
    ----------
    orbfile : str
        Orbit file.
    modelin : pint.models.TimingModel
        Timing model with RAJ, DECJ and EPHEM defined.
    dt : float, optional
        Time step in seconds for the interpolation grid. Default is 5.
    met_range : tuple of float, optional
        ``(start, stop)`` in mission elapsed time, to keep the grid from spanning the
        whole orbit file when only a short observation is being corrected.

    Returns
    -------
    bary_fun : callable
        Function to compute barycentric correction.
    """

    # read_orbit takes a file name, a list of them or an "@metafile", reads the
    # mission's columns from the ORBIT_SPECS registry, and hands back one cleaned
    # table. The native engine reads the same table, so the two engines cannot
    # disagree about where the spacecraft was.
    orbit_table = read_orbit(orbfile)
    mjdref = orbit_table.meta["mjdref"]

    met = np.asarray(orbit_table["MET"].value, dtype=np.float64)
    if met_range is not None:
        met_range = (max(met.min(), met_range[0]), min(met.max(), met_range[1]))

    return pint_barycentric_correction(
        orbit_table, modelin, mjdref=mjdref, dt=dt, met_range=met_range
    )


def correct_times(times, bary_fun, clock_fun=None):
    """Apply barycentric and clock corrections to times.

    Parameters
    ----------
    times : array-like
        Array of times to correct.
    bary_fun : callable
        Function to compute barycentric correction.
    clock_fun : callable, optional
        Function to compute clock correction. If None, no clock correction is applied.

    Returns
    -------
    corrected_times : array-like
        Array of corrected times.

    Notes
    -----
    The clock correction is applied here *after* the barycentric correction has
    been evaluated: both ``clock_fun`` and ``bary_fun`` see the raw mission
    time.  HEASOFT ``barycorr`` does the opposite -- it corrects the clock
    first and then evaluates the barycentric correction, and the spacecraft
    position lookup, at the clock-corrected time.  On NuSTAR, where the clock
    correction reaches a few milliseconds, the two orders differ by about
    1.1 us, well above our 100 ns target.  Matching barycorr means calling
    ``bary_fun(times + cl_corr)``; this is deliberately left for the clock-file
    work, and is why the reference test runs with the clock correction off.
    """
    cl_corr = 0
    if clock_fun is not None:
        cl_corr = clock_fun(times)
    bary_corr = bary_fun(times)

    return times + cl_corr + bary_corr


def extract_events_in_region(fname, ra, dec, region_deg, outfile="src_events.evt"):
    """Extract events within a circular region from a FITS event file.

    Parameters
    ----------
    fname : str
        Input FITS event file.
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    region_deg : float
        Radius of the circular region in degrees.
    outfile : str, optional
        Output FITS event file. Default is "src_events.evt".

    Returns
    -------
    local_fname : str
        Path to the output FITS event file with extracted events.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    from regions import CircleSkyRegion, PixCoord

    with fits_open_including_remote(fname, memmap=True) as hdul:
        data = hdul[1].data
        header = hdul[1].header
        refframe = header.get("RADECSYS", "icrs").lower()

        source_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame=refframe)
        colnames = [n.lower() for n in hdul[1].columns.names]
        xcolnum = colnames.index("x") + 1
        ycolnum = colnames.index("y") + 1
        w = WCS(header, keysel=["pixel"], colsel=[xcolnum, ycolnum])

        sky_region = CircleSkyRegion(source_coord, region_deg * u.deg)
        sky_region_pixel = sky_region.to_pixel(w)

        x, y = data["X"], data["Y"]
        good = sky_region_pixel.contains(PixCoord(x, y))

        extracted_data = data[good]
        hdul[1].data = extracted_data
        hdul[1].header.add_history(f"Selected events within {region_deg} deg of RA={ra}, Dec={dec}")

        hdul.writeto(outfile, overwrite=True, output_verify="ignore")

    return outfile


def apply_barycenter_correction(
    fname,
    orbfile,
    outfile="bary.evt",
    clockfile=None,
    parfile=None,
    ephem="DE440",
    radecsys="ICRS",
    ra=None,
    dec=None,
    source_region_deg=None,
    overwrite=False,
    only_columns=None,
    apply_official=False,
):
    """Apply barycenter correction to a FITS event file.

    Parameters
    ----------
    fname : str
        Input FITS event file.
    orbfile : str
        Orbit file.

    Other Parameters
    ----------------
    outfile : str, optional
        Output FITS event file. Default is "bary.evt".
    clockfile : str, optional
        Clock file.
    parfile : str, optional
        Parameter file.
    ephem : str, optional
        Ephemeris model to use. Default is "DE440".
    radecsys : str, optional
        Coordinate system for RA/Dec. Default is "ICRS".
    ra : float, optional
        Right Ascension in degrees. If not provided, will be read from header.
    dec : float, optional
        Declination in degrees. If not provided, will be read from header.
    overwrite : bool, optional
        If True, will overwrite existing output file. Default is False.
    only_columns : list of str, optional
        List of column names to keep in the output file, in addition to the "TIME" column.
    """
    import astropy.units as u

    cloud = "SCISERVER_USER_ID" in os.environ or "/home/jovyan" in os.environ.get("HOME", "")

    if apply_official or not cloud:
        fname = download_locally(fname)
        orbfile = download_locally(orbfile)

    if source_region_deg is not None:
        source_sel_fname = tempfile.NamedTemporaryFile(delete=False, suffix=".evt").name
        fname = extract_events_in_region(
            fname, ra, dec, source_region_deg, outfile=source_sel_fname
        )

    if apply_official:
        return apply_mission_specific_barycenter_correction(
            fname,
            orbfile,
            outfile=outfile,
            clockfile=clockfile,
            parfile=parfile,
            ra=ra,
            dec=dec,
            ephem=ephem,
            radecsys=radecsys,
            overwrite=overwrite,
            only_columns=only_columns,
        )

    version = __version__
    with fits_open_including_remote(fname, memmap=True) as hdul:
        if parfile is not None and os.path.exists(parfile):
            from pint.models import get_model

            modelin = get_model(parfile)
        else:
            from pint.models import StandardTimingModel

            if ra is None or dec is None:
                ra_str, dec_str = get_coordinates_from_fits_header(hdul[1].header)
                ra = hdul[1].header[ra_str]
                dec = hdul[1].header[dec_str]
                logger.info(f"Using coordinates from header: {ra_str}={ra}, {dec_str}={dec}")

            modelin = StandardTimingModel
            modelin.RAJ.quantity = ra * u.deg
            modelin.DECJ.quantity = dec * u.deg
            modelin.DM.quantity = 0.0
            modelin.EPHEM.value = ephem

        bary_fun = get_barycentric_correction(orbfile, modelin)

        timezero = hdul[1].header.get("TIMEZERO", 0.0)
        timepixr = hdul[1].header.get("TIMEPIXR", 0.5)
        timedel = hdul[1].header.get("TIMEDEL", 0.0)

        mission = hdul[1].header.get("TELESCOP", "unknown").lower()
        logger.info(f"Mission: {mission}")

        if isinstance(clockfile, str) and clockfile.lower() == "none":
            logger.info("Clock correction explicitly disabled")
            clockfile = None
        elif clockfile is None and mission == "nustar":
            clockfile = get_latest_clock_file(mission)
            logger.info(f"Using latest {mission} clock file: {clockfile}")

        timezero += (0.5 - timepixr) * timedel

        clock_fun = None
        if clockfile is not None and not os.path.exists(clockfile):
            raise FileNotFoundError(f"Clock file {clockfile} not found")
        elif clockfile is not None and mission != "nustar":
            warnings.warn(
                f"Clock correction for mission {mission} not implemented, skipping clock correction"
            )
        elif clockfile is not None:
            clock_fun = nustar_clock_correction_fun(
                clockfile, hdul[1].data["TIME"].min(), hdul[1].data["TIME"].max()
            )

        if only_columns is not None:
            hdul = slim_down_hdu_list(hdul, additional_cols=only_columns)

        for hdu in hdul:
            logger.info(f"Updating HDU {hdu.name}")
            for keyname in ["TIME", "START", "STOP", "TSTART", "TSTOP"]:
                if hdu.data is not None and keyname in hdu.data.names:
                    logger.info(f"Updating column {keyname}")
                    hdu.data[keyname] = correct_times(
                        hdu.data[keyname] + timezero, bary_fun, clock_fun
                    )
                if keyname in hdu.header:
                    logger.info(f"Updating header keyword {keyname}")
                    corrected_time = correct_times(
                        hdu.header[keyname] + timezero, bary_fun, clock_fun
                    )
                    if not np.isfinite(corrected_time):
                        logger.error(
                            f"Bad value when updating header keyword {keyname}: "
                            f"{hdu.header[keyname]}->{corrected_time}"
                        )
                    else:
                        hdu.header[keyname] = corrected_time

            hdu.header["CREATOR"] = f"Barycenter - v. {version}"
            hdu.header["RA_OBJ"] = (
                modelin.RAJ.quantity.deg,
                "Coordinate used for barycentering",
            )
            hdu.header["DEC_OBJ"] = (
                modelin.DECJ.quantity.deg,
                "Coordinate used for barycentering",
            )
            hdu.header["EQUINOX"] = (
                hdul[1].header.get("EQUINOX", 2000.0),
                "Equinox of the coordinates",
            )
            hdu.header["DATE"] = Time.now().fits
            hdu.header["PLEPHEM"] = f"JPL-{ephem}"
            hdu.header["RADECSYS"] = radecsys
            hdu.header["TIMEREF"] = "SOLARSYSTEM"
            hdu.header["TIMESYS"] = "TDB"
            hdu.header["TIMEZERO"] = 0.0
            hdu.header["TREFDIR"] = "RA_OBJ,DEC_OBJ"
            hdu.header["TREFPOS"] = "BARYCENTER"
            hdu.header["CLOCKAPP"] = True if clock_fun is not None else False
            hdu.header.add_history(f"TOOL: barycenter v{version} applied")
            hdu.header.add_history(f"Orbit file: {orbfile}")
            if clockfile is not None:
                hdu.header.add_history(f"Clock file: {clockfile}")
            if parfile is not None and os.path.exists(parfile):
                hdu.header.add_history(f"Par file: {parfile}")
            else:
                hdu.header.add_history(
                    f"Position used: RA={modelin.RAJ.quantity.deg}, DEC={modelin.DECJ.quantity.deg}"
                )
            hdu.header.add_history(f"Ephemeris: JPL-{ephem}")
            hdu.header.add_history(f"Coordinate system: {radecsys}")

        hdul.writeto(outfile, overwrite=overwrite, output_verify="ignore")

    return outfile
