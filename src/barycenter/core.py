"""The mission-agnostic correction: read a file, correct every time in it, write it out.

This is the pure-Python workflow. It reads the spacecraft orbit through
:mod:`barycenter.orbit`, gets a correction function from one of the engines, optionally
adds a clock correction from :mod:`barycenter.clock`, and applies both to every time
column and time keyword in every extension.
"""

import logging as logger
import os
import tempfile

import astropy.units as u
import numpy as np
from astropy.time import Time

from ._version import __version__
from .clock import clock_correction_fun
from .native import native_barycentric_correction
from .official import apply_mission_specific_barycenter_correction
from .orbit import read_orbit
from .remote import download_locally
from .utils import fits_open_including_remote, slim_down_hdu_list

#: The engines that can compute the correction. ``native`` is the default: it is exact
#: in float64 on every platform, about 30 times faster, and matches HEASOFT ``barycorr``
#: to the resolution the reference files can express. ``pint`` is kept as an independent
#: second opinion and for timing models with proper motion.
ENGINES = ("native", "pint")

__all__ = [
    "ENGINES",
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

    from .pintengine import timing_model_for_position

    with fits_open_including_remote(orbfile, memmap=True) as hdul:
        ra_label, dec_label = get_coordinates_from_fits_header(hdul[1].header)
        ra = hdul[1].header[ra_label]
        dec = hdul[1].header[dec_label]

    # Should check if 12:13:14.2 syntax is used and support that as well!
    return timing_model_for_position(Angle(ra, unit="deg").deg, Angle(dec, unit="deg").deg)


def position_from_model(model, ephem=None):
    """The coordinates and ephemeris a PINT timing model describes.

    Returns
    -------
    ra_deg, dec_deg : float
    ephem : str
        The model's own ``EPHEM``, if it has one. A par file's ephemeris wins over the
        command line, because the model was fitted with it -- and the output header is
        then stamped with the one actually used, rather than with whatever ``--ephem``
        said.
    """
    ra_deg = model.RAJ.quantity.to_value(u.deg)
    dec_deg = model.DECJ.quantity.to_value(u.deg)
    model_ephem = getattr(getattr(model, "EPHEM", None), "value", None)
    return ra_deg, dec_deg, model_ephem or ephem


def get_barycentric_correction(
    orbfile,
    ra=None,
    dec=None,
    ephem="DE440",
    radecsys="ICRS",
    model=None,
    engine="native",
    dt=None,
    met_range=None,
):
    """Get a function to compute the barycentric correction, from MET(TT) to MET(TDB).

    Parameters
    ----------
    orbfile : str or list of str
        Orbit file, a list of them, or an ``"@metafile"``.

    Other Parameters
    ----------------
    ra, dec : float, optional
        Source coordinates in degrees. Required unless ``model`` is given.
    ephem : str, optional
        JPL ephemeris. Default ``"DE440"``.
    radecsys : str, optional
        Frame of the coordinates. The native engine rotates the source direction into
        the frame the ephemeris itself uses, which matters by 45 us between FK5 (DE200)
        and ICRS (DE405 and later). The PINT engine ignores this and assumes ICRS.
    model : pint.models.TimingModel, optional
        A model read from a ``.par`` file. Its coordinates and ephemeris take precedence
        over ``ra``/``dec``/``ephem``.
    engine : {"native", "pint"}, optional
        Which implementation to use. See :data:`ENGINES`.
    dt : float, optional
        Grid spacing in seconds. ``None`` means each engine's own default: the native
        engine evaluates the correction directly at the times asked for, the PINT engine
        uses a 5 s grid.
    met_range : tuple of float, optional
        ``(start, stop)`` in mission elapsed time, to keep a grid from spanning the whole
        orbit file when only a short observation is being corrected.

    Returns
    -------
    bary_fun : callable
        ``bary_fun(met)`` gives the correction in seconds.
    """
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine {engine!r}. Choose one of {ENGINES}.")

    # read_orbit takes a file name, a list of them or an "@metafile", reads the
    # mission's columns from the ORBIT_SPECS registry, and hands back one cleaned
    # table. Both engines read the same table, so they cannot disagree about where the
    # spacecraft was.
    orbit_table = read_orbit(orbfile)

    met = np.asarray(orbit_table["MET"].value, dtype=np.float64)
    if met_range is not None:
        met_range = (max(met.min(), met_range[0]), min(met.max(), met_range[1]))

    if model is not None:
        ra, dec, ephem = position_from_model(model, ephem)

    if engine == "native":
        if ra is None or dec is None:
            raise ValueError("The native engine needs ra and dec, or a timing model")
        return native_barycentric_correction(
            orbit_table,
            ra,
            dec,
            ephem=ephem,
            frame=radecsys,
            dt=dt,
            met_range=met_range,
        )

    from .pintengine import pint_barycentric_correction, timing_model_for_position

    if model is None:
        model = timing_model_for_position(ra, dec, ephem)
    return pint_barycentric_correction(
        orbit_table, model, dt=5.0 if dt is None else dt, met_range=met_range
    )


def correct_times(times, bary_fun, clock_fun=None):
    """Apply the clock and barycentric corrections to an array of times.

    Parameters
    ----------
    times : array-like
        Times to correct, in mission elapsed seconds.
    bary_fun : callable
        The barycentric correction, from :func:`get_barycentric_correction`.
    clock_fun : callable, optional
        The clock correction. If None, no clock correction is applied.

    Returns
    -------
    corrected_times : array-like

    Notes
    -----
    The clock correction is applied **first**, and the barycentric correction is then
    evaluated at the clock-corrected time::

        t' = t + clock(t)
        t_bary = t' + bary(t')

    which is what HEASOFT ``barycorr`` does, and it also means the spacecraft position is
    looked up at ``t'``, since ``bary_fun`` interpolates the orbit at whatever time it is
    given. This package used to compute ``t + clock(t) + bary(t)``, evaluating both on the
    raw mission time. On NuSTAR, where the clock correction reaches 25 ms, that was
    measured at +1146 ns mean and 1878 ns peak against ``barycorr`` -- an order of
    magnitude above the 100 ns target, and the reason the reference tests used to be run
    with the clock correction switched off.
    """
    if clock_fun is None:
        return times + bary_fun(times)

    clock_corrected = times + clock_fun(times)
    return clock_corrected + bary_fun(clock_corrected)


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
    engine="native",
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
    engine : {"native", "pint"}, optional
        Which implementation computes the correction. See :data:`ENGINES`.
    """
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
        # A par file is the only thing that needs PINT on this path: it is the one
        # source of coordinates we do not parse ourselves.
        model = None
        if parfile is not None and os.path.exists(parfile):
            from pint.models import get_model

            model = get_model(parfile)
            ra, dec, ephem = position_from_model(model, ephem)
            logger.info(f"Using coordinates from {parfile}: RA={ra}, Dec={dec}, ephem={ephem}")
        elif ra is None or dec is None:
            ra_str, dec_str = get_coordinates_from_fits_header(hdul[1].header)
            ra = hdul[1].header[ra_str]
            dec = hdul[1].header[dec_str]
            logger.info(f"Using coordinates from header: {ra_str}={ra}, {dec_str}={dec}")

        bary_fun = get_barycentric_correction(
            orbfile,
            ra=ra,
            dec=dec,
            ephem=ephem,
            radecsys=radecsys,
            model=model if engine == "pint" else None,
            engine=engine,
        )

        # TIMEZERO is folded in, but TIMEPIXR deliberately is not. This used to add
        # ``(0.5 - TIMEPIXR) * TIMEDEL``, moving every RXTE PCA event half a clock tick
        # (477 ns) away from what barycorr produces, while leaving TIMEPIXR itself
        # unchanged in the output header -- so the file then claimed a convention its
        # times no longer followed. Where the time stamp sits inside its bin is not the
        # barycentring tool's business.
        timezero = hdul[1].header.get("TIMEZERO", 0.0)

        mission = hdul[1].header.get("TELESCOP", "unknown").lower()
        logger.info(f"Mission: {mission}")

        clock_fun = None
        if isinstance(clockfile, str) and clockfile.lower() == "none":
            logger.info("Clock correction explicitly disabled")
            clockfile = None
        else:
            # Which missions have a clock correction, and where each one comes from, is
            # barycenter.clock's business; this function stays mission-agnostic.
            clock_fun, clockfile = clock_correction_fun(
                mission, clockfile, instrument=hdul[1].header.get("INSTRUME")
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
            hdu.header["RA_OBJ"] = (ra, "Coordinate used for barycentering")
            hdu.header["DEC_OBJ"] = (dec, "Coordinate used for barycentering")
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
            hdu.header.add_history(f"TOOL: barycenter v{version} applied, {engine} engine")
            hdu.header.add_history(f"Orbit file: {orbfile}")
            if clockfile is not None:
                hdu.header.add_history(f"Clock file: {clockfile}")
            if parfile is not None and os.path.exists(parfile):
                hdu.header.add_history(f"Par file: {parfile}")
            else:
                hdu.header.add_history(f"Position used: RA={ra}, DEC={dec}")
            hdu.header.add_history(f"Ephemeris: JPL-{ephem}")
            hdu.header.add_history(f"Coordinate system: {radecsys}")

        hdul.writeto(outfile, overwrite=overwrite, output_verify="ignore")

    return outfile
