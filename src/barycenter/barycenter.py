import os
import numpy as np

from astropy.io import fits
import logging as logger
from pint.observatory.satellite_obs import get_satellite_observatory
from pint.models import get_model, StandardTimingModel
import astropy.units as u
from stingray.io import high_precision_keyword_read
from pint.fits_utils import read_fits_event_mjds
from astropy.table import Table, vstack


from astropy.table import Table
from scipy.interpolate import interp1d
from astropy import log
import pint.models
import pint.toa as toa
from pint.models import StandardTimingModel
from pint.observatory.satellite_obs import get_satellite_observatory

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import Angle
from scipy.interpolate import Akima1DInterpolator
import barycenter


def load_Fermi_FT2(ft2_filename):
    """Load data from a Fermi FT2 file

    The contents of the FT2 file are described here:
    https://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data/LAT_Data_Columns.html#SpacecraftFile
    The coordinates are X, Y, Z in the ECI (Earth-Centered Inertial)
    frame. I (@paulray) **believe** this is the same as astropy's GCRS
    <http://docs.astropy.org/en/stable/api/astropy.coordinates.GCRS.html>,
    but this should be confirmed.

    Parameters
    ----------
    ft2_filename : str
        Name of file to load

    Returns
    -------
    astropy Table containing Time, x, y, z, v_x, v_y, v_z data

    """
    # Load photon times from FT1 file
    hdulist = fits_open_including_remote(ft2_filename)
    FT2_hdr = hdulist[1].header
    FT2_dat = hdulist[1].data

    logger.info("Opened FT2 FITS file {0}".format(ft2_filename))
    # TIMESYS should be 'TT'
    # TIMEREF should be 'LOCAL', since no delays are applied
    timesys = FT2_hdr["TIMESYS"]
    logger.info("FT2 TIMESYS {0}".format(timesys))
    timeref = FT2_hdr["TIMEREF"]
    logger.info("FT2 TIMEREF {0}".format(timeref))

    # The X, Y, Z position are for the START time
    mjds_TT = read_fits_event_mjds(hdulist[1], timecolumn="START")
    mjds_TT = mjds_TT * u.d
    # SC_POS is in meters in X,Y,Z Earth-centered Inertial (ECI) coordinates
    SC_POS = FT2_dat.field("SC_POSITION")
    X = SC_POS[:, 0] * u.m
    Y = SC_POS[:, 1] * u.m
    Z = SC_POS[:, 2] * u.m
    try:
        # If available, get the velocities from the FT2 file
        SC_VEL = FT2_dat.field("SC_VELOCITY")
        Vx = SC_VEL[:, 0] * u.m / u.s
        Vy = SC_VEL[:, 1] * u.m / u.s
        Vz = SC_VEL[:, 2] * u.m / u.s
    except Exception:
        # Otherwise, compute velocities by differentiation because FT2 does not have velocities
        # This is not the best way. Should fit an orbit and determine velocity from that.
        dt = mjds_TT[1] - mjds_TT[0]
        logger.info(f"FT2 spacing is {str(dt.to(u.s))}")
        # Use "spacing" argument for gradient to handle nonuniform entries
        tt = mjds_TT.to(u.s).value
        Vx = np.gradient(X.value, tt) * u.m / u.s
        Vy = np.gradient(Y.value, tt) * u.m / u.s
        Vz = np.gradient(Z.value, tt) * u.m / u.s
    logger.info("Building FT2 table covering MJDs {0} to {1}".format(mjds_TT.min(), mjds_TT.max()))
    return Table(
        [mjds_TT, X, Y, Z, Vx, Vy, Vz],
        names=("MJD_TT", "X", "Y", "Z", "Vx", "Vy", "Vz"),
        meta={"name": "FT2"},
    )


def load_FPorbit(orbit_filename):
    """Load data from an (RXTE or NICER) FPorbit file

    Reads a FPorbit FITS file

    Parameters
    ----------
    orbit_filename : str
        Name of file to load

    Returns
    -------
    astropy Table containing Time, x, y, z, v_x, v_y, v_z data

    """
    # Load orbit FITS file
    hdulist = fits_open_including_remote(orbit_filename)
    # logger.info('orb file HDU name is {0}'.format(hdulist[1].name))
    if hdulist[1].name not in ("ORBIT", "XTE_PE"):
        logger.error(
            "NICER orb file first extension is {0}. It should be ORBIT".format(hdulist[1].name)
        )
    FPorbit_hdr = hdulist[1].header
    FPorbit_dat = hdulist[1].data

    logger.info("Opened FPorbit FITS file {0}".format(orbit_filename))
    # TIMESYS should be 'TT'

    # TIMEREF should be 'LOCAL', since no delays are applied

    if "TIMESYS" not in FPorbit_hdr:
        logger.warning("Keyword TIMESYS is missing. Assuming TT")
        timesys = "TT"
    else:
        timesys = FPorbit_hdr["TIMESYS"]
        logger.debug("FPorbit TIMESYS {0}".format(timesys))

    if "TIMEREF" not in FPorbit_hdr:
        logger.warning("Keyword TIMESYS is missing. Assuming TT")
        timeref = "LOCAL"
    else:
        timeref = FPorbit_hdr["TIMEREF"]
        logger.debug("FPorbit TIMEREF {0}".format(timeref))

    mjds_TT = read_fits_event_mjds(hdulist[1])

    mjds_TT = mjds_TT * u.d
    logger.debug("FPorbit spacing is {0}".format((mjds_TT[1] - mjds_TT[0]).to(u.s)))
    X = FPorbit_dat.field("X") * u.m
    Y = FPorbit_dat.field("Y") * u.m
    Z = FPorbit_dat.field("Z") * u.m
    Vx = FPorbit_dat.field("Vx") * u.m / u.s
    Vy = FPorbit_dat.field("Vy") * u.m / u.s
    Vz = FPorbit_dat.field("Vz") * u.m / u.s
    logger.info(
        "Building FPorbit table covering MJDs {0} to {1}".format(mjds_TT.min(), mjds_TT.max())
    )
    FPorbit_table = Table(
        [mjds_TT, X, Y, Z, Vx, Vy, Vz],
        names=("MJD_TT", "X", "Y", "Z", "Vx", "Vy", "Vz"),
        meta={"name": "FPorbit"},
    )
    # Make sure table is sorted by time
    logger.debug("Sorting FPorbit table")
    FPorbit_table.sort("MJD_TT")

    good = np.diff(FPorbit_table["MJD_TT"]) > 0
    if not np.all(good):
        logger.warning("The orbit table has duplicate entries. Please check.")
        good = np.concatenate((good, [True]))
        FPorbit_table = FPorbit_table[good]

    # Now delete any bad entries where the positions are 0.0
    idx = np.where(np.logical_and(FPorbit_table["X"] != 0.0, FPorbit_table["Y"] != 0.0))[0]
    if len(idx) != len(FPorbit_table):
        logger.warning(
            "Dropping {0} zero entries from FPorbit table".format(len(FPorbit_table) - len(idx))
        )
        FPorbit_table = FPorbit_table[idx]
    return FPorbit_table


def load_nustar_orbit(orb_filename):
    """Load data from a NuSTAR orbit file

    Parameters
    ----------
    orb_filename : str
        Name of file to load

    Returns
    -------
    astropy.table.Table
        containing Time, x, y, z, v_x, v_y, v_z data

    """
    # Load photon times from FT1 file

    if "_orb" in orb_filename:
        logger.warning(
            "The NuSTAR orbit file you are providing is known to give"
            "a solution precise only to the ~0.5ms level. Use the "
            "pipeline-produced attitude-orbit file ('*.attorb.gz') for"
            "better precision."
        )

    hdulist = fits_open_including_remote(orb_filename)
    orb_hdr = hdulist[1].header
    orb_dat = hdulist[1].data

    logger.info("Opened orb FITS file {0}".format(orb_filename))
    # TIMESYS should be 'TT'
    # TIMEREF should be 'LOCAL', since no delays are applied
    timesys = orb_hdr["TIMESYS"]
    logger.info("orb TIMESYS {0}".format(timesys))
    try:
        timeref = orb_hdr["TIMEREF"]
    except KeyError:
        timeref = "LOCAL"

    logger.info("orb TIMEREF {0}".format(timeref))

    # The X, Y, Z position are for the START time
    mjds_TT = read_fits_event_mjds(hdulist[1])
    mjds_TT = mjds_TT * u.d
    # SC_POS is in meters in X,Y,Z Earth-centered Inertial (ECI) coordinates
    SC_POS = orb_dat.field("POSITION")
    X = SC_POS[:, 0] * u.km
    Y = SC_POS[:, 1] * u.km
    Z = SC_POS[:, 2] * u.km
    SC_VEL = orb_dat.field("VELOCITY")
    Vx = SC_VEL[:, 0] * u.km / u.s
    Vy = SC_VEL[:, 1] * u.km / u.s
    Vz = SC_VEL[:, 2] * u.km / u.s

    logger.info("Building orb table covering MJDs {0} to {1}".format(mjds_TT.min(), mjds_TT.max()))
    return Table(
        [mjds_TT, X, Y, Z, Vx, Vy, Vz],
        names=("MJD_TT", "X", "Y", "Z", "Vx", "Vy", "Vz"),
        meta={"name": "orb"},
    )


def load_orbit(obs_name, orb_filename):
    """Generalized function to load one or more orbit files.

    Parameters
    ----------
    obs_name : str
        Observatory name. (Fermi, NICER, RXTE, and NuSTAR are valid.)
    orb_filename : str
        An FT2-like file tabulating orbit position.  If the first character
        is @, interpreted as a metafile listing multiple orbit files.

    Returns
    -------
    orb_table: astropy.table.Table
        A table containing entries MJD_TT, X, Y, Z, Vx, Vy, Vz
    """

    if str(orb_filename).startswith("@"):
        # Read multiple orbit files names
        fnames = [ll.strip() for ll in open(orb_filename[1:]).readlines()]
        orb_list = [load_orbit(obs_name, fn) for fn in fnames]
        full_orb = vstack(orb_list)
        # Make sure full table is sorted
        full_orb.sort("MJD_TT")
        return full_orb

    lower_name = obs_name.lower()
    if "fermi" in lower_name:
        return load_Fermi_FT2(orb_filename)
    elif "nicer" in lower_name:
        return load_FPorbit(orb_filename)
    elif "ixpe" in lower_name:
        return load_FPorbit(orb_filename)
    elif "xte" in lower_name:
        return load_FPorbit(orb_filename)
    elif "nustar" in lower_name:
        return load_nustar_orbit(orb_filename)
    else:
        raise ValueError(f"Unrecognized satellite observatory {obs_name}.")


import pint
import pint.observatory
import pint.observatory.satellite_obs

# Monkey-patch pint to use our improved load_orbit function
pint.observatory.satellite_obs.load_orbit = load_orbit


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
    except (PermissionError, botocore.exceptions.NoCredentialsError) as e:
        if "://" in filename:
            logger.info(f"Permission denied for {filename}, trying with fsspec.")
            return fits.open(filename, use_fsspec=True, fsspec_kwargs={"anon": True}, **kwargs)
        raise e


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


class OrbitalFunctions:
    lat_fun = None
    lon_fun = None
    alt_fun = None
    lst_fun = None


# def get_orbital_functions(orbfile):
#     from astropy.time import Time
#     import astropy.units as u

#     orbtable = Table.read(orbfile)
#     mjdref = high_precision_keyword_read(orbtable.meta, "MJDREF")

#     times = Time(np.array(orbtable["TIME"] / 86400 + mjdref), format="mjd")
#     if "GEODETIC" in orbtable.colnames:
#         geod = np.array(orbtable["GEODETIC"])
#         lat, lon, alt = geod[:, 0] * u.deg, geod[:, 1] * u.deg, geod[:, 2] * u.m
#     else:
#         geod = np.array(orbtable["POLAR"])
#         lat, lon, alt = (
#             (geod[:, 0] * u.rad).to(u.deg),
#             (geod[:, 1] * u.rad).to(u.deg),
#             geod[:, 2] * 1000 * u.m,
#         )

#     lat_fun = interp1d(times.mjd, lat, bounds_error=False, fill_value="extrapolate")
#     lon_fun = interp1d(times.mjd, lon, bounds_error=False, fill_value="extrapolate")
#     alt_fun = interp1d(times.mjd, alt, bounds_error=False, fill_value="extrapolate")
#     gst = times.sidereal_time("apparent", "greenwich")
#     lst = lon.to(u.hourangle) + gst.to(u.hourangle)
#     lst[lst.value > 24] -= 24 * u.hourangle
#     lst[lst.value < 0] += 24 * u.hourangle
#     lst_fun = interp1d(times.mjd, lst, bounds_error=False, fill_value="extrapolate")

#     orbfunc = OrbitalFunctions()
#     orbfunc.lat_fun = lat_fun
#     orbfunc.lon_fun = lon_fun
#     orbfunc.alt_fun = alt_fun
#     orbfunc.lst_fun = lst_fun

#     return orbfunc


def get_dummy_parfile_for_position(orbfile):

    # Construct model by hand
    with fits.open(orbfile, memmap=True) as hdul:
        label = "_NOM"
        if "RA_OBJ" in hdul[1].header:
            label = "_OBJ"
        ra = hdul[1].header[f"RA{label}"]
        dec = hdul[1].header[f"DEC{label}"]

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
):
    with fits_open_including_remote(orbfile) as hdul:
        mjdref = high_precision_keyword_read(hdul[1].header, "MJDREF")
        telescope = hdul[1].header["TELESCOP"].lower()

    no = get_satellite_observatory(telescope, orbfile, overwrite=True)

    knots = no.X.get_knots()
    mjds = np.arange(knots[1], knots[-2], dt / 86400)
    mets = (mjds - mjdref) * 86400

    obs, scale = telescope.lower(), "tt"
    toalist = [None] * len(mjds)

    for i in range(len(mjds)):
        # Create TOA list
        toalist[i] = toa.TOA(mjds[i], obs=obs, scale=scale)

    ts = toa.get_TOAs_list(
        toalist,
        ephem=modelin.EPHEM.value,
        include_bipm=False,
        planets="PLANET_SHAPIRO" in modelin.params and modelin.PLANET_SHAPIRO.value,
        tdb_method="default",
    )
    bats = modelin.get_barycentric_toas(ts)
    return Akima1DInterpolator(
        mets,
        (bats.value - mjds) * 86400,
        extrapolate=True,
    )


def correct_times(times, bary_fun, clock_fun=None):
    cl_corr = 0
    if clock_fun is not None:
        cl_corr = clock_fun(times)
    bary_corr = bary_fun(times)

    return times + cl_corr + bary_corr


from numba import vectorize, float64


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


def apply_clock_correction(
    fname,
    orbfile,
    outfile="bary.evt",
    clockfile=None,
    parfile=None,
    ephem="DE440",
    radecsys="ICRS",
    ra=None,
    dec=None,
    overwrite=False,
):
    version = barycenter.__version__
    if parfile is not None and os.path.exists(parfile):
        modelin = get_model(parfile)
    else:
        if ra is None or dec is None:
            raise ValueError("Either parfile or both ra and dec must be provided")

        modelin = StandardTimingModel
        modelin.RAJ.quantity = ra * u.deg
        modelin.DECJ.quantity = dec * u.deg
        modelin.DM.quantity = 0.0
        modelin.RADECSYS.value = radecsys
        modelin.EPHEM.value = ephem

    bary_fun = get_barycentric_correction(orbfile, modelin)

    with fits.open(fname, memmap=True) as hdul:
        times = hdul[1].data["TIME"]
        unique_times = np.unique(times)
        clock_fun = None
        if clockfile is not None and os.path.exists(clockfile):
            hduname = "NU_FINE_CLOCK"
            logger.info(f"Read extension {hduname}")
            clocktable = Table.read(clockfile, hdu=hduname)
            clock_corr, _ = interpolate_clock_function(clocktable, unique_times)
            clock_fun = interp1d(
                unique_times,
                clock_corr,
                assume_sorted=True,
                bounds_error=False,
                fill_value="extrapolate",
            )
        elif clockfile is not None and not os.path.exists(clockfile):
            raise FileNotFoundError(f"Clock file {clockfile} not found")

        for hdu in hdul:
            logger.info(f"Updating HDU {hdu.name}")
            for keyname in ["TIME", "START", "STOP", "TSTART", "TSTOP"]:
                if hdu.data is not None and keyname in hdu.data.names:
                    logger.info(f"Updating column {keyname}")
                    hdu.data[keyname] = correct_times(hdu.data[keyname], bary_fun, clock_fun)
                if keyname in hdu.header:
                    logger.info(f"Updating header keyword {keyname}")
                    corrected_time = correct_times(hdu.header[keyname], bary_fun, clock_fun)
                    if not np.isfinite(corrected_time):
                        logger.error(
                            f"Bad value when updating header keyword {keyname}: "
                            f"{hdu.header[keyname]}->{corrected_time}"
                        )
                    else:
                        hdu.header[keyname] = corrected_time

            hdu.header["CREATOR"] = f"Barycenter - v. {version}"
            hdu.header["RA_OBJ"] = (
                modelin.RAJ.degree,
                "Coordinate used for barycentering",
            )
            hdu.header["DEC_OBJ"] = (
                modelin.DECJ.degree,
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
        hdul.writeto(outfile, overwrite=overwrite)


def splitext_improved(path):
    """
    Examples
    --------
    >>> np.all(splitext_improved("a.tar.gz") ==  ('a', '.tar.gz'))
    True
    >>> np.all(splitext_improved("a.tar") ==  ('a', '.tar'))
    True
    >>> np.all(splitext_improved("a.f/a.tar") ==  ('a.f/a', '.tar'))
    True
    >>> np.all(splitext_improved("a.a.a.f/a.tar.gz") ==  ('a.a.a.f/a', '.tar.gz'))
    True
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


def _default_out_file(args):
    outfile = "bary_" + os.path.basename(args.file).replace(".evt", "").replace(".evt", "")

    outfile += ".evt"

    return outfile


def main_barycenter(args=None):
    import argparse

    description = "Apply the barycenter correction to NuSTAR" "event files"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("file", help="Uncorrected event file")
    parser.add_argument("orbitfile", help="Orbit file")
    parser.add_argument(
        "-p",
        "--parfile",
        help="Parameter file in TEMPO/TEMPO2/PINT " "format (for precise coordinates)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--ra", help="Right ascension (deg) if no parfile", default=None, type=float
    )
    parser.add_argument("--dec", help="Declination (deg) if no parfile", default=None, type=float)
    parser.add_argument(
        "--radecsys", help="Coordinate system (default ICRS)", default="ICRS", type=str
    )
    parser.add_argument(
        "--ephem", help="Solar system ephemeris (default DE400)", default="DE400", type=str
    )

    parser.add_argument(
        "-o", "--outfile", default=None, help="Output file name (default bary_<opts>.evt)"
    )
    parser.add_argument("-c", "--clockfile", default=None, help="Clock correction file")
    parser.add_argument(
        "--overwrite", help="Overwrite existing data", action="store_true", default=False
    )

    args = parser.parse_args(args)

    outfile = args.outfile
    if outfile is None:
        outfile = _default_out_file(args)

    apply_clock_correction(
        args.file,
        args.orbitfile,
        parfile=args.parfile,
        outfile=outfile,
        overwrite=args.overwrite,
        clockfile=args.clockfile,
    )

    return outfile


if __name__ == "__main__":
    main_barycenter()
