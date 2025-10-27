import os
import glob
import shutil
import subprocess as sp
import warnings
import logging as logger
import numpy as np

from astropy.io import fits
from pint.observatory.satellite_obs import get_satellite_observatory
from pint.models import get_model, StandardTimingModel
import astropy.units as u
from stingray.io import high_precision_keyword_read
from pint.fits_utils import read_fits_event_mjds
from astropy.table import Table, vstack


from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.interpolate import Akima1DInterpolator
from astropy import log
import pint.models
import pint.toa as toa
from pint.models import StandardTimingModel
from pint.observatory.satellite_obs import get_satellite_observatory

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import Angle
import barycenter
from .utils import fits_open_including_remote, slim_down_hdu_list, get_remote_directory_listing
import barycenter.monkeypatch  # noqa: F401


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

    Returns
    -------
    bary_fun : callable
        Function to compute barycentric correction.
    """
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
    """
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


def official_barycorr(fname, orbfile, ra=None, dec=None, ephem="DE440", refframe="ICRS", outfile="bary.evt", clockfile=None):
    """Apply barycorr to a FITS event file.

    Parameters
    ----------
    fname : str
        Input FITS event file.
    orbfile : str
        Orbit file.
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    ephem : str, optional
        Ephemeris model to use. Default is "DE440".
    """
    import heasoftpy as hsp
    print("Applying official barycorr...")
    hsp.barycorr(
        infile=fname,
        outfile=outfile,
        ra=ra,
        dec=dec,
        ephem="JPLEPH." + ephem.replace("DE", ""),
        refframe=refframe,
        clobber="yes",
        orbitfiles=orbfile,
        clockfile=clockfile,
        verbose=1,
    )
    if not os.path.exists(outfile):
        raise RuntimeError(f"heasoft barycorr failed to produce output file {outfile}")

    return outfile


def apply_mission_specific_barycenter_correction(
    fname,
    orbfile,
    outfile="bary.evt",
    clockfile=None,
    parfile=None,
    ra=None,
    dec=None,
    ephem="DE440",
    radecsys="ICRS",
    overwrite=False,
    only_columns=None,
):
    """Apply mission-specific barycenter correction to a FITS event file.

    Parameters
    ----------
    fname : str
        Input FITS event file.
    orbfile : str
        Orbit file.
    mission : str
        Mission name (e.g., 'nustar').

    Other Parameters
    ----------------
    outfile : str, optional
        Output FITS event file. Default is "bary.evt".
    clockfile : str, optional
        Clock file.
    ra : float, optional
        Right Ascension in degrees. If not provided, will be read from header.
    dec : float, optional
        Declination in degrees. If not provided, will be read from header.
    ephem : str, optional
        Ephemeris model to use. Default is "DE440".
    radecsys : str, optional
        Coordinate system for RA/Dec. Default is "ICRS".
    overwrite : bool, optional
        If True, will overwrite existing output file. Default is False.
    only_columns : list of str, optional
        List of column names to keep in the output file, in addition to the "TIME" column.
    """
    import tempfile
    if os.path.exists(outfile) and not overwrite:
        raise FileExistsError(f"Output file {outfile} already exists. Use overwrite=True to overwrite.")

    temp_outfile = tempfile.NamedTemporaryFile(delete=False, suffix=".evt").name

    if parfile is not None and os.path.exists(parfile):
        modelin = get_model(parfile)
        ra = modelin.RAJ.quantity.deg
        dec = modelin.DECJ.quantity.deg

    with fits_open_including_remote(fname, memmap=True) as hdul:
        mission = hdul[1].header.get("TELESCOP", "unknown").lower()
        logger.info(f"Mission: {mission}")

    if clockfile is None and clockfile != "none" and mission == "nustar":
        clockfile = get_latest_clock_file(mission)
        logger.info(f"Using latest {mission} clock file: {clockfile}")

    if mission.lower() in ["nustar", "nicer", "xte", "rxte", "swift", "axaf", "chandra"]:
        official_barycorr(
            fname,
            orbfile,
            outfile=temp_outfile,
            clockfile=clockfile,
            ra=ra,
            dec=dec,
            ephem=ephem,
            refframe=radecsys,
        )
    elif mission.lower() == "asca":
        fname = download_locally(fname)

        if fname.endswith(".gz"):
            sp.check_call(["gunzip", fname])
            fname = fname[:-3]
        shutil.copy(fname, temp_outfile)
        # Add download for frf.orbit
        download_locally("https://heasarc.gsfc.nasa.gov/FTP/software/ftools/ALPHA/ftools/refdata/earth.dat")
        download_locally("https://heasarc.gsfc.nasa.gov/FTP/asca/data/trend/orbit/frf.orbit.255")
        cmd = f"timeconv {temp_outfile} 2 {ra} {dec} earth.dat frf.orbit.255"
        print(f"Executing {cmd}")
        sp.check_call(cmd.split())
    else:
        raise NotImplementedError(f"Barycenter correction for mission {mission} not implemented")

    if only_columns is not None:
        with fits.open(temp_outfile) as hdul:
            hdul = slim_down_hdu_list(hdul, additional_cols=only_columns)
            hdul.writeto(outfile, overwrite=True)
        os.remove(temp_outfile)
    else:
        os.rename(temp_outfile, outfile)
    return outfile


def download_locally(fname):
    """Download a remote file locally if needed.
    Manages S3 and HTTP(s) URLs. For S3, only public buckets are supported at the moment

    Parameters
    ----------
    fname : str
        Input file path or URL.
    Returns
    -------
    local_fname : str
        Local file path.
    """
    if fname.startswith("http://") or fname.startswith("https://"):
        from astropy.utils.data import download_file

        local_fname = download_file(fname, cache=True)
        logger.info(f"Downloaded remote file {fname} to local file {local_fname}")
        return local_fname
    elif fname.startswith("s3://"):
        import boto3
        import botocore
        from urllib.parse import urlparse

       # Parse S3 URL
        parsed = urlparse(fname)
        bucket_name = parsed.netloc
        config = botocore.client.Config(signature_version=botocore.UNSIGNED)
        s3_resource = boto3.resource("s3", config=config)
        s3_client = s3_resource.meta.client
        path = fname.replace(f"s3://{bucket_name}/", "")
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=path)
        objects = response.get("Contents", [])
        if len(objects) == 0:
            print(objects)
            raise FileNotFoundError(f"No objects found at S3 path {fname}")
        key = objects[0]["Key"]
        path2 = "/".join(path.strip("/").split("/")[:-1])
        dest = key[len(path2) + 1 :]
        if os.path.exists(dest):
            logger.info(f"{dest} already exists, skipping download.")
        else:
            s3_client.download_file(bucket_name, key, dest)
        logger.info(f"Downloaded remote file {fname} to local file {dest}")
        return dest

    return fname


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
    cloud = "SCISERVER_USER_ID" in os.environ or "/home/jovyan" in os.environ.get("HOME", "")

    if apply_official or not cloud:
        fname = download_locally(fname)
        orbfile = download_locally(orbfile)

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

    version = barycenter.__version__
    with fits_open_including_remote(fname, memmap=True) as hdul:
        if parfile is not None and os.path.exists(parfile):
            modelin = get_model(parfile)
        else:
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

        if clockfile is None and clockfile != "none" and mission == "nustar":
            clockfile = get_latest_clock_file(mission)
            logger.info(f"Using latest {mission} clock file: {clockfile}")

        timezero += (0.5 - timepixr) * timedel

        clock_fun = None
        if clockfile is not None and os.path.exists(clockfile):
            if mission != "nustar":
                warnings.warn(
                    f"Clock correction for mission {mission} not implemented, skipping clock correction"
                )
            clock_fun = nustar_clock_correction_fun(
                clockfile, hdul[1].data["TIME"].min(), hdul[1].data["TIME"].max()
            )
        elif clockfile is not None and not os.path.exists(clockfile):
            raise FileNotFoundError(f"Clock file {clockfile} not found")

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
    if args.only_columns is not None:
        outfile += "_slim"
    if args.clockfile == "none":
        outfile += "_noclk"
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
        "--radecsys",
        help="Coordinate system (default ICRS for DE4XX, FK5 for DE200)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--ephem", help="Solar system ephemeris (default DE440)", default="DE440", type=str
    )

    parser.add_argument(
        "-o", "--outfile", default=None, help="Output file name (default bary_<opts>.evt)"
    )
    parser.add_argument(
        "-c",
        "--clockfile",
        default=None,
        help=(
            "Clock correction file. If not provided, the latest clock file will be used for NuSTAR."
            " Specify 'none' to skip clock correction."
        ),
    )
    parser.add_argument(
        "--overwrite", help="Overwrite existing data", action="store_true", default=False
    )
    parser.add_argument(
        "--apply-official",
        help="Use mission-specific official barycenter correction (e.g., heasoft barycorr)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--only-columns",
        type=str,
        default=None,
        help="Only keep these additional columns in the output file, "
        "in addition to the TIME column. It is a comma separated list, like PI,PRIOR",
    )

    args = parser.parse_args(args)

    outfile = args.outfile
    if outfile is None:
        outfile = _default_out_file(args)

    if args.radecsys is None:
        if args.ephem == "DE200":
            args.radecsys = "FK5"
        else:
            args.radecsys = "ICRS"

    apply_barycenter_correction(
        args.file,
        args.orbitfile,
        parfile=args.parfile,
        outfile=outfile,
        overwrite=args.overwrite,
        clockfile=args.clockfile,
        ephem=args.ephem,
        radecsys=args.radecsys,
        ra=args.ra,
        dec=args.dec,
        only_columns=args.only_columns.split(",") if args.only_columns else None,
        apply_official=args.apply_official,
    )

    return outfile


if __name__ == "__main__":
    main_barycenter()
