import os
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
from .utils import fits_open_including_remote
import barycenter.monkeypatch  # noqa: F401


class OrbitalFunctions:
    lat_fun = None
    lon_fun = None
    alt_fun = None
    lst_fun = None


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


def get_coordinates_from_fits_header(hdr):
    if "RA_OBJ" in hdr:
        return "RA_OBJ", "DEC_OBJ"
    elif "RA_PNT" in hdr:
        return "RA_PNT", "DEC_PNT"
    elif "RA_NOM" in hdr:
        return "RA_NOM", "DEC_NOM"
    else:
        raise ValueError("No coordinates found in header")


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
    with fits.open(fname, memmap=True) as hdul:
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
        "--radecsys",
        help="Coordinate system (default ICRS for DE4XX, FK5 for DE200)",
        default=None,
        type=str,
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

    if args.radecsys is None:
        if args.ephem == "DE200":
            args.radecsys = "FK5"
        else:
            args.radecsys = "ICRS"

    apply_clock_correction(
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
    )

    return outfile


if __name__ == "__main__":
    main_barycenter()
