"""Shelling out to the missions' own barycentring tools.

This path exists for cross-checking, and for missions we do not yet correct natively.
It needs the mission's software installed -- HEASOFT for ``barycorr`` and ``timeconv``
-- so it is never exercised in CI.
"""

import logging as logger
import os
import shutil
import subprocess as sp
import tempfile
import warnings

from astropy.io import fits

from .clock import CLOCK_CALDB, get_latest_clock_file
from .missions import mission_for
from .remote import download_locally
from .utils import fits_open_including_remote, slim_down_hdu_list

__all__ = ["official_barycorr", "apply_mission_specific_barycenter_correction"]


def official_barycorr(
    fname,
    orbfile,
    ra=None,
    dec=None,
    ephem="DE440",
    refframe="ICRS",
    outfile="bary.evt",
    clockfile=None,
):
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

    if clockfile is None:
        clockfile = "CALDB"
    logger.info("Applying official barycorr...")
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


def _refuse_to_clobber(outfile, overwrite):
    """Refuse an existing output file unless the caller asked for it to be replaced."""
    if os.path.exists(outfile) and not overwrite:
        raise FileExistsError(
            f"Output file {outfile} already exists. Use overwrite=True to overwrite."
        )


def _copy_decompressing(fname, dest):
    """Copy ``fname`` to ``dest``, gunzipping it on the way if it ends in ``.gz``.

    ``timeconv`` rewrites the file it is given, so it gets this private copy. The input
    is only ever read: gunzipping it in place would change the user's data, fail in a
    read-only directory, and leave a decompressed file that a later run could pick up stale.
    """
    if fname.endswith(".gz"):
        import gzip

        with gzip.open(fname, "rb") as fin, open(dest, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    else:
        shutil.copy(fname, dest)


def _deliver(temp_outfile, outfile, only_columns=None):
    """Put the finished temporary file at ``outfile``, keeping only ``only_columns``.

    ``shutil.move`` rather than ``os.rename``: the temporary file lives in the system
    temporary directory, which on many machines -- any Linux box whose ``/tmp`` is a
    tmpfs, or any run writing to a mounted volume -- is a different filesystem from the
    output. ``os.rename`` refuses to cross one (EXDEV); ``shutil.move`` copies instead.
    """
    if only_columns is not None:
        with fits.open(temp_outfile) as hdul:
            hdul = slim_down_hdu_list(hdul, additional_cols=only_columns)
            hdul.writeto(outfile, overwrite=True, output_verify="ignore")
        os.remove(temp_outfile)
    else:
        shutil.move(temp_outfile, outfile)
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

    _refuse_to_clobber(outfile, overwrite)

    temp_outfile = tempfile.NamedTemporaryFile(delete=False, suffix=".evt").name

    if parfile is not None and os.path.exists(parfile):
        from pint.models import get_model

        modelin = get_model(parfile)
        ra = modelin.RAJ.quantity.deg
        dec = modelin.DECJ.quantity.deg

    with fits_open_including_remote(fname, memmap=True) as hdul:
        telescope = hdul[1].header.get("TELESCOP", "unknown")
    mission = mission_for(telescope)
    logger.info(f"Mission: {mission.name}")

    if mission.official is None:
        raise NotImplementedError(
            f"No official barycentring tool is wired up for {mission.name}. "
            "Name one in its entry in barycenter.missions.MISSIONS, or drop "
            "--apply-official and use the native engine."
        )

    if isinstance(clockfile, str) and clockfile.lower() == "none":
        # barycorr takes "NONE" (upper case) to mean "no clock correction".
        clockfile = "NONE"
    elif clockfile is None and mission.name in CLOCK_CALDB:
        clockfile = get_latest_clock_file(mission.name)
        logger.info(f"Using latest {mission.name} clock file: {clockfile}")

    if mission.official_ephem is not None and ephem != mission.official_ephem:
        warnings.warn(
            f"{mission.official} can only use the {mission.official_ephem} ephemeris for "
            f"{mission.name}, overriding --ephem {ephem}. The native engine has no such "
            "limit."
        )
        if ephem in outfile:
            outfile = outfile.replace(ephem, mission.official_ephem)
            # The guard above ran against the name the caller passed. This is a different
            # file, and it must be protected too, or the delivery below silently replaces
            # it on POSIX whatever ``overwrite`` said.
            _refuse_to_clobber(outfile, overwrite)
        ephem = mission.official_ephem

    if mission.official == "barycorr":
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
    elif mission.official == "timeconv":
        fname = download_locally(fname, outdir=os.path.dirname(outfile))
        _copy_decompressing(fname, temp_outfile)
        # Add download for frf.orbit
        download_locally(
            "https://heasarc.gsfc.nasa.gov/FTP/software/ftools/ALPHA/ftools/refdata/earth.dat",
            outdir=os.path.dirname(outfile),
        )
        download_locally(
            "https://heasarc.gsfc.nasa.gov/FTP/asca/data/trend/orbit/frf.orbit.255",
            outdir=os.path.dirname(outfile),
        )
        cmd = f"timeconv {temp_outfile} 2 {ra:.7f} {dec:.7f} earth.dat frf.orbit.255"
        logger.info(f"Executing {cmd}")
        sp.check_call(cmd.split())

        logger.info("Updating header keywords...")

        def _add_to_header_if_missing(header, label, value, comment):
            if label not in header or header[label].strip() == "":
                header[label] = (value, comment)

        with fits.open(temp_outfile) as hdul:
            _add_to_header_if_missing(hdul[1].header, "TIMESYS", "TDB", "Added by barycenter")
            _add_to_header_if_missing(
                hdul[1].header, "TIMEREF", "SOLARSYSTEM", "Added by barycenter"
            )
            _add_to_header_if_missing(
                hdul[1].header, "PLEPHEM", f"JPL-{ephem}", "Added by barycenter"
            )
            _add_to_header_if_missing(hdul[1].header, "RA_BARY", ra, "Added by barycenter")
            _add_to_header_if_missing(hdul[1].header, "DEC_BARY", dec, "Added by barycenter")
            hdul[1].header.add_history("TOOL: timeconv applied for barycentering")
            hdul.writeto(temp_outfile, overwrite=True, output_verify="ignore")
    else:
        raise NotImplementedError(
            f"Official tool {mission.official!r}, named for {mission.name}, has no wrapper here."
        )

    return _deliver(temp_outfile, outfile, only_columns)
