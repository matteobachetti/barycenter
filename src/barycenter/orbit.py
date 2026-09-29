"""Reading spacecraft orbit files.

Every mission tabulates the same thing -- time, geocentric position, geocentric
velocity -- and differs only in which extension it lives in, what the columns are
called, whether the position is one vector column or three scalars, and whether the
units are metres or kilometres. So there is one reader, driven by an
:class:`OrbitSpec`, and adding a mission is one entry in :data:`ORBIT_SPECS`.

The result is a single table that both engines consume, always in metres and metres per
second: the ``MJD_TT``/``X``...``Vz`` columns are the contract PINT's ``SatelliteObs``
expects, and the ``MET`` column is what the native engine uses, so neither engine has to
re-derive the other's time base.
Reading the orbit file once, in one place, is what makes a native-versus-PINT
comparison meaningful.
"""

import logging as logger
from collections.abc import Iterable
from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from astropy.table import Table, vstack

from .utils import fits_open_including_remote

__all__ = ["OrbitSpec", "ORBIT_SPECS", "read_orbit", "spec_for_mission"]


@dataclass(frozen=True)
class OrbitSpec:
    """How to find position and velocity in one mission's orbit file.

    Parameters
    ----------
    pos : str or tuple of str
        Either the name of a single column holding an (x, y, z) vector, or the three
        scalar column names.
    vel : str or tuple of str or None
        The same for velocity. ``None``, or a column that turns out to be absent, means
        the velocity is obtained by differentiating the position -- adequate, but the
        tabulated velocity is always better, so prefer it when the file has one.
    pos_unit, vel_unit : astropy unit
        The units the file uses. Not read from the column: orbit files are unreliable
        about `TUNITn`, and getting a factor of 1000 wrong here is a 20 ms error.
    hdu : int or str
        Extension holding the table.
    time_col : str
        Column holding the mission elapsed time.
    expected_extnames : tuple of str
        Extension names that are known to be right. A file whose name is not in the
        list still gets read, with a warning -- it is a sanity check, not a gate.
    """

    pos: "str | tuple" = "POSITION"
    vel: "str | tuple | None" = "VELOCITY"
    pos_unit: u.UnitBase = u.m
    vel_unit: u.UnitBase = u.m / u.s
    hdu: "int | str" = 1
    time_col: str = "TIME"
    expected_extnames: tuple = field(default_factory=tuple)


#: One entry per mission, keyed by a substring of the ``TELESCOP`` keyword.
ORBIT_SPECS = {
    "fermi": OrbitSpec(
        pos="SC_POSITION",
        vel="SC_VELOCITY",
        time_col="START",
        expected_extnames=("SC_DATA",),
    ),
    # NuSTAR is the only one in kilometres.
    "nustar": OrbitSpec(pos="POSITION", vel="VELOCITY", pos_unit=u.km, vel_unit=u.km / u.s),
    "svom": OrbitSpec(pos="POSITION", vel="VELOCITY"),
    # The FPorbit shape, shared by NICER, RXTE and IXPE: three scalar columns.
    "nicer": OrbitSpec(pos=("X", "Y", "Z"), vel=("Vx", "Vy", "Vz"), expected_extnames=("ORBIT",)),
    "xte": OrbitSpec(
        pos=("X", "Y", "Z"), vel=("Vx", "Vy", "Vz"), expected_extnames=("ORBIT", "XTE_PE")
    ),
    "ixpe": OrbitSpec(pos=("X", "Y", "Z"), vel=("Vx", "Vy", "Vz"), expected_extnames=("ORBIT",)),
}


def spec_for_mission(telescope):
    """The :class:`OrbitSpec` for a ``TELESCOP`` keyword value.

    Matching is by substring, because the keyword is written inconsistently
    (``XTE``, ``RXTE``, ``NuSTAR``, ``NUSTAR``).
    """
    name = str(telescope).lower()
    for key, spec in ORBIT_SPECS.items():
        if key in name:
            return spec
    raise ValueError(
        f"No orbit file specification for mission {telescope!r}. "
        f"Known missions: {', '.join(sorted(ORBIT_SPECS))}. "
        "Adding one is a single entry in barycenter.orbit.ORBIT_SPECS."
    )


def _columns(data, names, unit):
    """Pull an (N, 3) array out of either one vector column or three scalar ones."""
    if isinstance(names, str):
        vec = np.asarray(data.field(names), dtype=np.float64)
        if vec.ndim != 2 or vec.shape[1] != 3:
            raise ValueError(f"Column {names} has shape {vec.shape}, expected (N, 3)")
    else:
        vec = np.column_stack([np.asarray(data.field(n), dtype=np.float64) for n in names])
    return vec * unit


def _read_one(fname, spec=None):
    """Read a single orbit file into the common table."""
    from .utils import high_precision_mjdref

    with fits_open_including_remote(fname) as hdul:
        hdu = hdul[spec.hdu] if spec is not None else hdul[1]
        header, data = hdu.header, hdu.data
        telescope = header.get("TELESCOP", "unknown")
        if spec is None:
            spec = spec_for_mission(telescope)
            hdu = hdul[spec.hdu]
            header, data = hdu.header, hdu.data

        if spec.expected_extnames and hdu.name not in spec.expected_extnames:
            logger.warning(
                f"Orbit extension of {fname} is {hdu.name!r}, expected one of "
                f"{spec.expected_extnames}. Reading it anyway."
            )
        if header.get("TIMESYS", "TT").upper() != "TT":
            logger.warning(f"Orbit file {fname} has TIMESYS={header.get('TIMESYS')}, not TT")

        mjdref = high_precision_mjdref(header)
        timezero = header.get("TIMEZERO", 0.0)
        met = np.asarray(data.field(spec.time_col), dtype=np.float64) + timezero
        pos = _columns(data, spec.pos, spec.pos_unit)

        if spec.vel is not None and _has_columns(data, spec.vel):
            vel = _columns(data, spec.vel, spec.vel_unit)
        else:
            # Differentiating is a fallback, not a choice: it is noisier than the
            # tabulated velocity and only affects the topocentric Einstein term, which
            # is microseconds, so it is tolerable when a file simply has no velocities.
            logger.warning(
                f"Orbit file {fname} has no velocity columns; differentiating the position instead."
            )
            vel = np.gradient(pos.to_value(u.m), met, axis=0) * u.m / u.s

    # One unit throughout, whatever the file used. PINT reads the unit back off the
    # table and so would cope with kilometres, but the native engine and every test
    # would then have to remember which mission is which.
    pos = pos.to(u.m)
    vel = vel.to(u.m / u.s)

    # MJD_TT is what PINT wants; MET is what the native engine wants. Computing MJD_TT
    # in long double first keeps the reference epoch from being rounded twice.
    mjd_tt = np.asarray(np.longdouble(mjdref) + np.longdouble(met) / 86400, dtype=np.float64)

    table = Table(
        [mjd_tt * u.d, met * u.s, pos[:, 0], pos[:, 1], pos[:, 2], vel[:, 0], vel[:, 1], vel[:, 2]],
        names=("MJD_TT", "MET", "X", "Y", "Z", "Vx", "Vy", "Vz"),
        meta={"name": "orbit", "mjdref": mjdref, "telescope": telescope},
    )
    logger.info(
        f"Read {len(table)} orbit rows from {fname}, MJD {mjd_tt.min():.6f} to {mjd_tt.max():.6f}"
    )
    return table


def _has_columns(data, names):
    names = (names,) if isinstance(names, str) else names
    return all(n in data.names for n in names)


def _clean(table):
    """Sort by time and drop the rows a spline cannot use.

    Orbit files routinely contain rows out of order, rows repeated at the same time
    (especially where two files overlap) and all-zero placeholder rows. A spline
    through a repeated abscissa is undefined and a zero position is a 6400 km error, so
    all three have to go. PINT does this only for FPorbit files; every mission needs it.
    """
    table.sort("MJD_TT")

    keep = np.ones(len(table), dtype=bool)
    if len(table) > 1:
        keep[1:] = np.diff(table["MJD_TT"].value) > 0
    n_dup = np.count_nonzero(~keep)
    if n_dup:
        logger.warning(f"Dropping {n_dup} duplicate orbit rows")

    zero = (table["X"].value == 0) & (table["Y"].value == 0) & (table["Z"].value == 0)
    if np.any(zero):
        logger.warning(f"Dropping {np.count_nonzero(zero)} all-zero orbit rows")
    return table[keep & ~zero]


def read_orbit(orbit_files, spec=None):
    """Read one or more orbit files into a single cleaned table.

    Parameters
    ----------
    orbit_files : str or iterable of str
        One file name, an iterable of them, or ``"@metafile"`` naming a text file that
        lists them one per line. Names may be local paths, ``https://`` or ``s3://``
        URLs. Several files are stacked and sorted, so a stack of daily orbit files can
        be passed straight through.
    spec : OrbitSpec, optional
        Override the specification. By default it is looked up from the ``TELESCOP``
        keyword of the first file.

    Returns
    -------
    astropy.table.Table
        Columns ``MJD_TT`` (days), ``MET`` (seconds), ``X``, ``Y``, ``Z`` (metres) and
        ``Vx``, ``Vy``, ``Vz`` (metres per second), with ``mjdref`` and ``telescope``
        in ``meta``.
    """
    if isinstance(orbit_files, str) and orbit_files.startswith("@"):
        # Blank lines are skipped: a trailing newline would otherwise become "".
        with open(orbit_files[1:]) as metafile:
            orbit_files = [line.strip() for line in metafile if line.strip()]

    if isinstance(orbit_files, str) or not isinstance(orbit_files, Iterable):
        return _clean(_read_one(orbit_files, spec=spec))

    orbit_files = list(orbit_files)
    if len(orbit_files) == 1:
        return _clean(_read_one(orbit_files[0], spec=spec))

    tables = [_read_one(fname, spec=spec) for fname in orbit_files]
    stacked = vstack(tables, metadata_conflicts="silent")
    stacked.meta.update(tables[0].meta)
    return _clean(stacked)
