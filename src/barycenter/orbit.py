"""Reading spacecraft orbit files.

Every mission tabulates the same thing -- time, geocentric position, geocentric
velocity -- and differs only in which extension it lives in, what the columns are
called, whether the position is one vector column or three scalars, and whether the
units are metres or kilometres. So there is one reader, driven by an
:class:`OrbitSpec`, and adding a mission is one entry in
:data:`barycenter.missions.MISSIONS`.

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

from .utils import column_named, fits_open_including_remote

__all__ = ["OrbitCoverage", "OrbitSpec", "read_orbit"]


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


def _columns(data, names, unit):
    """Pull an (N, 3) array out of either one vector column or three scalar ones."""
    if isinstance(names, str):
        vec = np.asarray(data.field(names), dtype=np.float64)
        if vec.ndim != 2 or vec.shape[1] != 3:
            raise ValueError(f"Column {names} has shape {vec.shape}, expected (N, 3)")
    else:
        vec = np.column_stack([np.asarray(data.field(n), dtype=np.float64) for n in names])
    return vec * unit


def _spec_for_file(hdul, specs):
    """The spec to read an open orbit file with, out of the one or more a mission has.

    A mission with several kinds of orbit file under one ``TELESCOP`` lists one spec per
    kind, each naming its extension, and the file's extensions decide. A file with none
    of them falls back to the first spec, whose reader then reports what is missing.
    """
    if isinstance(specs, OrbitSpec):
        return specs
    for spec in specs:
        if spec.hdu in hdul:
            return spec
    return specs[0]


def _read_one(fname, spec=None):
    """Read a single orbit file into the common table."""
    from .utils import high_precision_mjdref

    with fits_open_including_remote(fname) as hdul:
        hdu = hdul[spec.hdu] if spec is not None else hdul[1]
        header, data = hdu.header, hdu.data
        telescope = header.get("TELESCOP", "unknown")
        if spec is None:
            # Imported here rather than at module level: the registry describes missions in
            # terms of OrbitSpec, so importing it the other way round would be circular.
            from .missions import mission_for

            mission = mission_for(telescope)
            if mission.orbit is None:
                raise ValueError(
                    f"{fname} is a {telescope} orbit file, and there is no native reader "
                    f"for {mission.name} yet. Use --apply-official, or add an OrbitSpec to "
                    "its entry in barycenter.missions.MISSIONS."
                )
            spec = _spec_for_file(hdul, mission.orbit)
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
    """Whether every one of ``names`` is a column of ``data``, ignoring case.

    Case matters here because the alternative to finding a velocity column is silently
    differentiating the position instead -- a quiet degradation rather than an error.
    Chandra spells its velocities ``Vx``, ``Vy``, ``Vz`` and its time ``Time``.
    """
    names = (names,) if isinstance(names, str) else names
    return all(column_named(data, n) is not None for n in names)


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
    spec : ~barycenter.orbit.OrbitSpec, optional
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


@dataclass(frozen=True)
class OrbitCoverage:
    """Where an orbit file actually knows the spacecraft position.

    Interpolating splines answer every time they are asked about, extrapolating past
    their last knot and coasting straight through an interior gap without complaint.
    That is the right behaviour for the sub-second shortfalls orbit files routinely
    have, and the wrong one for a file that is missing half an orbit: the answer is
    then a guess presented as a measurement. This class is what lets a caller tell the
    two apart.

    Parameters
    ----------
    samples : ndarray
        Sorted times of the tabulated positions, in mission elapsed seconds.
    cadence : float
        The file's own sampling interval.
    filled : ndarray, shape (n, 2), optional
        Gaps inside the file, as (start, end) pairs, across which a position is supplied
        by a fitted orbit (see :mod:`barycenter.gapfill`). Times inside them count as
        covered: the caller has chosen to accept that position.

    Notes
    -----
    A tabulated position is taken to describe the spacecraft for one sampling interval
    either side of itself, because that is exactly what interpolating between two
    samples already assumes. Without that allowance a mission which tabulates its
    position every 30 s, as Fermi does, would have a third of its perfectly good events
    counted as uncovered purely for sitting between two samples.

    Examples
    --------
    >>> import numpy as np
    >>> cov = OrbitCoverage.from_met(np.arange(0.0, 100.0, 10.0))
    >>> float(cov.cadence)
    10.0
    >>> cov.uncovered([45.0])            # between two samples: covered
    array([0.])
    >>> cov.uncovered([200.0])           # 110 s past the last sample, less the allowance
    array([100.])
    """

    samples: np.ndarray
    cadence: float
    filled: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    @classmethod
    def from_met(cls, met, filled=None):
        """Build the coverage of a set of sample times.

        The cadence is the median spacing rather than the mean: an orbit file with a
        few long gaps in it still has a well defined normal sampling interval, and the
        median is what reports it.
        """
        met = np.unique(np.asarray(met, dtype=np.float64))
        steps = np.diff(met)
        cadence = float(np.median(steps)) if len(steps) else 0.0
        filled = np.empty((0, 2)) if filled is None else np.asarray(filled, dtype=np.float64)
        return cls(samples=met, cadence=cadence, filled=filled)

    @classmethod
    def from_table(cls, table):
        """Build the coverage of a table from :func:`read_orbit`."""
        return cls.from_met(np.asarray(table["MET"].value, dtype=np.float64))

    def uncovered(self, times):
        """Seconds by which each time falls outside the tabulated positions.

        Zero where the orbit file has something to say, whether the time sits between
        two samples or within one sampling interval of the ends. Positive where it does
        not: past either end, or far enough into an interior gap that the position
        there is an extrapolation rather than an interpolation. The value is how far
        the time reaches beyond what the file covers, so it can be compared against a
        tolerance in seconds.
        """
        times = np.asarray(times, dtype=np.float64)
        if len(self.samples) == 0:
            return np.full(times.shape, np.inf)
        if len(self.samples) == 1:
            nearest = np.abs(times - self.samples[0])
        else:
            right = np.clip(np.searchsorted(self.samples, times), 1, len(self.samples) - 1)
            nearest = np.minimum(
                np.abs(self.samples[right] - times), np.abs(self.samples[right - 1] - times)
            )
        missing = np.maximum(nearest - self.cadence, 0.0)
        for start, end in self.filled:
            missing = np.where((times > start) & (times < end), 0.0, missing)
        return missing
