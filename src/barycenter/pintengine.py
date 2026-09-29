"""The PINT barycentring engine.

PINT is optional. It is kept for two reasons: reading a full ``.par`` file when the
source needs proper astrometry (proper motion, parallax), and as a second, independent
implementation to cross-check the native engine against. The default engine is
:mod:`barycenter.native`.

Why this module exists at all
-----------------------------

PINT's ``SatelliteObs`` takes a *filename* and calls its own ``load_orbit`` on it. That
is not usable here: we need to read missions PINT has no loader for (SVOM), read from
``https://`` and ``s3://`` URLs, accept a list of orbit files, and clean every mission's
table rather than only FPorbit's.

The package used to get all of that by monkey-patching
``pint.observatory.satellite_obs.load_orbit`` and ``SatelliteObs._check_bounds`` at
import time. That worked, but it changed PINT's behaviour for every other piece of code
in the same interpreter -- including silently disabling PINT's own extrapolation guard --
and it drifted: PINT has since grown its own NuSTAR loader, so part of the patch had
become dead weight without anyone noticing.

:class:`TableSatelliteObs` does the same job by subclassing instead. It takes an
already-parsed table from :mod:`barycenter.orbit`, so the orbit file is read by the same
code the native engine uses, and nothing outside this class is modified. Verified to
give bit-identical positions and velocities to the monkey-patched path.

It still depends on PINT's internal attribute names (``FT2``, ``X``..``Vz``,
``_geocenter``, ``_maxextrap``). That is a narrower coupling than a monkey patch, not no
coupling: if PINT renames them this class breaks -- but it breaks visibly, here, and
only for the optional engine.
"""

import logging as logger

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from scipy.interpolate import Akima1DInterpolator, InterpolatedUnivariateSpline

__all__ = ["TableSatelliteObs", "pint_barycentric_correction"]


def _satellite_obs_bases():
    """Import PINT lazily, with a message that says what to install."""
    try:
        from pint.observatory.satellite_obs import SatelliteObs
        from pint.observatory.special_locations import SpecialLocation
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The PINT engine needs pint-pulsar. Install it, or use the default "
            "native engine, which has no such dependency."
        ) from exc
    return SatelliteObs, SpecialLocation


def TableSatelliteObs(name, table, maxextrap=2, apply_gps2utc=False, overwrite=True):
    """Register a PINT satellite observatory from an orbit table.

    A factory rather than a plain class, because PINT has to be imported before the
    base class exists, and importing PINT is optional here.

    Parameters
    ----------
    name : str
        Observatory name to register. PINT looks observatories up by name, and
        satellite observatories cannot have aliases, so this has to match what the TOAs
        will say.
    table : astropy.table.Table
        From :func:`barycenter.orbit.read_orbit`: columns ``MJD_TT`` and
        ``X``..``Vz``.
    maxextrap : float
        Minutes of extrapolation beyond the orbit table to tolerate before complaining.
    apply_gps2utc : bool
        Whether to apply the UTC(GPS) to UTC correction. Measured effect on NuSTAR:
        about 0.1 ns, so the default is off, matching what the reference tools do.
    overwrite : bool
        Replace an existing registry entry of the same name.

    Returns
    -------
    pint.observatory.satellite_obs.SatelliteObs
    """
    SatelliteObs, SpecialLocation = _satellite_obs_bases()

    class _TableSatelliteObs(SatelliteObs):
        """A SatelliteObs built from a table instead of a file name."""

        def __init__(self, name, table, maxextrap, apply_gps2utc, overwrite):
            # Deliberately skipping SatelliteObs.__init__, whose only extra job is to
            # call PINT's load_orbit and build the splines; we supply the table and do
            # the same six splines ourselves.
            SpecialLocation.__init__(self, name, apply_gps2utc=apply_gps2utc, overwrite=overwrite)
            self.FT2 = table
            mjd_tt = table["MJD_TT"]
            for column in ("X", "Y", "Z", "Vx", "Vy", "Vz"):
                setattr(
                    self,
                    column,
                    InterpolatedUnivariateSpline(mjd_tt, table[column], ext="extrapolate"),
                )
            self._geocenter = EarthLocation.from_geocentric(0.0 * u.m, 0.0 * u.m, 0.0 * u.m)
            self._maxextrap = maxextrap

        def _check_bounds(self, t):
            """Warn about extrapolation beyond the orbit table instead of raising.

            PINT raises ``ValueError``. In practice an orbit file routinely stops a
            fraction of a second short of the last event, and refusing to process the
            file over a sub-second extrapolation of a smooth orbit is not useful. A gap
            five times longer than the allowance is a different matter -- that means
            the orbit file is missing a chunk and the positions there are guesses -- so
            it is reported as an error.
            """
            orbit_mjd = np.asarray(self.FT2["MJD_TT"])
            asked = np.atleast_1d(t.tt.mjd)
            right = np.clip(np.searchsorted(orbit_mjd, asked), 1, len(orbit_mjd) - 1)
            gap = np.minimum(np.abs(orbit_mjd[right] - asked), np.abs(orbit_mjd[right - 1] - asked))
            bad = gap > self._maxextrap / 1440.0
            if not np.any(bad):
                logger.debug(
                    f"All {len(asked)} times are within {self._maxextrap} minutes of a "
                    "tabulated spacecraft position."
                )
                return
            report = logger.error if np.any(gap > 5 * self._maxextrap / 1440.0) else logger.warning
            report(
                f"Extrapolating the spacecraft position by more than "
                f"{self._maxextrap} minutes for {np.count_nonzero(bad)} times, between "
                f"MJD {orbit_mjd[right - 1][bad].min()} and {orbit_mjd[right][bad].max()}."
            )

    return _TableSatelliteObs(name, table, maxextrap, apply_gps2utc, overwrite)


def pint_barycentric_correction(orbit_table, model, mjdref=None, dt=5.0, met_range=None):
    """Barycentric correction from PINT, as an interpolator over mission elapsed time.

    Parameters
    ----------
    orbit_table : astropy.table.Table
        From :func:`barycenter.orbit.read_orbit`.
    model : pint.models.TimingModel
        With ``RAJ``, ``DECJ`` and ``EPHEM`` set.
    mjdref : float, optional
        Reference epoch. Taken from the orbit table's metadata if omitted.
    dt : float
        Grid spacing in seconds. Measured cost of a 5 s grid: 1.2 ns.
    met_range : tuple of float, optional
        ``(start, stop)`` to restrict the grid to, in MET. Without it the grid spans the
        whole orbit file, which for a multi-day orbit file means tens of thousands of
        TOAs that are never used.

    Returns
    -------
    callable
        ``fun(met)`` gives the correction in seconds.
    """
    import pint.toa as toa

    if mjdref is None:
        mjdref = orbit_table.meta["mjdref"]

    telescope = str(orbit_table.meta["telescope"]).lower()
    TableSatelliteObs(telescope, orbit_table, overwrite=True)

    met = np.asarray(orbit_table["MET"].value, dtype=np.float64)
    start, stop = met.min(), met.max()
    if met_range is not None:
        # One grid step of margin either side, so the events are interpolated and not
        # extrapolated.
        start = max(start, met_range[0] - dt)
        stop = min(stop, met_range[1] + dt)

    grid = np.arange(start, stop + dt, dt)
    mjds = np.asarray(np.longdouble(mjdref) + np.longdouble(grid) / 86400, dtype=np.float64)

    toas = [toa.TOA(mjd, obs=telescope, scale="tt") for mjd in mjds]
    ts = toa.get_TOAs_list(
        toas,
        ephem=model.EPHEM.value,
        include_bipm=False,
        planets="PLANET_SHAPIRO" in model.params and model.PLANET_SHAPIRO.value,
        tdb_method="default",
    )
    bats = model.get_barycentric_toas(ts)
    correction = np.asarray((bats.to_value(u.d) - mjds) * 86400, dtype=np.float64)

    return Akima1DInterpolator(grid, correction, extrapolate=True)
