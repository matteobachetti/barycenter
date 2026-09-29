"""The barycentric correction, computed directly from astropy, ERFA and a JPL ephemeris.

This is an alternative to the PINT engine in :mod:`barycenter.barycenter`. It exists for
three reasons:

* **It is auditable.** Each term below is written out explicitly, so "does ``axBary``
  include this?" is a switch we set on purpose rather than a side effect of which
  components a PINT timing model happened to load.
* **It is precise everywhere.** Every quantity computed here is a small number of
  seconds, so float64 gives about 1e-13 s. The PINT engine subtracts two absolute MJDs,
  which needs an 80-bit ``longdouble`` to reach 100 ns, and there is no such thing on
  Apple Silicon or Windows.
* **It is fast.** No TOA objects, no observatory registry, no monkey patching.

The correction
--------------

Given a photon recorded at spacecraft time ``t`` (TT), the time it would have had at
the solar-system barycentre (TDB) is ``t`` plus

.. math::

    \\Delta(t) = \\underbrace{(\\mathrm{TDB}-\\mathrm{TT})_\\oplus}_\\text{Einstein}
               + \\underbrace{\\frac{\\vec r_\\mathrm{sc}\\cdot\\vec v_\\oplus}{c^2}}_\\text{topocentric Einstein}
               + \\underbrace{\\frac{\\vec r_\\mathrm{obs}\\cdot\\hat n}{c}}_\\text{Roemer}
               + \\underbrace{2T_\\odot\\ln(1+\\cos\\theta)}_\\text{Shapiro}
               + \\underbrace{\\text{parallax}}_\\text{only if a distance is given}

with :math:`\\vec r_\\mathrm{obs} = \\vec r_\\oplus + \\vec r_\\mathrm{sc}` the observer's
barycentric position, :math:`\\hat n` the unit vector towards the source, and
:math:`\\theta` the Sun-centred angle between the observer and the source.

Typical sizes for a low-Earth-orbit mission: Einstein 1.7 ms, topocentric Einstein
2.3 us, Roemer +-500 s, Shapiro a few tens of us.

Conventions
-----------

The Shapiro term is written the way ``axBary`` writes it, as
:math:`2T_\\odot\\ln(1+\\cos\\theta)`. PINT instead uses
:math:`2T_\\odot\\ln(r_\\odot(1+\\cos\\theta)/\\mathrm{AU})`, which differs by
:math:`2T_\\odot\\ln(r_\\odot/\\mathrm{AU})` -- an annual term of about 170 ns amplitude.
Neither is wrong: the Shapiro delay is only defined up to a constant, and the difference
is absorbed by a pulsar's spin parameters. But it has to be matched when comparing
against a tool.
"""

import astropy.units as u
import erfa
import numpy as np
from astropy.constants import GM_sun, au, c
from astropy.coordinates import (
    SkyCoord,
    get_body_barycentric,
    get_body_barycentric_posvel,
    solar_system_ephemeris,
)
from astropy.time import Time
from scipy.interpolate import CubicHermiteSpline, CubicSpline

#: Half the Sun's Schwarzschild radius crossing time, GM_sun / c**3, in seconds.
#: 4.925490947e-06 s.
T_SUN = float((GM_sun / c**3).to_value(u.s))

#: Speed of light in m/s and the astronomical unit in m, as plain floats: everything
#: below works on bare numpy arrays, because Quantity arithmetic on a million events is
#: slow and buys nothing here.
C_M_S = float(c.to_value(u.m / u.s))
AU_M = float(au.to_value(u.m))


def spacecraft_interpolator(met, position, velocity=None):
    """Interpolate a tabulated spacecraft orbit.

    Parameters
    ----------
    met : array-like, shape (N,)
        Times of the orbit samples, in mission elapsed seconds.
    position : array-like, shape (N, 3)
        Geocentric position in metres, in the same frame as the ephemeris.
    velocity : array-like, shape (N, 3), optional
        Geocentric velocity in m/s. When present, a cubic Hermite spline is used,
        which matches both the position and its slope at every sample; without it a
        plain cubic spline is fitted through the positions.

    Returns
    -------
    fun : callable
        ``fun(met)`` gives the position in metres, shape ``(len(met), 3)``.

    Notes
    -----
    Rows out of time order, repeated times and all-zero rows are dropped: orbit files
    routinely contain all three, and a spline through a repeated abscissa blows up.
    """
    met = np.asarray(met, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)

    order = np.argsort(met, kind="stable")
    met, position = met[order], position[order]
    keep = np.concatenate([[True], np.diff(met) > 0])
    keep &= np.any(position != 0, axis=1)
    if velocity is None:
        return CubicSpline(met[keep], position[keep], axis=0)

    velocity = np.asarray(velocity, dtype=np.float64)[order]
    return CubicHermiteSpline(met[keep], position[keep], velocity[keep], axis=0)


def ephemeris_frame(ephem):
    """The reference frame a JPL ephemeris is expressed in.

    DE200 is referred to the FK5 dynamical equinox of J2000; DE405 and everything
    after it are aligned with the ICRF. The two differ by about 20 mas, which is
    nothing on a spacecraft position but 15 km on the Earth's barycentric position --
    some 45 us of Roemer delay. Getting this wrong is the largest single mistake
    available here.
    """
    name = str(ephem).lower()
    return "fk5" if "de200" in name or "de118" in name else "icrs"


def source_unit_vector(ra_deg, dec_deg, frame="icrs", ephem_frame="icrs"):
    """Unit vector towards the source, in the frame the ephemeris uses.

    The positions a JPL kernel returns are the numbers in the kernel: reading it
    applies no rotation, whatever frame astropy labels the result with. So the source
    direction has to be brought into the *kernel's* frame, not into ICRS. Pairing
    DE200 with FK5 coordinates, or DE440 with ICRS coordinates, therefore needs no
    rotation at all -- which is what HEASOFT's ``refframe`` parameter is really saying.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Source coordinates in degrees, in ``frame``.
    frame : str, optional
        The frame those coordinates are in, typically the ``RADECSYS`` keyword.
    ephem_frame : str, optional
        The frame of the ephemeris, from :func:`ephemeris_frame`.

    Returns
    -------
    n_hat : ndarray, shape (3,)
    """
    frame, ephem_frame = frame.lower(), ephem_frame.lower()
    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame=frame)
    if frame != ephem_frame:
        coord = coord.transform_to(ephem_frame)
    return coord.cartesian.xyz.value.astype(np.float64)


def met_to_time(met, mjdref):
    """Build an astropy ``Time`` (TT) from mission elapsed time.

    The integer and fractional parts of the reference epoch are kept apart so that
    ``Time``'s internal two-double representation carries the full precision. It hardly
    matters -- the correction is a smooth function of time, so even a microsecond of
    error in the argument moves the answer by well under a nanosecond -- but it costs
    nothing.
    """
    mjdref = np.longdouble(mjdref)
    day = np.float64(np.floor(mjdref))
    frac = np.float64(mjdref - np.longdouble(day))
    return Time(day, frac + np.asarray(met, dtype=np.float64) / 86400.0, format="mjd", scale="tt")


def barycentric_correction(
    met,
    mjdref,
    ra_deg,
    dec_deg,
    sc_position,
    sc_velocity=None,
    ephem="de440",
    frame="icrs",
    distance_kpc=None,
    shapiro="axbary",
):
    """Barycentric correction, in seconds, for an array of mission elapsed times.

    Parameters
    ----------
    met : array-like
        Mission elapsed times, in seconds, on the TT scale. Any clock correction must
        already have been applied (see the note in
        :func:`barycenter.barycenter.correct_times` about the order).
    mjdref : float
        The mission's reference epoch, MJD(TT). Pass it at full precision, e.g. as
        ``MJDREFI + MJDREFF``.
    ra_deg, dec_deg : float
        Source coordinates in degrees.
    sc_position : callable or array-like, shape (N, 3)
        The spacecraft's geocentric position in metres, either as an array evaluated at
        ``met`` or as a callable taking ``met``.
    sc_velocity : callable or array-like, shape (N, 3), optional
        The spacecraft's geocentric velocity in m/s. Only used for the topocentric
        Einstein term, where it contributes a few picoseconds; leaving it out is fine.
    ephem : str, optional
        JPL ephemeris, e.g. ``de440``, ``de430``, ``de200``.
    frame : str, optional
        Frame of ``ra_deg``/``dec_deg``, typically the ``RADECSYS`` keyword. The
        direction is rotated into the ephemeris's own frame; see
        :func:`source_unit_vector`.
    distance_kpc : float, optional
        Source distance. Without it the parallax term is left out, which is what the
        official tools do.
    shapiro : {"axbary", "pint", "none"}, optional
        Which convention to use for the solar-system Shapiro delay. See the module
        docstring.

    Returns
    -------
    correction : ndarray
        Seconds to add to ``met`` to get the barycentric arrival time, as a MET on the
        TDB scale.
    """
    met = np.atleast_1d(np.asarray(met, dtype=np.float64))
    time = met_to_time(met, mjdref)

    pos_sc = np.atleast_2d(sc_position(met) if callable(sc_position) else sc_position)
    pos_sc = np.asarray(pos_sc, dtype=np.float64)

    n_hat = source_unit_vector(ra_deg, dec_deg, frame=frame, ephem_frame=ephemeris_frame(ephem))

    with solar_system_ephemeris.set(ephem):
        earth_pos, earth_vel = get_body_barycentric_posvel("earth", time)
        sun_pos = get_body_barycentric("sun", time)

    # Metres and m/s, shape (N, 3). Both are ICRS-aligned, as the orbit files are.
    r_earth = earth_pos.xyz.to_value(u.m).T
    v_earth = earth_vel.xyz.to_value(u.m / u.s).T
    r_sun = sun_pos.xyz.to_value(u.m).T

    r_obs = r_earth + pos_sc

    # -- Einstein: the geocentric (TDB - TT) series, as ERFA's dtdb ------------------
    # Note this cannot be written as ``time.tdb - time.tt``: those are the same instant
    # in two scales, so astropy aligns them and the difference is identically zero.
    # The last three arguments are the observer's distance from the Earth's spin axis
    # and from the equatorial plane; zero means the geocentre, and the observer's own
    # offset is the separate topocentric term below.
    einstein = erfa.dtdb(time.jd1, time.jd2, 0.0, 0.0, 0.0, 0.0)

    # -- topocentric Einstein: the observer is not at the geocentre -------------------
    # About 2.3 us for a low Earth orbit, and it varies over the orbit, so it is not
    # something a constant offset can absorb.
    topo_einstein = np.einsum("ij,ij->i", pos_sc, v_earth) / C_M_S**2

    # -- Roemer: light travel time from the observer to the barycentre ----------------
    roemer = r_obs @ n_hat / C_M_S

    # -- Shapiro: the Sun's gravity delays the signal ---------------------------------
    if shapiro == "none":
        shapiro_term = np.zeros_like(met)
    else:
        u_vec = r_obs - r_sun  # Sun -> observer
        r_o = np.linalg.norm(u_vec, axis=1)
        cos_theta = (u_vec @ n_hat) / r_o
        if shapiro == "axbary":
            shapiro_term = 2 * T_SUN * np.log1p(cos_theta)
        elif shapiro == "pint":
            shapiro_term = 2 * T_SUN * np.log(r_o * (1 + cos_theta) / AU_M)
        else:
            raise ValueError(f"Unknown shapiro convention: {shapiro}")

    correction = einstein + topo_einstein + roemer + shapiro_term

    # -- parallax: the wavefront is a sphere, not a plane, for a nearby source --------
    if distance_kpc is not None:
        d = distance_kpc * 1000.0 * u.pc.to(u.m)
        r_perp_sq = np.sum(r_obs**2, axis=1) - (r_obs @ n_hat) ** 2
        correction = correction - r_perp_sq / (2 * C_M_S * d)

    return correction
