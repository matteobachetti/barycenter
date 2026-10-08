"""The mission-agnostic correction: read a file, correct every time in it, write it out.

This is the pure-Python workflow. It reads the spacecraft orbit through
:mod:`barycenter.orbit`, gets a correction function from one of the engines, optionally
adds a clock correction from :mod:`barycenter.clock`, and applies both to every time
column and time keyword in every extension.
"""

import logging as logger
import os
import textwrap
import tempfile
from functools import partial

import astropy.units as u
import numpy as np
from astropy.time import Time

from ._version import __version__
from .clock import clock_correction_fun
from .native import (
    coordinates_in_ephemeris_frame,
    met_to_time,
    native_barycentric_correction,
)
from .official import apply_mission_specific_barycenter_correction
from .gapfill import MIN_GAP_S, find_gaps
from .orbit import OrbitCoverage, read_orbit
from .remote import download_locally
from .utils import (
    column_named,
    fits_open_including_remote,
    high_precision_mjdref,
    leap_seconds_since_mjdref,
    slim_down_hdu_list,
)

#: The engines that can compute the correction. ``native`` is the default: it is exact
#: in float64 on every platform, about 30 times faster, and matches HEASOFT ``barycorr``
#: to the resolution the reference files can express. ``pint`` is kept as an independent
#: second opinion and for timing models with proper motion.
ENGINES = ("native", "pint")

__all__ = [
    "ABSORBED_KEYWORDS",
    "AUTO_GRID_DT_S",
    "AUTO_GRID_EVENTS",
    "COORDINATE_KEYWORDS",
    "DERIVED_KEYWORDS",
    "ENGINES",
    "HEASOFT_COORDINATE_KEYWORDS",
    "MET_RANGE_PAD_S",
    "TIME_COLUMNS",
    "apply_barycenter_correction",
    "correct_times",
    "extract_events_in_region",
    "get_barycentric_correction",
    "get_coordinates_from_fits_header",
    "get_dummy_parfile_for_position",
    "grid_spacing_for",
    "met_range_for_file",
    "pre_bary_shift",
    "update_derived_keywords",
    "COVERAGE_TOLERANCE_S",
    "clamp_uncovered_keyword",
    "enforce_orbit_coverage",
    "gti_intervals",
    "good_time_span",
    "inside_gti",
]


#: Every column holding a time that a barycentric correction must move. Matched against
#: a file's columns without regard to case; see :func:`barycenter.utils.column_named`.
TIME_COLUMNS = ("TIME", "START", "STOP", "TSTART", "TSTOP")

#: Above this many events the native engine switches from evaluating the correction at
#: every event to interpolating it on a grid. Measured on the committed NuSTAR orbit
#: file: at a million events the exact path takes 6.72 s and the grid 0.14 s, a factor
#: of 48, for a maximum error of 1.6 ns. Below the threshold the exact path costs well
#: under a second, so there is nothing to buy and the answer stays exact -- which keeps
#: every reference comparison in the test suite on the unapproximated code.
AUTO_GRID_EVENTS = 100_000

#: The grid spacing used when the threshold above is crossed. At 5 s the interpolation
#: error over the committed orbit file is mean +0.002 ns, std 0.39 ns, max 1.6 ns,
#: against a 100 ns target and reference files whose own float64 granularity is 30 ns
#: (NuSTAR) to 119 ns (RXTE).
AUTO_GRID_DT_S = 5.0

#: How far outside the file's own ``TSTART``/``TSTOP`` a clipped grid is extended.
#: The range is taken from the headers rather than from the data, because reading a
#: strided time column out of a memory-mapped table pages in the whole file; the pad
#: covers the gap between what a header claims and what its extensions hold. All five
#: committed reference files keep their times inside their own TSTART/TSTOP, the widest
#: margin being Chandra's 823 s of trailing slack, and a file that does not is caught by
#: the coverage check in :func:`apply_barycenter_correction`.
MET_RANGE_PAD_S = 1000.0


def grid_spacing_for(n_events, dt=None):
    """The grid spacing to use, or ``None`` for the exact per-event path.

    Parameters
    ----------
    n_events : int
        The largest number of rows in any time column in the file.

    Other Parameters
    ----------------
    dt : float, optional
        What the caller asked for. ``None`` decides from ``n_events``; a positive value
        is used as given; zero or negative forces the exact path whatever the size.

    Returns
    -------
    float or None

    Examples
    --------
    >>> grid_spacing_for(1000) is None
    True
    >>> grid_spacing_for(10_000_000)
    5.0
    >>> grid_spacing_for(10_000_000, dt=0)   # the escape hatch
    >>> grid_spacing_for(1000, dt=2.5)
    2.5
    """
    if dt is not None:
        return float(dt) if float(dt) > 0 else None
    return AUTO_GRID_DT_S if n_events > AUTO_GRID_EVENTS else None


def met_range_for_file(hdul, timezero=0.0, clock_fun=None, leap_fun=None, pad=MET_RANGE_PAD_S):
    """The span of times the barycentric correction will actually be asked for.

    The widest ``TSTART``/``TSTOP`` over every extension, padded, and then shifted by the
    clock and leap-second terms -- because the barycentric correction is evaluated at the
    clock-corrected time, not the raw one (see :func:`correct_times`). On Swift that shift
    is nearly 20 s, so ignoring it would put the events outside a grid clipped to the raw
    span.

    Returns
    -------
    tuple of float or None
        ``None`` when no extension carries either keyword, in which case the caller must
        not clip: there is nothing to clip to.
    """
    lo, hi = np.inf, -np.inf
    for hdu in hdul:
        for keyname in ("TSTART", "TSTOP"):
            value = hdu.header.get(keyname)
            if value is not None and np.isfinite(float(value)):
                lo, hi = min(lo, float(value)), max(hi, float(value))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None

    ends = np.array([lo - pad, hi + pad], dtype=np.float64)
    shifted = pre_bary_shift(ends, timezero, clock_fun, leap_fun)
    return float(np.min(shifted)), float(np.max(shifted))


def pre_bary_shift(times, timezero=0.0, clock_fun=None, leap_fun=None):
    """The times at which the barycentric correction gets evaluated.

    Everything :func:`correct_times` does *before* it calls ``bary_fun``: fold in
    ``TIMEZERO``, then add the leap-second and clock terms, both evaluated on the raw
    times. Factored out because two callers need it -- the one that decides how wide an
    interpolation grid has to be, and the one that checks afterwards that the times fell
    inside it.
    """
    raw = np.asarray(times, dtype=np.float64) + timezero
    shifted = raw
    if leap_fun is not None:
        shifted = shifted + leap_fun(raw)
    if clock_fun is not None:
        shifted = shifted + clock_fun(raw)
    return shifted


#: Keywords describing a correction that the output has now absorbed, and which cannot
#: be rewritten because there is nothing left for them to describe. They are removed.
#: ``UTCFINIT`` is "the UTC correction factor at TSTART": after barycentring TSTART has
#: moved, ``TIMESYS`` is TDB, and the factor itself has been folded into every time, so
#: anyone applying it again would move a Swift event a further 15.56 s. HEASOFT
#: ``barycorr`` deletes it from every extension for the same reason (v1.7, "remove the
#: UTCFINIT keyword since leap seconds have now been adjusted for"), and does so whether
#: or not a clock file was used.
ABSORBED_KEYWORDS = ("UTCFINIT",)

#: Keywords that are not times themselves but are *computed* from ``TSTART`` and
#: ``TSTOP``, and so become wrong the moment those move. No official tool rewrites all
#: four and each rewrites a different subset, so this list is not copied from any of
#: them; what they agree on is that a file must not leave claiming a duration or a
#: calendar date that its own corrected ``TSTART`` and ``TSTOP`` contradict. See
#: :func:`update_derived_keywords`.
DERIVED_KEYWORDS = ("TELAPSE", "DATE-OBS", "DATE-END", "MJD-OBS")

#: Our order of preference for the source position, when it is not given explicitly.
#: ``RA_OBJ`` is the position of the object the observation was aimed at, which is the
#: thing a barycentric correction is actually about; the others describe where the
#: spacecraft was pointing, which is the same thing only to within the pointing
#: accuracy. This deliberately differs from HEASOFT -- see
#: :data:`HEASOFT_COORDINATE_KEYWORDS`.
COORDINATE_KEYWORDS = (
    ("RA_OBJ", "DEC_OBJ"),
    # Chandra's name for the same thing. It writes no RA_OBJ at all, and its RA_NOM can
    # sit several arcminutes away -- 324 arcsec on the ACIS test file, worth 0.8 s of
    # Roemer delay -- so leaving RA_TARG out is not a refinement but a blunder.
    ("RA_TARG", "DEC_TARG"),
    ("RA_NOM", "DEC_NOM"),
    ("RA_PNT", "DEC_PNT"),
    ("RA", "DEC"),
)

#: The order HEASOFT ``barycorr`` uses, read off its own ``kwfallback`` call (barycorr
#: 2.19, line 278). We do not follow it, but we keep it here so that
#: :func:`get_coordinates_from_fits_header` can say when the two would disagree, and by
#: how much: the two keywords routinely differ by a fraction of an arcsecond, and
#: 0.1 arcsec is 172 us of light travel time.
HEASOFT_COORDINATE_KEYWORDS = (
    ("RA_NOM", "DEC_NOM"),
    ("RA_PNT", "DEC_PNT"),
    ("RA_OBJ", "DEC_OBJ"),
    ("RA", "DEC"),
)

#: The largest Roemer delay any position error can produce: the Earth's orbit is about
#: 500 light seconds in radius, so an angular error of ``theta`` radians is worth at
#: most ``500 * theta`` seconds of delay.
_MAX_ROEMER_DELAY_S = 499.0


def _first_keyword_pair_present(hdr, chain):
    """The first (ra, dec) keyword pair in ``chain`` that the header actually has."""
    for ra_key, dec_key in chain:
        if ra_key in hdr and dec_key in hdr:
            return ra_key, dec_key
    return None


def _log_if_heasoft_would_disagree(hdr, chosen, heasoft):
    """Warn when HEASOFT would have used other keywords, quantified as a time delay."""
    from astropy.coordinates import angular_separation

    ours = [np.deg2rad(float(hdr[key])) for key in chosen]
    theirs = [np.deg2rad(float(hdr[key])) for key in heasoft]
    separation = angular_separation(ours[0], ours[1], theirs[0], theirs[1])
    delay = _MAX_ROEMER_DELAY_S * separation
    if delay < 1e-9:
        return
    logger.warning(
        f"Using {chosen[0]}/{chosen[1]} for the source position; HEASOFT barycorr would "
        f"have used {heasoft[0]}/{heasoft[1]}. They differ by "
        f"{np.rad2deg(separation) * 3600:.3f} arcsec, i.e. up to {delay * 1e6:.1f} us of "
        "Roemer delay. Pass --ra and --dec explicitly when the answer must match another "
        "tool."
    )


def get_coordinates_from_fits_header(hdr):
    """Get RA/Dec coordinate keywords from FITS header.

    The order of preference is :data:`COORDINATE_KEYWORDS`: ``RA_OBJ``/``DEC_OBJ``, then
    ``RA_NOM``/``DEC_NOM``, then ``RA_PNT``/``DEC_PNT``, then plain ``RA``/``DEC``.

    That is **not** HEASOFT's order, which is :data:`HEASOFT_COORDINATE_KEYWORDS` and
    starts from ``RA_NOM``. Preferring ``RA_OBJ`` is deliberate: it is the position of
    the target, while ``RA_NOM`` is where the spacecraft was aimed, and it is the target
    position that the arrival times should be referred to. When the two disagree by
    enough to matter this logs a warning saying so, because that is precisely when a
    comparison against an official tool will not match unless ``--ra``/``--dec`` are
    given explicitly.

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
    chosen = _first_keyword_pair_present(hdr, COORDINATE_KEYWORDS)
    if chosen is None:
        looked_for = ", ".join(f"{ra}/{dec}" for ra, dec in COORDINATE_KEYWORDS)
        raise ValueError(
            f"No coordinates found in header. Looked for {looked_for}. Pass --ra and "
            "--dec, or a .par file, instead."
        )

    heasoft = _first_keyword_pair_present(hdr, HEASOFT_COORDINATE_KEYWORDS)
    if heasoft is not None and heasoft != chosen:
        _log_if_heasoft_would_disagree(hdr, chosen, heasoft)

    return chosen


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
    fill_gaps=False,
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
        Frame of the coordinates. Both engines rotate the source position into the frame
        the ephemeris itself uses, which matters by 45 us between FK5 (DE200) and ICRS
        (DE405 and later). Ignored when ``model`` is given: a par file's astrometry is
        in the frame it was fitted in.
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
    fill_gaps : bool, optional
        Fit an orbit across the long gaps in the orbit file, rather than refusing the
        times that fall in them. Native engine only. See :mod:`barycenter.gapfill`.

    Returns
    -------
    bary_fun : callable
        ``bary_fun(met)`` gives the correction in seconds. Its ``coverage`` attribute is
        the :class:`~barycenter.orbit.OrbitCoverage` of the orbit file it was built from,
        so a caller can ask which of the times it is about to hand over are answered from
        a tabulated position and which from an extrapolation.
    """
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine {engine!r}. Choose one of {ENGINES}.")

    # read_orbit takes a file name, a list of them or an "@metafile", reads the
    # mission's columns from the MISSIONS registry, and hands back one cleaned
    # table. Both engines read the same table, so they cannot disagree about where the
    # spacecraft was.
    orbit_table = read_orbit(orbfile)

    met = np.asarray(orbit_table["MET"].value, dtype=np.float64)
    if met_range is not None:
        met_range = (max(met.min(), met_range[0]), min(met.max(), met_range[1]))

    if model is not None:
        ra, dec, ephem = position_from_model(model, ephem)

    # Carried on the returned callable rather than returned beside it, because every
    # caller wants the correction and only one wants the coverage -- and that one would
    # otherwise have to read the orbit file a second time to find out where it ends.
    if fill_gaps and engine != "native":
        raise ValueError("Filling orbit gaps is only implemented for the native engine")
    coverage = OrbitCoverage.from_met(met, filled=find_gaps(met, MIN_GAP_S) if fill_gaps else None)

    if engine == "native":
        if ra is None or dec is None:
            raise ValueError("The native engine needs ra and dec, or a timing model")
        bary_fun = native_barycentric_correction(
            orbit_table,
            ra,
            dec,
            ephem=ephem,
            frame=radecsys,
            dt=dt,
            met_range=met_range,
            fill_gaps=fill_gaps,
        )
        bary_fun.coverage = coverage
        return bary_fun

    from .pintengine import pint_barycentric_correction, timing_model_for_position

    if model is None:
        # PINT takes a position rather than a direction, labels it ICRS whatever it
        # really is, and -- like astropy -- applies no rotation when it reads a JPL
        # kernel. So ``radecsys`` has to be honoured here, by handing PINT coordinates
        # already expressed in the kernel's own frame; the native engine does the same
        # thing to its unit vector. Without this the flag reached only the output
        # header, and DE200 with FK5 coordinates came out 45 us wrong.
        ra, dec = coordinates_in_ephemeris_frame(ra, dec, frame=radecsys, ephem=ephem)
        model = timing_model_for_position(ra, dec, ephem)
    # A model read from a .par file is left alone: its astrometry is in the frame the
    # model was fitted in, and ``radecsys`` describes the event file's keywords, not it.
    bary_fun = pint_barycentric_correction(
        orbit_table, model, dt=5.0 if dt is None else dt, met_range=met_range
    )
    bary_fun.coverage = coverage
    return bary_fun


def _mission_counts_utc_seconds(telescope):
    """Whether this mission's MET counts UTC seconds, so leap seconds must be added.

    An unrecognised ``TELESCOP`` gets False rather than an error: the orbit reader will
    complain first if the mission really is unknown, and a file with a garbled keyword but
    usable columns should still be correctable. False is also the safe answer, because it
    is what every mission but Swift needs.
    """
    from .missions import mission_for

    try:
        return mission_for(telescope).met_is_utc
    except ValueError:
        return False


def update_derived_keywords(hdr, mjdref):
    """Rewrite the keywords computed from ``TSTART`` and ``TSTOP``, once those have moved.

    Called per extension *after* the times in that extension have been corrected, so
    everything is simply recomputed from the values now in the header. Only keywords the
    header already carried are written: adding a ``DATE-END`` to a file that never had
    one would be inventing metadata.

    Parameters
    ----------
    hdr : astropy.io.fits.Header
        An extension header whose ``TSTART``/``TSTOP`` have already been corrected.
    mjdref : numpy.longdouble or None
        The file's reference epoch. ``None`` leaves the date keywords alone --
        ``TELAPSE`` needs no epoch and is still updated.

    Notes
    -----
    ``TELAPSE`` is ``TSTOP - TSTART``, and it moves by as much as the corrections at the
    two ends differ: 3.4 s over the NuSTAR test observation, 1.6 s over the 7.6 h XMM
    one. SAS ``barycen`` updates it. HEASOFT ``barycorr`` does not, and its output is
    therefore 3.4 s self-inconsistent, which is a bug rather than a convention worth
    copying.

    ``DATE-OBS``, ``DATE-END`` and ``MJD-OBS`` are recomputed from ``MJDREF +
    TSTART/86400``, not shifted by however far ``TSTART`` moved. That is what ``barycorr``
    and CIAO ``axbary`` do -- checked against all five committed references -- and it is
    the self-consistent answer, because the output header says ``TIMESYS = TDB`` and the
    date should be the date of the time the file now records. It does change the
    convention on a mission that writes ``DATE-OBS`` in UTC while counting its MET in TT
    seconds: XMM by 63 s, Swift by 68 s, Fermi by 66.2 s, RXTE by 3.8 s -- in every case
    TT - UTC at the epoch. ``barycen`` shifts the string instead, and ``gtbary`` likewise
    keeps its dates in UTC, so both preserve that convention -- at the price of keeping
    ``DATE-OBS`` and ``TSTART`` as inconsistent with each other on the way out as they
    were on the way in, which is the worse of the two.

    ``ONTIME``, ``LIVETIME`` and ``EXPOSURE`` are deliberately left alone, as all three
    official tools leave them: they are sums of good-time interval lengths rather than
    differences between the file's ends, and the corrections at the two edges of one
    interval differ by microseconds.
    """
    tstart, tstop = hdr.get("TSTART"), hdr.get("TSTOP")

    if "TELAPSE" in hdr and tstart is not None and tstop is not None:
        hdr["TELAPSE"] = tstop - tstart

    if mjdref is None:
        return

    for keyword, met in (("DATE-OBS", tstart), ("DATE-END", tstop)):
        if keyword in hdr and met is not None:
            # isot rather than fits: the same string, without astropy ever being tempted
            # to append a time-scale suffix. met_to_time keeps MJDREF's integer and
            # fractional halves apart, so the rendered milliseconds are real.
            hdr[keyword] = met_to_time(met, mjdref).isot

    if "MJD-OBS" in hdr and tstart is not None:
        hdr["MJD-OBS"] = float(mjdref + np.longdouble(tstart) / 86400)


def _warn_outside_grid(bary_fun, times, where, timezero=0.0, clock_fun=None, leap_fun=None):
    """Say so if times fall outside an interpolated correction's grid.

    An interpolator extrapolates past its last knot rather than refusing, which is what
    we want for the sub-second shortfalls orbit files routinely have, but it means a file
    whose times lie well outside its own ``TSTART``/``TSTOP`` -- the span the grid was
    built from -- would be quietly answered from an extrapolation. The comparison is
    against the *shifted* times, since those are what ``bary_fun`` is asked for. The
    exact per-event path has no grid and so nothing to fall outside of.
    """
    grid = getattr(bary_fun, "x", None)
    if grid is None or len(times) == 0:
        return
    ends = pre_bary_shift([np.min(times), np.max(times)], timezero, clock_fun, leap_fun)
    short = max(grid[0] - np.min(ends), np.max(ends) - grid[-1])
    if short > 0:
        logger.warning(
            f"{where}: times fall up to {short:.3f} s outside the interpolation grid and "
            f"are extrapolated. Pass dt=0 to evaluate the correction at every event."
        )


#: How far a time may reach beyond the orbit file's coverage before it is refused.
#: Coverage already allows one sampling interval past the last position (see
#: :class:`~barycenter.orbit.OrbitCoverage`), so this is a further grace on top of that,
#: for the ragged ends real files have. A time inside a good time interval that misses by
#: more than this is an error rather than a warning: the position there is a guess, and a
#: guess silently written into an event list is indistinguishable from a measurement.
COVERAGE_TOLERANCE_S = 10.0

#: What to tell a user whose times fall in a hole in the middle of the orbit file.
FILL_HINT = (
    "Some of them lie in a gap inside the orbit file (for the Fermi LAT, the South "
    "Atlantic Anomaly): rerun with --fill-orbit-gaps to fit an orbit across it. That "
    "costs a position error of the order of 100 m (0.4 us of light time), up to about "
    "1 us in the worst case, against kilometres for the spline."
)


def gti_intervals(hdul):
    """The good time intervals of a file, raw and unsorted, or ``None`` if it has none.

    A GTI extension is one carrying ``START`` and ``STOP`` but no ``TIME``: that is what
    separates it from an event list, whose own start and stop are header keywords rather
    than columns. Several extensions are concatenated, because XMM writes one per CCD and
    an event is in a good time if it is in any of them.
    """
    starts, stops = [], []
    for hdu in hdul:
        data = getattr(hdu, "data", None)
        if data is None or column_named(data, "TIME") is not None:
            continue
        start, stop = column_named(data, "START"), column_named(data, "STOP")
        if start is None or stop is None:
            continue
        starts.append(np.asarray(data[start], dtype=np.float64))
        stops.append(np.asarray(data[stop], dtype=np.float64))
    if not starts:
        return None
    return np.concatenate(starts), np.concatenate(stops)


def inside_gti(times, gti):
    """Whether each time falls in one of the good time intervals.

    ``gti`` of ``None`` means the file declared none, in which case every time counts as
    good: a file that does not say which of its times are trustworthy is not thereby
    saying that none of them are.
    """
    times = np.asarray(times, dtype=np.float64)
    if gti is None:
        return np.ones(times.shape, dtype=bool)
    start, stop = gti
    order = np.argsort(start)
    start, stop = start[order], stop[order]
    previous = np.searchsorted(start, times, side="right") - 1
    return (previous >= 0) & (times <= stop[np.clip(previous, 0, None)])


def good_time_span(hdul):
    """The span to pull an out-of-range ``TSTART``/``TSTOP`` back to.

    The GTIs where there are any, and the event times otherwise -- both being statements
    about where the data actually is, as opposed to the requested range a ``TSTART``
    keyword often holds. ``None`` when the file offers neither.
    """
    gti = gti_intervals(hdul)
    if gti is not None and len(gti[0]):
        return float(np.min(gti[0])), float(np.max(gti[1]))
    times = [
        np.asarray(hdu.data[column], dtype=np.float64)
        for hdu in hdul
        if (column := column_named(getattr(hdu, "data", None), "TIME")) is not None
        and len(hdu.data)
    ]
    if not times:
        return None
    return float(min(t.min() for t in times)), float(max(t.max() for t in times))


def clamp_uncovered_keyword(
    value,
    keyname,
    coverage,
    span,
    timezero=0.0,
    clock_fun=None,
    leap_fun=None,
    tolerance=COVERAGE_TOLERANCE_S,
):
    """Pull a ``TSTART``/``TSTOP`` the orbit file cannot place back to where the data is.

    These two keywords routinely hold the range that was *asked* for rather than the one
    that was delivered -- a Fermi LAT extraction returns a ``TSTART`` set to the start of
    the requested window, while the GTIs and the spacecraft file begin whenever the data
    really does. Correcting such a keyword means extrapolating the spacecraft position to
    a time no observation covers, so it is moved to the edge of the good time intervals
    first, which is the earliest (or latest) moment the file actually describes.

    Only these two keywords are treated this way, and only when they miss by more than
    ``tolerance``; anything else is left exactly as it is.
    """
    if coverage is None or span is None or keyname not in ("TSTART", "TSTOP"):
        return value
    shifted = pre_bary_shift([value], timezero, clock_fun, leap_fun)
    missing = float(coverage.uncovered(shifted)[0])
    if missing <= tolerance:
        return value
    first = keyname == "TSTART"
    replacement = span[0] if first else span[1]
    logger.warning(
        f"{keyname}={value!r} is {missing:.3f} s outside the orbit file, which only covers "
        f"{coverage.samples[0]:.3f} to {coverage.samples[-1]:.3f}. It is the requested "
        f"range rather than the observed one, and barycentring it would extrapolate the "
        f"spacecraft position; moving it to {replacement!r}, where the data actually "
        f"{'starts' if first else 'ends'}."
    )
    return replacement


def enforce_orbit_coverage(
    hdul,
    coverage,
    timezero=0.0,
    clock_fun=None,
    leap_fun=None,
    tolerance=COVERAGE_TOLERANCE_S,
    fill_hint=True,
):
    """Drop rows the orbit file cannot place, and refuse the ones that matter.

    Times are tested where the correction is actually evaluated -- after ``TIMEZERO``,
    the leap seconds and the clock correction -- because that, and not the raw column, is
    what the orbit file is asked about.

    Rows outside every good time interval that the orbit file cannot place are dropped
    with a warning: they are junk the orbit file also happens not to cover, and keeping
    them would mean writing an extrapolated time into an event list. Rows *inside* a good
    time interval that miss by more than ``tolerance`` raise instead, because there is no
    honest thing to write for them and quietly dropping real events would be worse than
    refusing the file.

    Returns
    -------
    int
        How many rows were dropped.
    """
    if coverage is None:
        return 0

    gti = gti_intervals(hdul)
    dropped = 0
    for hdu in hdul:
        data = getattr(hdu, "data", None)
        for keyname in TIME_COLUMNS:
            column = column_named(data, keyname)
            if column is None or not len(data):
                continue
            raw = np.asarray(data[column], dtype=np.float64)
            shifted = pre_bary_shift(raw, timezero, clock_fun, leap_fun)
            missing = coverage.uncovered(shifted)
            if not np.any(missing > tolerance):
                continue
            # Only a hole *inside* the file can be filled; past either end there is
            # nothing to fit an orbit to.
            interior = (shifted > coverage.samples[0]) & (shifted < coverage.samples[-1])
            hint = f" {FILL_HINT}" if fill_hint and np.any(interior & (missing > tolerance)) else ""

            good = inside_gti(raw, gti)
            fatal = (missing > tolerance) & good
            if np.any(fatal):
                raise ValueError(
                    f"{hdu.name}/{column}: {np.count_nonzero(fatal)} times inside the good "
                    f"time intervals are up to {missing[fatal].max():.3f} s outside the "
                    f"orbit file, which only covers "
                    f"{coverage.samples[0]:.3f} to {coverage.samples[-1]:.3f}. The "
                    "spacecraft position there would be an extrapolation, so these times "
                    "cannot be barycentred. Supply an orbit file covering the observation." + hint
                )

            drop = (missing > tolerance) & ~good
            logger.warning(
                f"{hdu.name}/{column}: dropping {np.count_nonzero(drop)} rows that fall "
                f"outside every good time interval AND up to {missing[drop].max():.3f} s "
                f"outside the orbit file. The spacecraft position there is unknown, so "
                f"these times cannot be barycentred." + hint
            )
            hdu.data = data[~drop]
            data = hdu.data
            dropped += int(np.count_nonzero(drop))
    return dropped


def filled_gap_report(filler):
    """One line per gap whose position was fitted rather than tabulated.

    Only gaps that the correction has actually been asked about are listed. The fit's own
    residual is quoted, with the caveat that it understates the error inside the gap.
    """
    if filler is None or not filler.fit_rms:
        return []
    lines = [
        "POSITIONS FITTED, NOT TABULATED: the orbit file has gaps, and the spacecraft "
        "position inside them comes from a Kepler+J2 orbit fitted to the samples around "
        "each (--fill-orbit-gaps). Error of the order of 100 m (0.4 us), up to about "
        "1 us in the worst case."
    ]
    for index in sorted(filler.fit_rms):
        start, end = filler.gaps[index]
        lines.append(
            f"Orbit gap filled: MET {start:.1f} to {end:.1f} ({end - start:.0f} s), "
            f"fit residual {filler.fit_rms[index]:.0f} m"
        )
    return lines


def correct_times(times, bary_fun, clock_fun=None, leap_fun=None):
    """Apply the clock and barycentric corrections to an array of times.

    Parameters
    ----------
    times : array-like
        Times to correct, in mission elapsed seconds.
    bary_fun : callable
        The barycentric correction, from :func:`get_barycentric_correction`.
    clock_fun : callable, optional
        The clock correction. If None, no clock correction is applied.
    leap_fun : callable, optional
        The leap-second term for a mission whose MET counts UTC seconds, from
        :func:`barycenter.utils.leap_seconds_since_mjdref`. Not a clock correction: it is
        a time-system conversion, it is applied even when ``clock_fun`` is None, and both
        are evaluated on the **raw** times.

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
    given. With a leap-second term the first line becomes ``t' = t + clock(t) + leap(t)``,
    both still evaluated at the raw ``t``: on Swift, evaluating the clock polynomial at
    the leap-shifted time instead moves every event 214 ns, which is above the target and
    is not what ``barycorr`` does. This package used to compute ``t + clock(t) + bary(t)``, evaluating both on the
    raw mission time. On NuSTAR, where the clock correction reaches 25 ms, that was
    measured at +1146 ns mean and 1878 ns peak against ``barycorr`` -- an order of
    magnitude above the 100 ns target, and the reason the reference tests used to be run
    with the clock correction switched off.
    """
    if clock_fun is None and leap_fun is None:
        return times + bary_fun(times)

    shifted = times
    if leap_fun is not None:
        shifted = shifted + leap_fun(times)
    if clock_fun is not None:
        shifted = shifted + clock_fun(times)
    return shifted + bary_fun(shifted)


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
    dt=None,
    fill_orbit_gaps=False,
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
    dt : float, optional
        Grid spacing in seconds for interpolating the correction instead of evaluating it
        at every event. The default, ``None``, decides from the file's size: see
        :func:`grid_spacing_for` and :data:`AUTO_GRID_EVENTS`. Zero forces the exact
        per-event path whatever the size.
    fill_orbit_gaps : bool, optional
        Fit an orbit across long gaps inside the orbit file (the Fermi LAT's South
        Atlantic Anomaly passages, for instance) instead of refusing the times that fall
        in them. The position there then has an error of the order of 100 m (0.4 us of
        light time); see :mod:`barycenter.gapfill`. Native engine only.
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

        # TIMEZERO is folded in, but TIMEPIXR deliberately is not. This used to add
        # ``(0.5 - TIMEPIXR) * TIMEDEL``, moving every RXTE PCA event half a clock tick
        # (477 ns) away from what barycorr produces, while leaving TIMEPIXR itself
        # unchanged in the output header -- so the file then claimed a convention its
        # times no longer followed. Where the time stamp sits inside its bin is not the
        # barycentring tool's business.
        timezero = hdul[1].header.get("TIMEZERO", 0.0)

        mission = hdul[1].header.get("TELESCOP", "unknown").lower()
        logger.info(f"Mission: {mission}")

        clock_fun, clock_accuracy = None, None
        if isinstance(clockfile, str) and clockfile.lower() == "none":
            logger.info("Clock correction explicitly disabled")
            clockfile = None
        else:
            # Which missions have a clock correction, and where each one comes from, is
            # barycenter.clock's business; this function stays mission-agnostic.
            clock_fun, clockfile, clock_accuracy = clock_correction_fun(
                mission, clockfile, instrument=hdul[1].header.get("INSTRUME")
            )

        # The accuracy the clock correction achieves, for TIERABSO, measured over the
        # file's own span before anything moves. Left as None when no clock correction
        # was applied, in which case the keyword is not touched: the honest figure would
        # then be the size of the correction we did not apply, which is exactly the thing
        # we cannot know without the clock file.
        tierabso = None
        if clock_fun is not None and clock_accuracy is not None:
            tierabso = clock_accuracy(
                hdul[1].header.get("TSTART", 0.0), hdul[1].header.get("TSTOP", 0.0)
            )
            logger.info(f"Clock correction accurate to {tierabso * 1e6:.1f} us (TIERABSO)")

        # Needed by the leap-second term below and by the DATE-OBS/DATE-END/MJD-OBS
        # keywords at the end of the loop. A file with no MJDREF at all gets None and
        # keeps its date keywords, rather than the whole run failing over metadata.
        try:
            mjdref = high_precision_mjdref(hdul[1].header)
        except ValueError:
            mjdref = None
            logger.warning(
                "No MJDREF in the header: DATE-OBS, DATE-END and MJD-OBS cannot be "
                "recomputed and are left as they are."
            )

        # Not part of the clock correction, and so not switched off with it: a mission
        # whose MET counts UTC seconds needs the leap seconds since MJDREF added whatever
        # --clockfile said. Leaving them out is a whole-second error, which is not the
        # kind of thing a flag should be able to cause.
        leap_fun = None
        if _mission_counts_utc_seconds(mission):
            if mjdref is None:
                raise ValueError(
                    f"{mission} counts UTC seconds, so the leap seconds since MJDREF "
                    "have to be added, but the header has no MJDREF to count from."
                )
            leap_fun = partial(leap_seconds_since_mjdref, mjdref)
            logger.info(
                f"{mission} MET counts UTC seconds: adding "
                f"{leap_fun(hdul[1].header.get('TSTART', 0.0)):.1f} s of leap seconds"
            )

        if only_columns is not None:
            hdul = slim_down_hdu_list(hdul, additional_cols=only_columns)

        # The correction is built here, and not earlier, because the two things that
        # decide its shape are only known now: the clock and leap-second terms, which
        # say *where* it will be evaluated, and the file's size, which says how finely.
        n_events = max(
            (len(hdu.data) for hdu in hdul if column_named(getattr(hdu, "data", None), "TIME")),
            default=0,
        )
        grid_dt = grid_spacing_for(n_events, dt)
        met_range = met_range_for_file(hdul, timezero, clock_fun, leap_fun)
        if grid_dt is None:
            logger.info(f"{n_events} events: evaluating the correction at every event")
        else:
            logger.info(
                f"{n_events} events: interpolating the correction on a {grid_dt} s grid "
                f"(worth about 1.6 ns at 5 s; pass --dt 0 to evaluate it at every event)"
            )

        bary_fun = get_barycentric_correction(
            orbfile,
            ra=ra,
            dec=dec,
            ephem=ephem,
            radecsys=radecsys,
            model=model if engine == "pint" else None,
            engine=engine,
            dt=grid_dt,
            met_range=met_range,
            fill_gaps=fill_orbit_gaps,
        )

        # Before anything is written: refuse the file if the orbit does not reach the
        # data, and drop the rows it does not reach that were never good times anyway.
        # The span is read afterwards, so that a TSTART pulled back to the start of the
        # data is pulled back to data that survived.
        coverage = getattr(bary_fun, "coverage", None)
        enforce_orbit_coverage(hdul, coverage, timezero, clock_fun, leap_fun)
        span = good_time_span(hdul)

        for hdu in hdul:
            logger.info(f"Updating HDU {hdu.name}")
            for keyname in TIME_COLUMNS:
                # Not ``keyname in hdu.data.names``: FITS column names are
                # case-insensitive and Chandra writes ``time`` in lower case, so a
                # case-sensitive test would skip its events while still correcting the
                # capitalised START/STOP of the GTI beside them.
                column = column_named(hdu.data, keyname)
                if column is not None:
                    logger.info(f"Updating column {column}")
                    _warn_outside_grid(
                        bary_fun,
                        hdu.data[column],
                        f"{hdu.name}/{column}",
                        timezero,
                        clock_fun,
                        leap_fun,
                    )
                    hdu.data[column] = correct_times(
                        hdu.data[column] + timezero, bary_fun, clock_fun, leap_fun
                    )
                if keyname in hdu.header:
                    logger.info(f"Updating header keyword {keyname}")
                    raw_keyword = clamp_uncovered_keyword(
                        hdu.header[keyname],
                        keyname,
                        coverage,
                        span,
                        timezero,
                        clock_fun,
                        leap_fun,
                    )
                    corrected_time = correct_times(
                        raw_keyword + timezero, bary_fun, clock_fun, leap_fun
                    )
                    if not np.isfinite(corrected_time):
                        logger.error(
                            f"Bad value when updating header keyword {keyname}: "
                            f"{hdu.header[keyname]}->{corrected_time}"
                        )
                    else:
                        hdu.header[keyname] = corrected_time

            # TSTART and TSTOP have moved, so everything computed from them is now
            # wrong. This has to come after the loop above, not inside it.
            update_derived_keywords(hdu.header, mjdref)

            for keyname in ABSORBED_KEYWORDS:
                if keyname in hdu.header:
                    logger.info(f"Removing {keyname}, now folded into the times")
                    del hdu.header[keyname]

            if tierabso is not None:
                # Written even where the input had no such keyword, as hdaxbary does: it
                # is the accuracy of a correction this run applied, so it is measured
                # rather than invented. Where no clock correction was applied it is left
                # alone, which is what HEASOFT does for NuSTAR and RXTE.
                hdu.header["TIERABSO"] = (tierabso, "Absolute precision of clock correction")

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

        filled_report = filled_gap_report(getattr(bary_fun, "gap_filler", None))
        for line in filled_report:
            logger.warning(line)
            for hdu in hdul:
                for piece in textwrap.wrap(line, 70):
                    hdu.header.add_history(piece)

        hdul.writeto(outfile, overwrite=overwrite, output_verify="ignore")

    return outfile
