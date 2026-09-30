"""What the package knows about each mission, in one place.

Everything mission-specific lives here: which ``TELESCOP`` values identify a mission, where
its orbit file keeps position and velocity, whether it has a clock correction and what
builds it, and which official tool can be shelled out to for it. Nothing else in the
package branches on a mission name.

Adding a mission is one :class:`Mission` entry. The fields are checked by the dataclass, so
a misspelt one fails at import rather than silently doing nothing, and
:func:`mission_for` raises with the list of known missions rather than guessing.

Three columns of :data:`MISSIONS` deserve a word:

``orbit``
    ``None`` means there is no native orbit reader, because the mission does not
    distribute a spacecraft position in a format we read (ASCA). Such a mission still
    works through ``--apply-official``.
``clock``
    ``None`` means the mission needs no clock correction -- which for most of them is
    because the correction is already applied in the pipeline that produced the event file.
``official``
    The tool ``--apply-official`` hands the file to, or ``None`` if there is none -- which
    for XMM and Chandra means the native engine is the only way in, since ``barycorr``
    handles neither and their own tools, SAS ``barycen`` and CIAO ``axbary``, need all of
    SAS and all of CIAO respectively.
"""

from collections.abc import Callable
from dataclasses import dataclass

import astropy.units as u

from .clock import nustar_clock_builder, rxte_clock_builder, swift_clock_builder
from .orbit import OrbitSpec

__all__ = ["Mission", "MISSIONS", "mission_for"]


#: The shape shared by NICER, RXTE and IXPE: three scalar position columns in metres.
#: PINT calls this an FPorbit file, after the RXTE product it comes from.
FPORBIT = OrbitSpec(
    pos=("X", "Y", "Z"), vel=("Vx", "Vy", "Vz"), expected_extnames=("ORBIT", "XTE_PE")
)


@dataclass(frozen=True)
class Mission:
    """Everything the package knows about one mission.

    Parameters
    ----------
    name : str
        Canonical short name, matching the key in :data:`MISSIONS`.
    telescop : tuple of str
        Lower-case substrings of the ``TELESCOP`` keyword that identify this mission.
        Substrings, because the keyword is written inconsistently: ``XTE`` and ``RXTE``,
        ``NuSTAR`` and ``NUSTAR``, ``AXAF`` and ``CHANDRA``.
    orbit : ~barycenter.orbit.OrbitSpec or None
        How to read the orbit file. ``None`` if there is no native reader.
    clock : callable or None
        ``clock(clockfile, instrument)`` returning
        ``(correction_function, path, accuracy)``, where ``accuracy`` is a callable of the
        MET span giving the clock's absolute accuracy in seconds, which ``core.py`` writes
        into every extension as ``TIERABSO``. ``None`` if the mission needs no clock
        correction.
    official : str or None
        The mission's own tool: ``"barycorr"``, ``"timeconv"``, or ``None``.
    official_ephem : str or None
        The only ephemeris that tool can use, if it is limited to one. ASCA's ``timeconv``
        is fixed to DE200, which is the main reason to prefer the native engine for that
        mission.
    met_is_utc : bool
        Whether the mission's elapsed time counts **UTC** seconds rather than TT seconds,
        so that the leap seconds inserted since ``MJDREF`` have to be added before
        anything else. True for Swift, and for nothing else here that has been checked
        against an official tool. This is not a clock correction and is applied even with
        ``--clockfile none``: getting it wrong is a whole number of seconds, so it must
        not depend on a flag. See
        :func:`barycenter.utils.leap_seconds_since_mjdref`.
    """

    name: str
    telescop: tuple
    orbit: "OrbitSpec | None" = None
    clock: "Callable | None" = None
    official: "str | None" = None
    official_ephem: "str | None" = None
    met_is_utc: bool = False

    @property
    def has_native_support(self):
        """Whether the pure-Python path can handle this mission."""
        return self.orbit is not None


#: One entry per mission. The key is the canonical name; ``telescop`` is what is matched
#: against a file's header.
MISSIONS = {
    "nustar": Mission(
        name="nustar",
        telescop=("nustar",),
        # The only mission that tabulates position in kilometres.
        orbit=OrbitSpec(pos="POSITION", vel="VELOCITY", pos_unit=u.km, vel_unit=u.km / u.s),
        clock=nustar_clock_builder,
        official="barycorr",
    ),
    "nicer": Mission(
        name="nicer",
        telescop=("nicer",),
        orbit=OrbitSpec(pos=("X", "Y", "Z"), vel=("Vx", "Vy", "Vz"), expected_extnames=("ORBIT",)),
        official="barycorr",
    ),
    "rxte": Mission(
        name="rxte",
        telescop=("xte",),
        orbit=FPORBIT,
        clock=rxte_clock_builder,
        official="barycorr",
    ),
    "ixpe": Mission(
        name="ixpe",
        telescop=("ixpe",),
        orbit=OrbitSpec(pos=("X", "Y", "Z"), vel=("Vx", "Vy", "Vz"), expected_extnames=("ORBIT",)),
    ),
    "fermi": Mission(
        name="fermi",
        telescop=("fermi", "glast"),
        orbit=OrbitSpec(
            pos="SC_POSITION",
            vel="SC_VELOCITY",
            time_col="START",
            expected_extnames=("SC_DATA",),
        ),
    ),
    "svom": Mission(
        name="svom",
        telescop=("svom",),
        orbit=OrbitSpec(pos="POSITION", vel="VELOCITY"),
    ),
    "xmm": Mission(
        name="xmm",
        # The PPS `P*OBX000ORBTSR0000.FTZ` file offers two position triples: GEI is
        # geocentric equatorial, which is what the ephemeris is referred to, and GSE is
        # geocentric solar-ecliptic, which is the same vector rotated into the
        # Earth-Sun frame. Both have the same length, so picking the wrong one is not a
        # small error but a 160 ms one, and nothing in the file's units or comments
        # would give it away. Velocities have no such prefix; there is only one triple.
        orbit=OrbitSpec(
            pos=("GEI_X", "GEI_Y", "GEI_Z"),
            vel=("VX", "VY", "VZ"),
            pos_unit=u.km,
            vel_unit=u.km / u.s,
            expected_extnames=("ORBIT",),
        ),
        telescop=("xmm",),
        # `barycorr` refuses XMM outright ("Invalid Observatory/Spacecraft position
        # vector"), and the mission's own tool, SAS `barycen`, will not take an orbit
        # file on the command line: it reaches the spacecraft position through SAS's
        # observation access layer, so it needs a full SAS installation and an ingested
        # ODF. Wrapping that is not worth it when the native engine agrees with it to a
        # constant 40 ns; `tools/make_test_data.py` drives it for the reference file and
        # nowhere else.
        official=None,
    ),
    "chandra": Mission(
        name="chandra",
        # `primary/orbitf*_eph1.fits`, extension ORBITEPHEM, in metres. The column names
        # are spelt `Time`, `X`, `Vx`: FITS column names are case-insensitive by standard
        # and Chandra is the mission that relies on it, so every lookup here goes through
        # `utils.column_named`.
        orbit=OrbitSpec(
            pos=("X", "Y", "Z"), vel=("Vx", "Vy", "Vz"), expected_extnames=("ORBITEPHEM",)
        ),
        telescop=("chandra", "axaf"),
        # Not `barycorr`, despite what the mission list in its documentation suggests:
        # `hdaxbary` carries orbit-file readers for RXTE, NICER, Swift and NuSTAR only,
        # and on a Chandra orbit file it fails with "no bracketing sample found" followed
        # by "Invalid Observatory/Spacecraft position vector", on a file that brackets the
        # time comfortably. The mission's own tool is CIAO `axbary`, which needs all of
        # CIAO and can reach only DE200 (`refframe=FK5`) or DE405 (`refframe=ICRS`);
        # `tools/make_test_data.py` drives it for the reference files and nowhere else.
        official=None,
    ),
    "swift": Mission(
        name="swift",
        # `auxil/sw<obsid>sao.fits`, the prefilter product, extension PREFILTER. Same
        # vector-column shape as NuSTAR and SVOM, and in kilometres like NuSTAR -- SVOM's
        # identically named columns are metres, which is a 20 ms error either way round.
        orbit=OrbitSpec(
            pos="POSITION",
            vel="VELOCITY",
            pos_unit=u.km,
            vel_unit=u.km / u.s,
            expected_extnames=("PREFILTER",),
        ),
        telescop=("swift",),
        clock=swift_clock_builder,
        # The one mission here whose MET counts UTC seconds rather than TT seconds. See
        # `utils.leap_seconds_since_mjdref`: for a 2015 observation this is worth 4 s, and
        # it is applied whatever `--clockfile` says.
        met_is_utc=True,
        official="barycorr",
    ),
    # ASCA is the odd one: barycorr refuses it, and the only tool is `timeconv`, which
    # needs a downloaded earth.dat and frf.orbit file and can only do DE200.
    "asca": Mission(name="asca", telescop=("asca",), official="timeconv", official_ephem="DE200"),
}


def mission_for(telescope):
    """The :class:`Mission` matching a ``TELESCOP`` keyword value.

    Parameters
    ----------
    telescope : str
        The keyword as written in the file, in any case.

    Returns
    -------
    Mission

    Raises
    ------
    ValueError
        If nothing matches, listing what is known.
    """
    name = str(telescope).lower()
    for mission in MISSIONS.values():
        if any(alias in name for alias in mission.telescop):
            return mission
    raise ValueError(
        f"Unknown mission {telescope!r}. Known missions: {', '.join(sorted(MISSIONS))}. "
        "Adding one is a single entry in barycenter.missions.MISSIONS."
    )
