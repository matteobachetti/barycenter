"""How much of the native engine's per-event cost could a fused kernel reach?

Splits ``barycentric_correction`` into the parts a numba kernel could replace (our own
vector arithmetic, and the temporaries it allocates) and the parts it could not (astropy
Time construction, the JPL ephemeris evaluation, ERFA's dtdb). The ratio is the ceiling
on any such rewrite.
"""

import os
import pathlib
import time
import tracemalloc

import astropy.units as u
import erfa
import numpy as np
from astropy.coordinates import (
    get_body_barycentric,
    get_body_barycentric_posvel,
    solar_system_ephemeris,
)

from barycenter.native import (
    C_M_S,
    T_SUN,
    ephemeris_frame,
    met_to_time,
    resolve_ephemeris,
    source_unit_vector,
    spacecraft_interpolator,
)
from barycenter.orbit import read_orbit

DATADIR = os.environ.get(
    "BARY_DATADIR", str(pathlib.Path(__file__).resolve().parents[2] / "tests" / "data")
)
RA, DEC, EPHEM = 294.9107, 21.58308, "de440"

orbit = read_orbit(os.path.join(DATADIR, "dummy_orb.fits.gz"))
mjdref = orbit.meta["mjdref"]
met_tab = np.asarray(orbit["MET"].value, dtype=np.float64)
sc = spacecraft_interpolator(
    met_tab,
    np.column_stack([orbit[c].value for c in ("X", "Y", "Z")]),
    np.column_stack([orbit[c].value for c in ("Vx", "Vy", "Vz")]),
)
n_hat = source_unit_vector(RA, DEC, frame="icrs", ephem_frame=ephemeris_frame(EPHEM))


def timed(fn):
    a = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - a


print(
    f"{'N':>9} {'orbit+Time':>11} {'JPL ephem':>11} {'erfa.dtdb':>11} {'our maths':>11} {'ours %':>7}"
)
for n in (100_000, 1_000_000):
    met = np.linspace(met_tab.min() + 10, met_tab.max() - 10, n)

    (time_obj, pos_sc), t_setup = timed(lambda: (met_to_time(met, mjdref), sc(met)))

    def ephem_call():
        with solar_system_ephemeris.set(resolve_ephemeris(EPHEM)):
            return get_body_barycentric_posvel("earth", time_obj), get_body_barycentric(
                "sun", time_obj
            )

    ((earth, sun)), t_ephem = timed(ephem_call)
    _, t_dtdb = timed(lambda: erfa.dtdb(time_obj.jd1, time_obj.jd2, 0.0, 0.0, 0.0, 0.0))

    r_earth = earth[0].xyz.to_value(u.m).T
    v_earth = earth[1].xyz.to_value(u.m / u.s).T
    r_sun = sun.xyz.to_value(u.m).T

    def our_maths():
        r_obs = r_earth + pos_sc
        topo = np.einsum("ij,ij->i", pos_sc, v_earth) / C_M_S**2
        roemer = r_obs @ n_hat / C_M_S
        uv = r_obs - r_sun
        r_o = np.linalg.norm(uv, axis=1)
        return topo + roemer + 2 * T_SUN * np.log1p((uv @ n_hat) / r_o)

    _, t_maths = timed(our_maths)

    total = t_setup + t_ephem + t_dtdb + t_maths
    print(
        f"{n:>9} {t_setup:>9.3f}s {t_ephem:>9.3f}s {t_dtdb:>9.3f}s "
        f"{t_maths:>9.3f}s {100 * t_maths / total:>6.1f}%"
    )

# And the memory the arithmetic allocates, which is the other half of the case.
met = np.linspace(met_tab.min() + 10, met_tab.max() - 10, 1_000_000)
time_obj, pos_sc = met_to_time(met, mjdref), sc(met)
with solar_system_ephemeris.set(resolve_ephemeris(EPHEM)):
    earth = get_body_barycentric_posvel("earth", time_obj)
    sun = get_body_barycentric("sun", time_obj)
r_earth = earth[0].xyz.to_value(u.m).T
v_earth = earth[1].xyz.to_value(u.m / u.s).T
r_sun = sun.xyz.to_value(u.m).T

tracemalloc.start()
our = our_maths()
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"\n1e6 events: the arithmetic alone peaks at {peak / 1e6:.0f} MB of temporaries")
print(f"  (the result it returns is {our.nbytes / 1e6:.0f} MB)")
