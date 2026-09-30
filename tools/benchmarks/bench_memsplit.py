"""Which phase of the native engine actually holds the memory.

``tracemalloc`` peak, measured around each phase separately at 1e6 events, so the
factor-of-four in a whole-file run can be attributed to a phase rather than guessed at.
"""

import os
import pathlib
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
N = 1_000_000

orbit = read_orbit(os.path.join(DATADIR, "dummy_orb.fits.gz"))
mjdref = orbit.meta["mjdref"]
met_tab = np.asarray(orbit["MET"].value, dtype=np.float64)
sc = spacecraft_interpolator(
    met_tab,
    np.column_stack([orbit[c].value for c in ("X", "Y", "Z")]),
    np.column_stack([orbit[c].value for c in ("Vx", "Vy", "Vz")]),
)
n_hat = source_unit_vector(RA, DEC, frame="icrs", ephem_frame=ephemeris_frame(EPHEM))
met = np.linspace(met_tab.min() + 10, met_tab.max() - 10, N)


def peak_of(label, fn):
    tracemalloc.start()
    out = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  {label:<46s} {peak / 1e6:>7.0f} MB")
    return out


print(
    f"tracemalloc peak per phase, {N} events (one (N,3) float64 array = {N * 3 * 8 / 1e6:.0f} MB)"
)

time_obj = peak_of("met_to_time", lambda: met_to_time(met, mjdref))
pos_sc = peak_of("spacecraft interpolation", lambda: sc(met))


def ephem_call():
    with solar_system_ephemeris.set(resolve_ephemeris(EPHEM)):
        return get_body_barycentric_posvel("earth", time_obj), get_body_barycentric("sun", time_obj)


earth, sun = peak_of("astropy JPL ephemeris (earth posvel + sun)", ephem_call)
peak_of("erfa.dtdb", lambda: erfa.dtdb(time_obj.jd1, time_obj.jd2, 0.0, 0.0, 0.0, 0.0))

r_earth = earth[0].xyz.to_value(u.m).T
v_earth = earth[1].xyz.to_value(u.m / u.s).T
r_sun = sun.xyz.to_value(u.m).T
peak_of(
    "unpacking .xyz.to_value(u.m).T (three arrays)",
    lambda: (
        earth[0].xyz.to_value(u.m).T,
        earth[1].xyz.to_value(u.m / u.s).T,
        sun.xyz.to_value(u.m).T,
    ),
)


def our_maths():
    r_obs = r_earth + pos_sc
    topo = np.einsum("ij,ij->i", pos_sc, v_earth) / C_M_S**2
    roemer = r_obs @ n_hat / C_M_S
    uv = r_obs - r_sun
    r_o = np.linalg.norm(uv, axis=1)
    return topo + roemer + 2 * T_SUN * np.log1p((uv @ n_hat) / r_o)


peak_of("our own vector arithmetic (fusable by numba)", our_maths)

print("\nwhole function in one go, for comparison:")
from barycenter.native import barycentric_correction  # noqa: E402

peak_of(
    "barycentric_correction",
    lambda: barycentric_correction(met, mjdref, RA, DEC, sc, ephem=EPHEM),
)
