"""What clipping the PINT TOA grid to the events is worth.

The realistic case is a short snapshot inside a per-day orbit file: the committed
NuSTAR orbit spans 82800 s, and a 300 s observation inside it needs 60 TOAs rather
than 16560.
"""

import os
import pathlib
import time

import numpy as np

from barycenter.core import get_barycentric_correction
from barycenter.orbit import read_orbit

DATADIR = os.environ.get(
    "BARY_DATADIR", str(pathlib.Path(__file__).resolve().parents[2] / "tests" / "data")
)
ORBIT = os.path.join(DATADIR, "dummy_orb.fits.gz")
RA, DEC = 294.9107, 21.58308


def timed(fn):
    a = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - a


met = np.asarray(read_orbit(ORBIT)["MET"].value, dtype=np.float64)
t0 = met.min() + 1000.0

print(
    f"orbit span {met.max() - met.min():.0f} s -> {int((met.max() - met.min()) / 5) + 1} TOAs at dt=5"
)
print(
    f"\n{'observation':>14} {'TOAs':>7} {'no met_range':>14} {'with met_range':>16} {'speedup':>9}"
)
for span in (300.0, 3000.0, 30000.0):
    rng = (t0, t0 + span)
    _, t_all = timed(lambda: get_barycentric_correction(ORBIT, ra=RA, dec=DEC, engine="pint"))
    fun, t_clip = timed(
        lambda: get_barycentric_correction(ORBIT, ra=RA, dec=DEC, engine="pint", met_range=rng)
    )
    print(
        f"{span:>12.0f} s {len(fun.x):>7} {t_all:>12.2f} s {t_clip:>14.2f} s {t_all / t_clip:>8.1f}x"
    )

# The clipped grid must give the same answer inside the span it was asked for.
rng = (t0, t0 + 300.0)
full = get_barycentric_correction(ORBIT, ra=RA, dec=DEC, engine="pint")
clipped = get_barycentric_correction(ORBIT, ra=RA, dec=DEC, engine="pint", met_range=rng)
probe = np.linspace(t0, t0 + 300.0, 500)
d = clipped(probe) - full(probe)
print(f"\nclipped vs full grid, inside the span: max|diff| {np.max(np.abs(d)) * 1e9:.3f} ns")
