"""Is the 46 ns from clipping really an edge effect, and does the native spline share it?

Compares clipped against unclipped for both engines -- the comparison that isolates the
end condition, since the two splines share every interior knot -- and shows where in the
span the difference lives.

**This now reports 0.000 ns for PINT, which is the point: it is measuring the fixed
code.** Both engines pad a clipped grid by ``2 * dt`` of their own, so the margins below
are *on top of* that and the smallest one reachable from here is already two steps. To
reproduce the 46 ns that motivated the fix, change ``2 * dt`` back to ``dt`` in
``native.native_barycentric_correction`` and ``pintengine.pint_barycentric_correction``
and run this again.
"""

import os
import pathlib

import numpy as np
from loguru import logger as _loguru

_loguru.remove()

from barycenter.native import native_barycentric_correction  # noqa: E402
from barycenter.orbit import read_orbit  # noqa: E402
from barycenter.pintengine import pint_barycentric_correction, timing_model_for_position  # noqa: E402

DATADIR = os.environ.get(
    "BARY_DATADIR", str(pathlib.Path(__file__).resolve().parents[2] / "tests" / "data")
)
orbit = read_orbit(os.path.join(DATADIR, "dummy_orb.fits.gz"))
met = np.asarray(orbit["MET"].value, dtype=np.float64)
RA, DEC, DT = 294.9107, 21.58308, 5.0
model = timing_model_for_position(RA, DEC, "DE440")

t0, span = met.min() + 1000.0, 300.0
probe = np.linspace(t0, t0 + span, 400)

builders = {
    "pint": lambda rng: pint_barycentric_correction(orbit, model, dt=DT, met_range=rng),
    "native": lambda rng: native_barycentric_correction(
        orbit, RA, DEC, ephem="DE440", dt=DT, met_range=rng
    ),
}

for name, build in builders.items():
    full = build(None)
    for steps in (1, 2):
        pad = steps * DT
        clipped = build((t0 - pad + DT, t0 + span + pad - DT))
        d = clipped(probe) - full(probe)
        # Where does the difference live? Split the probe into thirds.
        third = len(probe) // 3
        head, mid, tail = d[:third], d[third:-third], d[-third:]
        print(
            f"{name:>7} {steps} step  max|diff| {np.max(np.abs(d)) * 1e9:8.3f} ns  "
            f"| first third {np.max(np.abs(head)) * 1e9:7.3f}  "
            f"middle {np.max(np.abs(mid)) * 1e9:7.3f}  "
            f"last third {np.max(np.abs(tail)) * 1e9:7.3f} ns"
        )
    print()
