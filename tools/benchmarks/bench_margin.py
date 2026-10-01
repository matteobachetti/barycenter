"""How much margin a clipped spline grid needs before its edges stop showing.

An interpolator's end conditions make its first and last few intervals a different
function from its interior, so a grid clipped to the events puts those intervals exactly
where the events are. This measures the error against the unclipped grid as the margin
grows.

**Both engines now pad by ``2 * dt`` of their own**, so the margins below sit on top of
that and every row should read 0.000 ns. That is the fix working. To see the 46 ns it was
built for, change ``2 * dt`` back to ``dt`` in ``native.native_barycentric_correction``
and ``pintengine.pint_barycentric_correction``.
"""

import os
import pathlib

import numpy as np
from loguru import logger as _loguru

_loguru.remove()  # PINT logs every TOA array at DEBUG

from barycenter.orbit import read_orbit  # noqa: E402
from barycenter.pintengine import pint_barycentric_correction, timing_model_for_position  # noqa: E402

DATADIR = os.environ.get(
    "BARY_DATADIR", str(pathlib.Path(__file__).resolve().parents[2] / "tests" / "data")
)
ORBIT = os.path.join(DATADIR, "dummy_orb.fits.gz")
DT = 5.0

orbit = read_orbit(ORBIT)
met = np.asarray(orbit["MET"].value, dtype=np.float64)
model = timing_model_for_position(294.9107, 21.58308, "DE440")

full = pint_barycentric_correction(orbit, model, dt=DT)

t0 = met.min() + 1000.0
span = 300.0
probe = np.linspace(t0, t0 + span, 400)

print(f"{'margin':>10} {'grid pts':>9} {'max|diff| vs full':>20}")
for steps in (1, 2, 3, 4, 6, 8, 16):
    pad = steps * DT
    clipped = pint_barycentric_correction(
        orbit, model, dt=DT, met_range=(t0 - pad + DT, t0 + span + pad - DT)
    )
    d = np.max(np.abs(clipped(probe) - full(probe)))
    print(f"{steps:>8} st {len(clipped.x):>9} {d * 1e9:>17.3f} ns")

# And the same question for the native engine, whose spline is ours.
from barycenter.native import native_barycentric_correction  # noqa: E402

exact = native_barycentric_correction(orbit, 294.9107, 21.58308, ephem="DE440")
print("\nnative engine, same test (exact answer available):")
print(f"{'margin':>10} {'grid pts':>9} {'max|diff| vs exact':>20}")
for steps in (1, 2, 3, 4, 6, 8, 16):
    pad = steps * DT
    sp = native_barycentric_correction(
        orbit,
        294.9107,
        21.58308,
        ephem="DE440",
        dt=DT,
        met_range=(t0 - pad + DT, t0 + span + pad - DT),
    )
    d = np.max(np.abs(sp(probe) - exact(probe)))
    print(f"{steps:>8} st {len(sp.x):>9} {d * 1e9:>17.3f} ns")
