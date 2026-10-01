"""Where the time goes in a native-engine run, as a function of event count.

Answers three questions at once:
  * what the native engine costs per event, and what a ``dt`` spline saves;
  * what that spline costs in accuracy, against the exact per-event answer;
  * what share of the run the numba-accelerated clock interpolation actually is.
"""

import os
import pathlib
import time

import numpy as np

from barycenter.clock import clock_correction_fun
from barycenter.native import native_barycentric_correction
from barycenter.orbit import read_orbit

DATADIR = os.environ.get(
    "BARY_DATADIR", str(pathlib.Path(__file__).resolve().parents[2] / "tests" / "data")
)
ORBIT = os.path.join(DATADIR, "dummy_orb.fits.gz")
CLK = os.path.join(DATADIR, "dummy_fine_clk.fits")
RA, DEC = 294.9107, 21.58308


def timed(fn):
    a = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - a


orbit, t_read = timed(lambda: read_orbit(ORBIT))
met = np.asarray(orbit["MET"].value, dtype=np.float64)
print(f"orbit file: {len(orbit)} samples, span {met.max() - met.min():.0f} s")
print(f"read_orbit: {t_read * 1000:.1f} ms\n")

clock_fun, _, _ = clock_correction_fun("nustar", CLK)

# The very first call builds the interpolation kernel -- and compiles it, if numba is
# installed. That is a real cost, but it is paid once per process, so it is reported on
# its own rather than being charged to whichever row of the table happens to run first.
_warm = timed(lambda: clock_fun(met[:2]))[1]
print(f"first clock call (builds and compiles the kernel): {_warm * 1000:.1f} ms")
print("  -- with numba absent this is under a millisecond; see bench_numba.py\n")

exact = native_barycentric_correction(orbit, RA, DEC, ephem="DE440")
spline, t_build = timed(
    lambda: native_barycentric_correction(orbit, RA, DEC, ephem="DE440", dt=5.0)
)
print(f"dt=5 spline build over the whole {met.max() - met.min():.0f} s orbit: {t_build:.3f} s")
print(f"  ({int((met.max() - met.min()) / 5) + 1} grid points)\n")

rng = np.random.default_rng(1)
hdr = f"{'events':>10} {'exact':>10} {'us/ev':>8} {'spline eval':>12} {'clock':>10} {'clock %':>8}"
print(hdr)
print("-" * len(hdr))
for n in (1_000, 10_000, 100_000, 1_000_000):
    times = np.sort(rng.uniform(met.min() + 10, met.max() - 10, n))
    _, t_exact = timed(lambda: exact(times))
    _, t_spline = timed(lambda: spline(times))
    _, t_clock = timed(lambda: clock_fun(times))
    share = 100 * t_clock / (t_exact + t_clock)
    print(
        f"{n:>10} {t_exact:>8.3f}s {t_exact / n * 1e6:>7.2f} "
        f"{t_spline * 1000:>10.1f}ms {t_clock * 1000:>8.1f}ms {share:>7.2f}%"
    )

# How much accuracy the spline costs, on a dense sample.
check = np.sort(rng.uniform(met.min() + 10, met.max() - 10, 20_000))
resid = spline(check) - exact(check)
print(
    f"\ndt=5 spline vs exact: mean {np.mean(resid) * 1e9:+.3f} ns, "
    f"std {np.std(resid) * 1e9:.3f} ns, max|.| {np.max(np.abs(resid)) * 1e9:.3f} ns"
)
