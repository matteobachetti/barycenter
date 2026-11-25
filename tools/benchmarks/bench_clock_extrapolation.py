"""How wrong is a NuSTAR clock correction extrapolated past the end of its table?

Cuts the newest cached NuSTAR clock table at 300 random epochs in its last three years,
extrapolates past each cut with four policies, and compares with what the table says
there. Prints 95th-percentile errors in microseconds. The numbers are quoted in
``docs/technical_details.md`` and justify ``CLOCK_EXTRAPOLATION_LIMIT_S``.

Usage: python tools/benchmarks/bench_clock_extrapolation.py [clockfile]
"""

import glob
import os
import sys

import numpy as np
from astropy.io import fits

from barycenter.clock import _cubic_interpolation, clock_cache_dir


def main(clockfile):
    data = fits.open(clockfile)["NU_FINE_CLOCK"].data
    t, o, f = (np.asarray(data[c], float) for c in ("TIME", "CLOCK_OFF_CORR", "CLOCK_FREQ_CORR"))
    rng = np.random.default_rng(1)
    recent = np.where(t > t[-1] - 3 * 3.15e7)[0]
    cuts = rng.choice(recent[:-20000], 300, replace=False)
    dts = np.array([100, 1000, 5000, 2e4, 86400, 3 * 86400, 7 * 86400, 14 * 86400.0])
    res = {k: [] for k in ("edge cubic", "linear", "hold", "zero")}
    for i in cuts:
        tt = t[i] + dts
        truth = np.interp(tt, t, o)
        res["edge cubic"].append(
            _cubic_interpolation(tt, t[i - 1], t[i], o[i - 1], o[i], f[i - 1], f[i]) - truth
        )
        res["linear"].append(o[i] + f[i] * dts - truth)
        res["hold"].append(o[i] - truth)
        res["zero"].append(-truth)
    print(f"{os.path.basename(clockfile)}: 95th percentile |error| in microseconds")
    print("dt [s]      " + "".join(f"{d:>10.0f}" for d in dts))
    for name, errors in res.items():
        row = np.percentile(np.abs(errors), 95, axis=0) * 1e6
        print(f"{name:11s} " + "".join(f"{x:10.3g}" for x in row))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main(sorted(glob.glob(os.path.join(clock_cache_dir(), "nuCclock*.fits*")))[-1])
