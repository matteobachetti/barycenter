"""Blow the dummy NuSTAR event file up to N events, keeping its header and columns."""

import os
import sys

import numpy as np
from astropy.io import fits

datadir, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rng = np.random.default_rng(0)

with fits.open(os.path.join(datadir, "dummy_evt.evt")) as hdul:
    events = hdul[1]
    hdr = events.header
    t0, t1 = hdr["TSTART"], hdr["TSTOP"]
    cols = []
    for col in events.columns:
        if col.name.upper() == "TIME":
            data = np.sort(rng.uniform(t0, t1, n))
        else:
            old = events.data[col.name]
            data = np.resize(np.asarray(old), n)
        cols.append(fits.Column(name=col.name, format=col.format, unit=col.unit, array=data))
    big = fits.BinTableHDU.from_columns(cols, header=hdr, name=events.name)
    out_hdul = fits.HDUList([fits.PrimaryHDU(header=hdul[0].header), big] + list(hdul[2:]))
    out_hdul.writeto(out, overwrite=True)

print(
    f"{out}: {n} events, {os.path.getsize(out) / 1e6:.1f} MB, {os.path.getsize(out) / n:.1f} B/event"
)
