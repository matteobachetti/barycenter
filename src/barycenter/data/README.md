# Bundled calibration data

## `tdc.dat`

RXTE fine clock correction coefficients, produced by `PUTTDC.PRO` (C. Markwardt) on
2012-01-10 and distributed with HEASOFT as `$HEADAS/refdata/tdc.dat`; the original
source is
<https://heasarc.gsfc.nasa.gov/FTP/xte/calib_data/clock/tdc.dat>.

It is bundled rather than fetched because RXTE stopped observing in January 2012 and
this file is final: it is 53 kB, it covers the whole mission, and there is no newer
version to pick up. HEASOFT's `barycorr` ignores its own `clockfile` parameter for RXTE
and reads this file, so having it here is what lets us reproduce `barycorr` without
HEASOFT installed. If a newer copy ever appears, point `$TIMING_DIR` or `$LHEA_DATA` at
it and :func:`barycenter.clock.rxte_tdc_file` will prefer that one.

The format is documented in
[`xCC.c`](https://heasarc.gsfc.nasa.gov/docs/xte/abc/xCC.c) by A. Rots, which is the
code this package's reader is translated from.
