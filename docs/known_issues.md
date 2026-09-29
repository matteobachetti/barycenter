# Known issues

An honest list of what is wrong or missing in the current release. Items are removed
from this page as they are fixed, and each fix should arrive with a test.

## Accuracy

**`--engine pint` sits ~116 ns from `barycorr`.** PINT's solar-system Shapiro delay
carries an extra `2·T☉·ln(r/AU)` annual term that `axBary` omits. Neither is wrong; they
differ by a term a pulsar fit would absorb. The default native engine uses the `axBary`
convention and does not have the offset. See
[Technical details](technical_details.md#accuracy).

**On Apple Silicon and Windows `--engine pint` is quantised at ~1.1 µs.** `numpy` has no
80-bit `longdouble` on those platforms, and PINT represents absolute times as
`longdouble` MJDs, so subtracting two large MJDs to get a small correction loses the
answer: measured 775 ns peak-to-peak on arm64 against 149 ns on x86. The default native
engine computes every term as a small quantity in float64 and gives bit-identical
answers on both.

**The coordinate fallback order differs from HEASOFT's.** We prefer
`RA_OBJ`/`DEC_OBJ`, `barycorr` prefers `RA_NOM`/`DEC_NOM`. The two keywords differ by
0.1 arcsec on the NuSTAR test file, which is 172 µs of light travel time. Always pass
`--ra`/`--dec` explicitly when the answer must match another tool.

**The `TIMEPIXR` half-bin shift is applied, but `barycorr` does not apply one.**
Measured on a NICER observation: `barycorr` shifts by `TIMEZERO` only, while this
package also adds `(0.5 - TIMEPIXR) * TIMEDEL`. That is 20 ns on NICER, but it scales
with `TIMEDEL` and would be much larger on a coarsely binned mission.

**`--radecsys` does not reach the computation with `--engine pint`.** PINT treats the
coordinates as ICRS whatever the keyword says, so asking for FK5 changes only the output
header. Getting this wrong is worth 45 µs. The default native engine handles it, by
rotating the source direction into the frame the ephemeris itself uses.

## Correctness

**`download_locally` corrupts astropy's download cache.** It calls
`download_file(cache=True)` and then `shutil.move`s the cached file out of the cache
directory, leaving the cache index pointing at nothing.

**`fits_open_remote` can return an unbound variable** if the fallback branch is not
taken.

## Missing features

**No XMM-Newton support.** `barycorr` refuses XMM data outright ("Invalid
Observatory/Spacecraft position vector"), and XMM event files carry no spacecraft
position at all — it has to come from the ODF or from the SAS `orbit` task. Doing this
natively would remove the need for a full SAS installation.

**Chandra is `--apply-official` only.** Native support would free Chandra timing from
`axbary`'s hard-coded DE200/DE405 choice: the CIAO build has no `-jpleph` switch, so
DE440 is simply not reachable through it.

**No RXTE clock correction.** `barycorr` ignores its own `clockfile` parameter for
RXTE and reads `$LHEA_DATA/tdc.dat` instead. The measured effect is 5.97e-5 s, so
matching `barycorr` on RXTE requires reading that file.

**Only the new NuSTAR clock format is read.** `nustar_clock_correction_fun` reads the
`NU_FINE_CLOCK` extension. Older files carry a `CLOCK_CORRECT` extension with C0/C1/C2
polynomial coefficients, documented by HEASOFT as accurate only to the millisecond;
they are refused with an explanatory error rather than supported, on the grounds that a
millisecond is four orders of magnitude worse than the target. Fetch a current clock
file from the CALDB instead.

## Performance

**`--engine pint`'s TOA grid spans the whole orbit file, not the events.** A 5-second
grid across a multi-day `.attorb` file, or across a stack of orbit files, costs tens of
thousands of PINT TOAs that are then never used. `pint_barycentric_correction` accepts a
`met_range` that clips the grid, but `apply_barycenter_correction` does not yet pass it.
The native engine uses no grid at all.

**The native engine evaluates the ephemeris once per event.** That is exact, and at
~6.5 µs per event it is still 30× faster than the PINT path, but a 10-million-event file
would take about a minute. `native_barycentric_correction` accepts a `dt` that puts a
cubic spline through a grid instead -- measured cost 1.2 ns at 5 s -- and nothing passes
it yet.

**The whole file is materialised in memory.** A 600 MB event file is read, modified and
written as one `HDUList`.

**`numba` is imported unconditionally** halfway down the module and compiles eagerly,
for a single small interpolation routine.
