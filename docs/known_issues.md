# Known issues

An honest list of what is wrong or missing in the current release. Items are removed
from this page as they are fixed, and each fix should arrive with a test.

## Accuracy

**The clock correction is applied in the wrong order.** We compute
`t + clock(t) + bary(t)`; HEASOFT `barycorr` computes `t' = t + clock(t)` and then
`t' + bary(t')`, also looking up the spacecraft position at `t'`. On NuSTAR this is a
~1.1 µs difference. Until it is fixed, comparisons against `barycorr` must be run with
`--clockfile none`, which is what the test suite does.

**PINT's Shapiro delay differs from `axBary`'s by ~100 ns.** PINT's solar-system
Shapiro delay carries an extra `2·T☉·ln(r/AU)` annual term that `axBary` omits. This
accounts for the +116 ns mean offset measured against the reference. Neither is wrong;
they differ by a term a pulsar fit would absorb. See
[Technical details](technical_details.md#accuracy).

**On Apple Silicon and Windows the correction is quantised at ~1.1 µs.** `numpy` has no
80-bit `longdouble` on those platforms, and PINT represents absolute times as
`longdouble` MJDs, so subtracting two large MJDs to get a small correction loses the
answer. The test suite loosens its tolerance from 200 ns to 2 µs where extended
precision is absent. Computing the correction terms directly, rather than as the
difference of two absolute epochs, removes the problem.

**The coordinate fallback order differs from HEASOFT's.** We prefer
`RA_OBJ`/`DEC_OBJ`, `barycorr` prefers `RA_NOM`/`DEC_NOM`. The two keywords differ by
0.1 arcsec on the NuSTAR test file, which is 172 µs of light travel time. Always pass
`--ra`/`--dec` explicitly when the answer must match another tool.

**With a `.par` file the `PLEPHEM` header keyword can lie.** The correction uses the
ephemeris named *in the par file*, but the output header is stamped with the value of
`--ephem`. If the two disagree, the file claims an ephemeris it was not computed with.

## Correctness

**`get_latest_clock_file` raises `UnboundLocalError` on a network failure** instead of
falling back to a local clock file, because `fname` is never assigned on that path.

**`interpolate_clock_function` returns a validity mask that its only caller discards.**
`nustar_clock_correction_fun` then builds an Akima spline from a full-length `x` and a
possibly shorter `y`, which raises a length mismatch whenever an event falls outside
the clock table's span.

**Mutating PINT's global timing model leaks between calls.** Without a `.par` file the
code sets `RAJ`/`DECJ`/`EPHEM` on PINT's module-level `StandardTimingModel`
*instance*, so the coordinates persist into every later call in the same process and
into any other PINT user.

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
`NU_FINE_CLOCK` extension. Older clock files carry a `CLOCK_CORRECT` extension with
C0/C1/C2 polynomial coefficients instead, and cannot be used — including the
`dummy_clk.fits` file in the test data.

**Clock files are re-downloaded every run.** `get_latest_clock_file` scrapes the CALDB
HTML index and downloads the newest clock file each time, with no caching. NuSTAR clock
files are ~11 MB.

## Performance

**The TOA grid spans the whole orbit file, not the events.** A 5-second grid across a
multi-day `.attorb` file, or across a stack of orbit files, costs tens of thousands of
PINT TOAs that are then never used. Clipping the grid to `TSTART`/`TSTOP` plus a margin
is the single biggest available speed-up.

**The clock correction is interpolated twice.** Hermite interpolation onto a 1-second
grid, then an Akima spline from that grid onto the events. One interpolation evaluated
directly at the event times would be both faster and more accurate.

**The whole file is materialised in memory.** A 600 MB event file is read, modified and
written as one `HDUList`.

**`numba` is imported unconditionally** halfway down the module and compiles eagerly,
for a single small interpolation routine.

## Packaging

**`pyproject.toml` declares no runtime dependencies at all.** `pip install barycenter`
installs nothing the package actually imports.
