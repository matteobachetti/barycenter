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

## Correctness

**`TELAPSE` is left stale.** `correct_times` updates `TIME`, `START`, `STOP`, `TSTART`
and `TSTOP` but not `TELAPSE`, which is `TSTOP − TSTART` and changes by as much as the
two ends' corrections differ — 1.6 s on the XMM test observation, where the Roemer delay
moves by that much over 7.6 h, and 1.0 s on the Chandra one. SAS `barycen` does update it. Nothing downstream in this
package reads `TELAPSE`, but a file leaving here claims a duration that no longer matches
its own start and stop times.

**A Swift time within ~15 s of a leap second can be a second out.** The leap-second term
is a step function, and it is evaluated on the raw MET, where the step falls at the MET of
the leap instant (457401600 for 2015-07-01). The clock file's own −1 s step falls at the
leap instant *in onboard-clock time*, which is 14.8 s later because the UTCF is −14.8 s
there, minus the one second being inserted: `TSTART = 457401613.791`. Between the two the
correction and the table disagree by a second. Evaluating the leap term on the
clock-corrected time instead would narrow the window from ~15 s to the inserted second
itself, which is genuinely ambiguous and cannot be narrowed further — but the clock term
must stay on the raw MET (evaluating its polynomial 4 s late costs 214 ns), so the two
terms would no longer share an argument, and `correct_times` deliberately gives them the
same one. No reference exists near a leap second to settle it, and Swift observations
straddling one are rare, so this is recorded rather than guessed at. Everything more than
15 s from a leap second is unaffected, including both committed references.

**Fermi may need Swift's leap-second term, and nothing checks.** Fermi's `MJDREF` is
51910.00074287037 — bit for bit Swift's, whose fractional part encodes TT − UTC at
2001-01-01. On Swift that is the signature of a MET counting *UTC* seconds, and leaving the
leap seconds out costs 4 s on a 2015 observation. `MISSIONS["fermi"].met_is_utc` is False,
matching every mission but Swift, but Fermi has never been compared against an official
tool here, so this is untested either way and the error, if it is one, is a whole number of
seconds. Resolving it needs an FT1/FT2 pair and a `gtbary` reference.

## Missing features

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
