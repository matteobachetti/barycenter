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

**`TIERRELA` is not written.** HEASOFT `barycorr` writes the *relative* clock accuracy
alongside `TIERABSO` — 1e-9 for NuSTAR — and writes it even with `clockfile=NONE`. We
write neither, because nothing here measures a relative accuracy and copying a constant
out of a reference file for a quantity we do not compute would be asserting something we
have no evidence for. `TIERABSO`, which we do write, is documented in
[technical_details](technical_details.md).

**`CLOCKAPP` disagrees with `barycorr` on Swift with no clock file.** `barycorr` sets
`CLOCKAPP = T` for a Swift run with `clockfile=NONE`; we set `F`, because no clock file
was applied. The argument for HEASOFT's choice is that the UTCF already in the header has
been folded in, which is a clock correction of a sort; the argument for ours is that the
keyword should say whether the run did what it was asked not to do. Nothing downstream is
known to depend on it, so it is recorded rather than changed.

**Only the new NuSTAR clock format is read.** `nustar_clock_correction_fun` reads the
`NU_FINE_CLOCK` extension. Older files carry a `CLOCK_CORRECT` extension with C0/C1/C2
polynomial coefficients, documented by HEASOFT as accurate only to the millisecond;
they are refused with an explanatory error rather than supported, on the grounds that a
millisecond is four orders of magnitude worse than the target. Fetch a current clock
file from the CALDB instead.

## Performance

**The whole file is still materialised in memory**, at about **2.3× its size**: an 858 MB
event file peaks at 1.98 GB of resident memory. It is read, modified and written as one
`HDUList`, so that factor is one input copy plus one output copy, which is the floor for
that approach. Fixing it means correcting and writing one extension at a time.

It used to be 4.4× (3.76 GB on the same file), and *where the other half went* is worth
recording, because it was not astropy: it was `native.barycentric_correction`'s own
temporaries. That function allocates five `(N, 3)` float64 arrays — `pos_sc`, `r_earth`,
`v_earth`, `r_sun`, `r_obs` — which at 3 million events is 1.7 GB, matching the 1.8 GB
that disappeared when the grid was introduced. Evaluating on a 16 560-point grid instead
of 3 million events makes them negligible, so the largest single contributor was removed
for free. On a run forced to `--dt 0` the 4.4× is still there.

**`numba` would be worth its import cost if more of the hot path used it.** It is now an
optional extra, imported lazily, because compiling its one kernel eagerly cost 0.52 s of
a 1.09 s module import while that kernel is under half a per cent of a run — it does not
pay for itself below roughly 50 million events. But the conclusion to draw is not "numba
is not useful here". The reason it cannot earn its keep today is that the hot path is not
ours: at 6.7 µs per event the native engine spends nearly all its time inside astropy's
and ERFA's JPL ephemeris evaluation. What *is* ours is the vector arithmetic in
`native.barycentric_correction` — an `einsum`, a matrix product, a `log1p` and the half
dozen `(N, 3)` temporaries described above — all memory-bound numpy that one fused kernel
would do in a single pass and a fraction of the memory. Doing that would speed up the
exact path *and* shrink the 2.3× above, and would justify the import several times over.
Measured first: numba is 6× faster than numpy on the existing kernel and bit-identical.
