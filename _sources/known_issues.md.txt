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

It used to be 4.4× (3.76 GB on the same file) and the grid removed the difference, because
on the exact path `native.barycentric_correction` runs over every event at once.
`tools/benchmarks/bench_memsplit.py` attributes that per phase, at a million events:

| phase | peak |
|---|---|
| **astropy's JPL ephemeris** (`get_body_barycentric_posvel` + `get_body_barycentric`) | **865 MB** |
| our own vector arithmetic | 104 MB |
| `met_to_time` | 89 MB |
| unpacking `.xyz.to_value(u.m).T` | 72 MB |
| spacecraft interpolation | 24 MB |
| `erfa.dtdb` | 8 MB |
| the whole function in one go | 905 MB |

So on the exact path the memory is **astropy's ephemeris evaluation**, 865 MB of the
905 MB total, and the grid helps simply by asking it for 16 560 points instead of three
million. The obvious way to cap it without a grid is to **evaluate the correction in
chunks** of a few hundred thousand events — the correction is pointwise in time, so
chunking is exact, and it would bound the peak regardless of file size. That is a smaller
and safer change than restructuring the FITS write path, and it is the one to make first
if a user hits a memory limit on `--dt 0`.

**A fused `numba` kernel over our own arithmetic is *not* worth including — measured.**
This was worth checking, because numba is 6× faster than numpy on the one kernel it
already has and bit-identical, so more of it sounded attractive.
`tools/benchmarks/bench_split.py` settles it: the arithmetic a kernel could replace — the
`einsum`, the matrix products, the `log1p` — is **0.6 % of the per-event cost** (0.042 s
of 6.9 s at a million events), against 60 % for astropy's JPL ephemeris and 40 % for
`erfa.dtdb`, neither of which numba can touch. On memory it is 104 MB inside an 865 MB
envelope it does not control. A perfect fusion would therefore buy under 1 % of the
runtime and about a tenth of the peak, for a compiled dependency and a second code path
to keep bit-identical. `numba` stays where it is: lazily imported, an optional `[speed]`
extra, earning its keep only on the clock kernel and only on very large files.
