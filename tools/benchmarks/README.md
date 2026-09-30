# Benchmarks behind the Step 12 performance work

Machine: Apple Silicon, 32 GB RAM. Env `~/mamba/envs/py313` (arm64), numpy 2.4.6.
Test data: `tests/data/dummy_orb.fits.gz` (82800 samples, 82799 s span),
`dummy_evt.evt`, `dummy_fine_clk.fits`. Source RA=294.9107 Dec=21.58308, DE440.

Scripts here, all runnable as `PYTHONPATH=$REPO/src python <script>` from anywhere;
they find `tests/data` relative to themselves, or take `BARY_DATADIR`. None of them
touch the network, and only `make_big.py` writes anything (pass it an output path).

`bench_metrange.py`, `bench_margin.py` and `bench_edge.py` need PINT. PINT logs every
TOA array at DEBUG, so the last two silence `loguru` on import; redirect stderr if you
run the first one.

| script | question |
|---|---|
| `bench_numba.py` | numba `@vectorize` vs plain numpy on `_cubic_interpolation` |
| `bench_numpy_horner.py` | does avoiding numpy temporaries close the gap? |
| `bench_pipeline.py` | where the time goes: native engine, `dt` spline, clock interp |
| `bench_metrange.py` | what clipping the PINT TOA grid is worth |
| `bench_margin.py` | how much grid margin a clipped spline needs |
| `bench_edge.py` | is the clipping error an edge effect? |
| `bench_split.py` | how much of the per-event cost a numba kernel could reach |
| `bench_memsplit.py` | which phase of the native engine holds the memory |
| `make_big.py` | blow `dummy_evt.evt` up to N events for the memory test |

## 1. numba

`@vectorize` on `_cubic_interpolation` (clock.py:192), imported at module level with an
explicit signature, so it compiles **eagerly at import**.

Import cost breakdown (`barycenter.clock` is ~1.1 s total):

| | s |
|---|---|
| numpy + astropy.io.fits + astropy.table | 0.227 |
| `from numba import vectorize` | 0.127 |
| the eager `@vectorize` compile | 0.520 |

So **numba is 0.65 s of the 1.1 s import**, paid by every run including the many that
never touch a clock file (XMM, Chandra, `--clockfile none`).

Throughput, bit-identical results (max|diff| exactly 0.0 at every size):

| N | numba | naive numpy | horner numpy | ratio |
|---|---|---|---|---|
| 1 (scalar) | 1.73 us | 0.18 us | — | numpy 10x *faster* |
| 1e4 | 0.010 ms | 0.051 ms | 0.050 ms | 5.1x |
| 1e6 | 2.19 ms | 14.9 ms | 14.0 ms | 6.4x |
| 1e7 | 23.9 ms | 141 ms | 137 ms | 5.7x |

Horner form with in-place ops does **not** help: memory-bandwidth bound (7 input arrays).
numba wins 6x on arrays because it fuses the expression into one pass.

**Break-even: 0.65 s import / 0.013 s per million events = ~54 million events**, and the
clock interpolation is only 0.3-0.4 % of a current run (see below).

## 2. Where the time goes (native engine, no clock file)

| events | exact | us/event | dt=5 spline eval | clock interp | clock share |
|---|---|---|---|---|---|
| 1e3 | 0.007 s | 7.31 | 0.1 ms | 0.4 ms | 4.9 % |
| 1e4 | 0.065 s | 6.49 | 0.3 ms | 0.2 ms | 0.35 % |
| 1e5 | 0.663 s | 6.63 | 1.2 ms | 1.9 ms | 0.28 % |
| 1e6 | 6.72 s | 6.72 | 7.9 ms | 27.0 ms | 0.40 % |

`dt=5` spline build over the whole 82799 s orbit: **0.133 s** (16560 grid points).
The first clock call is reported separately by the script, because since numba became
lazy that call is where its ~0.8 s compile now lands; charging it to whichever row ran
first made the table read 99 % clock.
So 1e6 events: **6.72 s exact vs 0.14 s splined = 48x**, for
**mean +0.002 ns, std 0.389 ns, max|.| 1.629 ns**.

`read_orbit` on the 82800-sample file: 64 ms.

## 3. PINT `met_range`

82799 s orbit = 16560 TOAs at dt=5. A short snapshot inside it:

| observation | TOAs | no met_range | with met_range | speedup |
|---|---|---|---|---|
| 300 s | 63 | 3.89 s | 0.12 s | 33x |
| 3000 s | 603 | 3.50 s | 0.22 s | 16x |
| 30000 s | 6003 | 3.66 s | 1.45 s | 2.5x |

**But one grid step of margin is not enough.** Clipped vs unclipped, inside the span:

| margin | pint | native |
|---|---|---|
| 1 step | **46.222 ns** | 0.045 ns |
| 2 steps | 0.000 ns | 0.013 ns |
| 4+ steps | 0.000 ns | 0.013 ns |

**These are pre-fix numbers.** Both engines now pad a clipped grid by `2 * dt`
themselves, so `bench_margin.py` and `bench_edge.py` report 0.000 ns for every row --
which is the fix working, not the measurement failing. To reproduce the 1-step row,
change `2 * dt` back to `dt` in `native.native_barycentric_correction` and
`pintengine.pint_barycentric_correction`.

`bench_edge.py` localises it: the whole difference is in the **last third** of the probe,
i.e. at the end of the clipped grid — it is the cubic spline's not-a-knot end condition,
not an interpolation error. Both engines currently pad by `dt`; **the pad must be `2*dt`.**
The native engine barely notices because its knots hold exact values, while PINT's carry
its own noise, which the end condition amplifies.

## 4. Memory

`dummy_evt.evt` blown up to 3,000,000 events = **858 MB** (286 B/event — that file has
many columns). Full `main_barycenter` run, `--clockfile none`:

- wall **23.1 s** (of which ~20 s is 3e6 x 6.7 us of ephemeris evaluation)
- **maximum resident set size 3.59 GB** = **4.2x the file size**
- peak memory footprint 2.72 GB

Extrapolating: a 2 GB event file would want ~8.4 GB of RAM.

## 5. After the change: end-to-end, same 858 MB / 3e6-event file

Controlled comparison, identical code, `--clockfile none`:

| | wall | peak RSS | x file size |
|---|---|---|---|
| `--dt 0` (exact, per event) | 23.0 s | 3.76 GB | 4.4x |
| `--dt 5` (the new default above 1e5 events) | **2.3 s** | **1.98 GB** | 2.3x |

**10x faster and half the memory.** The memory halving answers the "where does the 4.2x
go" question: roughly half of it was never astropy's read-modify-write at all, it was
`barycentric_correction`'s own temporaries. That function allocates five (N,3) float64
arrays — `pos_sc`, `r_earth`, `v_earth`, `r_sun`, `r_obs` — which at 3e6 events is
5 x 3e6 x 3 x 8 B = 1.7 GB, matching the 1.8 GB that disappeared. Evaluating on a 16560-
point grid instead makes them negligible. The remaining 2.3x is the read-modify-write
(one input copy + one output copy), which is about the floor for that approach.

Accuracy on those 3e6 real events, gridded vs exact:

- **99.03 % bit-identical**
- max|diff| 29.802 ns = **exactly one float64 step** at this MET
- mean +0.0046 ns, std 2.936 ns (the std is the quantisation, not interpolation error)

So at the precision a FITS float64 time column can store, the grid is indistinguishable
from the exact path.

## 6. Would a fused numba kernel be worth it? No.

`bench_split.py` times `barycentric_correction` phase by phase, separating what a numba
kernel could replace (our own vector arithmetic) from what it could not (astropy `Time`
construction, the JPL ephemeris, `erfa.dtdb`):

| N | orbit + Time | JPL ephem | erfa.dtdb | our maths | ours % |
|---|---|---|---|---|---|
| 1e5 | 0.006 s | 0.429 s | 0.275 s | 0.006 s | 0.8 % |
| 1e6 | 0.076 s | 4.058 s | 2.707 s | 0.042 s | **0.6 %** |

`bench_memsplit.py` does the same for peak memory with `tracemalloc`, at 1e6 events:

| phase | peak |
|---|---|
| **astropy JPL ephemeris** | **865 MB** |
| our own vector arithmetic | 104 MB |
| `met_to_time` | 89 MB |
| unpacking `.xyz.to_value(u.m).T` | 72 MB |
| spacecraft interpolation | 24 MB |
| `erfa.dtdb` | 8 MB |
| whole function in one go | 905 MB |

**Conclusion: do not write it.** A perfect fusion of our arithmetic would buy under 1 %
of the runtime and about a tenth of a peak it does not control, in exchange for a
compiled dependency and a second code path to keep bit-identical.

**This also corrects an earlier claim in these notes and in the docs.** Section 5 first
attributed the memory that vanished with the grid to `barycentric_correction`'s own
`(N, 3)` temporaries. It is not: those are 104 MB of a 905 MB peak. The memory is
astropy's ephemeris evaluation, and the grid helps only by asking it for 16 560 points
instead of three million. If a user hits a memory limit on `--dt 0`, the fix is to
**evaluate the correction in chunks** of a few hundred thousand events -- exact, since
the correction is pointwise in time, and it bounds the peak whatever the file size.
