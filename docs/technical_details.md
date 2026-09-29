# Technical details

This page describes how `barycenter` actually works: what happens, in what order,
which module does it, and where the answer can go wrong. It documents the code as it
is today, not as it is planned to be.

## What the tool is for

An X-ray satellite timestamps each photon with its own onboard clock, in seconds since
a mission-specific reference epoch (the *mission elapsed time*, MET). Two things make
those times unusable for timing science directly:

1. The spacecraft is moving. Over a year the Earth's orbit alone changes the light
   travel time to a source by up to ±500 s, and the spacecraft's own orbit adds a few
   tens of milliseconds on top.
2. Clocks tick at different rates depending on where they are. A clock on Earth and a
   clock at the solar-system barycentre disagree by up to ~1.7 ms over a year.

Barycentring removes both, giving the arrival time the photon *would* have had at the
solar-system barycentre, measured in Barycentric Dynamical Time (TDB). That is the
time you can safely fold on a pulsar period.

The official tools that do this are HEASOFT `barycorr`, XMM-SAS `barycen` and CIAO
`axbary`. Each needs a multi-gigabyte software installation, and each brings its own
limitations. This package does the same arithmetic in Python.

## The two workflows

`main_barycenter` (the `barycenter` command) parses arguments and calls
`apply_barycenter_correction`, which then branches in one of two directions.

### 1. The pure-Python workflow (the default)

This is the one that matters. It is `apply_barycenter_correction` in
[`barycenter.py`](../src/barycenter/barycenter.py), and it runs as follows.

**Step 1 — fetch the inputs.** `download_locally` pulls the event and orbit files if
they are `http(s)://` or `s3://` URLs. On SciServer (detected from the
`SCISERVER_USER_ID` environment variable or a `/home/jovyan` home directory) the HEASARC
archive is already mounted under `/FTP`, so nothing is downloaded.

**Step 2 — optional region cut.** If `--source-region-deg` is given,
`extract_events_in_region` keeps only the events inside a circle on the sky, using the
`X`/`Y` sky columns and the WCS keywords in the event header. Doing this first means
everything after it works on a much smaller file.

**Step 3 — decide where the source is.** The correction depends entirely on the source
direction: an error of 0.1 arcsec is 172 µs of light travel time, which is a thousand
times our accuracy target. The position comes from, in order of preference:

- a TEMPO/TEMPO2/PINT `.par` file given with `-p`, read by `pint.models.get_model`;
- explicit `--ra` and `--dec` on the command line;
- the event header, via `get_coordinates_from_fits_header`, which tries
  `RA_OBJ`/`DEC_OBJ`, then `RA_NOM`/`DEC_NOM`, then `RA_PNT`/`DEC_PNT`.

:::{warning}
That last fallback order is *not* the same as HEASOFT's. `barycorr` prefers
`RA_NOM`/`DEC_NOM`, we prefer `RA_OBJ`/`DEC_OBJ`, and the two keywords routinely differ
by a fraction of an arcsecond because `RA_OBJ` is the catalogue position of the target
while `RA_NOM` is the rounded pointing. On the NuSTAR test file the difference is
0.1 arcsec, which produced a 172 µs disagreement until the coordinates were pinned
explicitly. When comparing against an official tool, always pass `--ra` and `--dec`.
:::

Without a `.par` file the code builds a timing model by mutating PINT's module-level
`StandardTimingModel` object, setting `RAJ`, `DECJ`, `DM = 0` and `EPHEM`.

**Step 4 — build the barycentric correction function.** This is
`get_barycentric_correction`, and it is the heart of the package:

1. [`orbit.read_orbit`](#the-orbit-reader) reads the orbit file (or the list of them)
   into one cleaned table, taking `MJDREF` from `MJDREFI`+`MJDREFF` in extended
   precision.
2. [`pintengine.TableSatelliteObs`](#the-pint-engine) registers that table with PINT as
   a moving observatory. PINT fits cubic splines through the position columns; the
   correction is only valid where those splines are.
3. A **grid of TOAs is laid down every 5 seconds** across the span, one
   `pint.toa.TOA` object per grid point. With `met_range` the grid is clipped to the
   events plus one step of margin.
4. `model.get_barycentric_toas(ts)` computes, for each grid point, the barycentric
   arrival time. Internally that is `tdbld - delay`, where PINT's delay chain contains
   the geometric (Roemer) delay, the solar-system Shapiro delay, and whatever else the
   timing model happens to have switched on.
5. The difference (barycentric MJD − spacecraft MJD), converted to seconds, is wrapped
   in a `scipy.interpolate.Akima1DInterpolator` against MET, with extrapolation
   enabled. That interpolator is the returned `bary_fun`.

So the correction is *never* evaluated at the event times themselves; it is evaluated
on a 5 s grid and interpolated. The Akima spline over a smooth 5 s-sampled function
contributes well under a nanosecond, so this is not a precision problem — but the grid
covers the whole orbit file rather than the events, which is a performance problem (see
[Known issues](known_issues.md)).

**Step 5 — build the clock correction function.** Only NuSTAR has one implemented. If
no `-c/--clockfile` is given and the mission is NuSTAR, `get_latest_clock_file` scrapes
the CALDB HTML directory index and picks the highest-versioned `nuCclock*.fits`.
`nustar_clock_correction_fun` then:

1. reads the `NU_FINE_CLOCK` extension (columns `TIME`, `CLOCK_OFF_CORR`,
   `CLOCK_FREQ_CORR`, `CLOCK_ERR_CORR`);
2. interpolates it onto a 1-second grid with `interpolate_clock_function`, which uses
   `cubic_interpolation` — a numba port of the `cubeterp` routine from HEASOFT's
   `seekinterp.c`, so that we reproduce HEASOFT's Hermite interpolation exactly;
3. wraps *that* grid in a second Akima spline.

Passing `--clockfile none` skips this entirely.

**Step 6 — correct the times.** Before anything is added, two header quantities are
folded in: `TIMEZERO`, and the half-bin shift `(0.5 - TIMEPIXR) * TIMEDEL` that moves a
timestamp from the start of its bin to its centre. Then `correct_times` computes

```
t_out = t_in + clock_fun(t_in) + bary_fun(t_in)
```

for every `TIME`, `START`, `STOP`, `TSTART` and `TSTOP` **column**, and every `TSTART`
and `TSTOP` **keyword**, in *every* HDU of the file. Doing all the HDUs matters: GTIs
left behind on the spacecraft clock while the events move to the barycentre would
silently truncate up to ~500 s of data.

:::{note}
**The order in which the clock correction is applied differs from `barycorr`'s.**
We evaluate both `clock_fun` and `bary_fun` at the raw mission time and add the two.
HEASOFT `barycorr` corrects the clock *first* and then evaluates the barycentric
correction — and the spacecraft position lookup — at the clock-corrected time. On
NuSTAR, where the clock correction reaches a few milliseconds, the two orders differ by
about **1.1 µs**, well above the 100 ns target. Matching `barycorr` means evaluating
`bary_fun(t_in + clock_corr)`. This has not been changed yet; it is why the reference
test runs with `--clockfile none`. See [Known issues](known_issues.md).
:::

**Step 7 — stamp the headers and write.** Every HDU gets `TIMESYS = TDB`,
`TIMEREF = SOLARSYSTEM`, `TREFPOS = BARYCENTER`, `TREFDIR = RA_OBJ,DEC_OBJ`,
`TIMEZERO = 0`, `PLEPHEM = JPL-<ephem>`, `RADECSYS`, `CLOCKAPP`, the coordinates
actually used in `RA_OBJ`/`DEC_OBJ`, and a `HISTORY` block naming the orbit file, clock
file, par file, ephemeris and frame. The whole `HDUList` is then written out in one go.

### 2. The `--apply-official` workflow

`apply_mission_specific_barycenter_correction` shells out to the mission's own tool
instead of computing anything:

- NuSTAR, NICER, RXTE, Swift and Chandra go through `official_barycorr`, which calls
  `heasoftpy.barycorr` in a temporary working directory (HEASOFT tools are sensitive to
  the current directory and to `PFILES`).
- ASCA goes through a `timeconv` call, with `earth.dat` and `frf.orbit.255` downloaded
  from HEASARC first.

This path exists for cross-checking and for missions we do not yet implement natively.
It requires a working HEASOFT installation and is not exercised in CI.

## Module map

| Module | Role |
|---|---|
| `barycenter.py` | CLI, both workflows, the clock correction, region extraction, downloads. |
| `orbit.py` | The mission-agnostic orbit file reader: one `OrbitSpec` per mission, one table out. |
| `native.py` | The default engine: the correction from astropy + ERFA + a JPL ephemeris. |
| `pintengine.py` | The optional PINT engine, for `.par` models and as an independent cross-check. |
| `utils.py` | FITS I/O that also works on `http(s)://` and `s3://` URLs (`fits_open_including_remote`), column slimming (`slim_down_hdu_list`), HTML directory listing for the CALDB scrape, and the `MJDREFI`+`MJDREFF` reader. |

(the-orbit-reader)=
## The orbit reader

Every mission tabulates the same three things — time, geocentric position, geocentric
velocity — and differs only in the dialect. So [`orbit.py`](../src/barycenter/orbit.py)
has one reader driven by a declarative `OrbitSpec`, and adding a mission is a single
entry in `ORBIT_SPECS`:

| Mission | Extension | Position column | Velocity column | Units in the file |
|---|---|---|---|---|
| Fermi | `SC_DATA` | `SC_POSITION` | `SC_VELOCITY` | m |
| NuSTAR | 1 | `POSITION` | `VELOCITY` | **km** |
| SVOM | 1 | `POSITION` | `VELOCITY` | m |
| NICER, RXTE, IXPE | `ORBIT` (`XTE_PE` for RXTE) | scalar `X`,`Y`,`Z` | scalar `Vx`,`Vy`,`Vz` | m |

The units are declared in the spec, not read from `TUNITn`, because orbit files are
unreliable about that keyword and getting the factor of 1000 wrong is a 20 ms error.
The table that comes out is always in metres and metres per second.

Three cleanups are applied to **every** mission, not just to `FPorbit` files as PINT
does: sort by time, drop rows repeated at the same time (a spline through a repeated
abscissa is undefined, and repeats are common where two files overlap), and drop
all-zero placeholder rows (a zero position is a 6400 km error).

The output table carries both time bases — `MJD_TT`, which is what PINT's
`SatelliteObs` expects, and `MET`, which is what the native engine uses. Reading the
orbit file once, in one place, is what makes the native-versus-PINT comparison
meaningful: the two engines cannot disagree about where the spacecraft was.

(the-pint-engine)=
## The PINT engine

PINT's `SatelliteObs` takes a *file name* and calls its own `load_orbit` on it. That is
not usable here: we need SVOM, which PINT has no loader for, remote `https://` and
`s3://` input, lists of orbit files, and the cleanup above on every mission.

The package used to obtain all of that by **monkey-patching** PINT at import time,
replacing `pint.observatory.satellite_obs.load_orbit` and
`SatelliteObs._check_bounds`. That has been removed, for three reasons:

- It was global. Importing `barycenter` changed PINT's behaviour for everything else in
  the interpreter — including silently disabling PINT's own extrapolation guard.
- It had already drifted. PINT has since grown its own NuSTAR loader, so part of the
  patch was dead weight that nobody noticed.
- It made the two engines read the orbit file through different code, which makes a
  comparison between them uninterpretable.

[`pintengine.TableSatelliteObs`](../src/barycenter/pintengine.py) does the same job by
subclassing: it takes an already-parsed table from `orbit.py`, calls
`SpecialLocation.__init__` directly, builds the same six splines, and overrides
`_check_bounds` to *warn* about a short extrapolation and *log an error* about one five
times longer, rather than raising. Nothing outside the class is modified.

Verified against PINT's own NuSTAR loader: the spacecraft position agrees to 5 mm. That
is not exact and cannot be — PINT indexes the orbit by absolute MJD in float64, whose
step at MJD 57263 is 0.6 µs, and 0.6 µs at 7.6 km/s is 5 mm. It is 17 picoseconds of
Roemer delay.

The remaining coupling is to PINT's internal attribute names (`FT2`, `X`…`Vz`,
`_geocenter`, `_maxextrap`). That is narrower than a monkey patch, not absent: if PINT
renames them this class breaks — but it breaks visibly, in one place, and only for the
optional engine.

## The native engine

[`native.py`](../src/barycenter/native.py) computes the correction directly from
astropy, ERFA and a JPL ephemeris, without PINT. It writes out four terms explicitly:

| Term | What it is | Size (NuSTAR test file) |
|---|---|---|
| Einstein | `(TDB − TT)` at the geocentre, ERFA's `dtdb` | ±1.7 ms |
| topocentric Einstein | `r_sc · v_earth / c²`, the observer's offset from the geocentre | −0.7 µs, varying by 4.2 µs per orbit |
| Roemer | `r_obs · n̂ / c`, light travel time to the barycentre | ±500 s |
| Shapiro | `2·T☉·ln(1 + cos θ)`, the Sun's gravity | +4.7 µs |

plus a parallax term if a source distance is given, which the official tools omit.

Each of these is a *small* number of seconds, computed in float64 (~1e-13 s). Nothing
is ever obtained by subtracting two absolute epochs, which is the whole reason the
PINT engine needs an 80-bit `longdouble` and this one does not.

### Two things it gets right that are easy to get wrong

**The Shapiro delay is only defined up to a constant**, because it is the logarithm of
a distance ratio and you must pick a reference distance. `axBary` normalises by the
observer's distance from the Sun, PINT by the astronomical unit. The two differ by
`2·T☉·ln(r☉/AU)`, an annual term of about 170 ns amplitude and 96 ns on the test file —
which is the entire disagreement between the PINT engine and `barycorr`. The native
engine writes it the `axBary` way by default; `shapiro="pint"` reproduces PINT's.

**The source direction must be in the ephemeris's own frame.** Reading a JPL kernel
applies no rotation, whatever frame astropy labels the result with: DE200 is referred
to the FK5 dynamical equinox of J2000, and DE405 onwards to the ICRF. So DE200 paired
with FK5 coordinates, or DE440 with ICRS coordinates, needs no rotation — and pairing
them the wrong way round costs **45 µs**. That is what HEASOFT's `refframe` parameter
is really selecting. `ephemeris_frame()` encodes the rule and
`source_unit_vector()` applies it.

### Measured against `barycorr`

On the committed NuSTAR file (937 events, DE440, ICRS, no clock correction):

| engine | mean | std | peak-to-peak | max abs |
|---|---|---|---|---|
| **native**, `axBary` Shapiro | **+22.0 ns** | 19.5 ns | 59.6 ns | **59.6 ns** |
| native, PINT Shapiro | +117.3 ns | 19.6 ns | 59.6 ns | 149.0 ns |
| native, no Shapiro | −4671.5 ns | 13.9 ns | 59.6 ns | 4708.8 ns |
| PINT engine (`py313-x64`) | +117.2 ns | 19.6 ns | 59.6 ns | 149.0 ns |

There is nothing left to chase in that 59.6 ns. The reference times are around
1.8e8 s, where one float64 step is **29.802 ns**, and the residual takes exactly three
values — 0, 1 and 2 of those steps. The reference file cannot express a finer
difference, so the true disagreement is under one step and most likely the +22 ns mean.
`test_residual_is_only_rounding_of_the_stored_times` asserts precisely this.

Native and PINT, with the Shapiro convention matched, agree to **mean −95.7 ns,
std 0.63 ns** — i.e. two independent implementations of the same physics agree to
sub-nanosecond, and the offset is the convention and nothing else.

### An independent cross-check

A NICER observation of the Crab barycentred with `barycorr` 2.17 using **DE200 and FK5**
— a different mission, a different orbit file format (`ORBIT` extension, scalar `X`/`Y`/`Z`
in metres at 10 s sampling), a different ephemeris and a different frame:

| | mean | std | peak-to-peak |
|---|---|---|---|
| native, coordinates read as FK5 (correct) | +52.7 ns | 7.4 ns | 14.9 ns = 1 ulp |
| native, coordinates read as ICRS (wrong frame) | −45049.8 ns | 18.8 ns | 44.7 ns |

Two incidental findings from that comparison:

* `barycorr` does **not** apply the `TIMEPIXR` half-bin shift, but this package does.
  On NICER that is 20 ns; on a mission with a coarser `TIMEDEL` it would matter much
  more.
* `barycorr` **does** apply `TIMEZERO` (−1 s here), and so must we.

### Speed

| | per event | 1e6 events |
|---|---|---|
| native (arm64) | 6.5 µs | 6.5 s |
| native (x86 under emulation) | 13.6 µs | 13.6 s |
| PINT, at the event times | 581 µs | ~10 min |

and the correction is smooth enough that it need not be evaluated per event at all.
Sampling it on a grid and interpolating with a cubic spline costs:

| grid step | max error |
|---|---|
| 1 s | 0.5 ns |
| 5 s | 1.2 ns |
| 20 s | 1.3 ns |
| 60 s | 2.6 µs |

So the 5-second grid the package already uses is a good choice, and the problem with
it is not its spacing but its extent — it spans the whole orbit file rather than the
events.

## Accuracy

The target is **100 ns** against the mission's own tool.

The tracked reference `tests/data/dummy_evt_bary_DE440_noclk.evt.gz` was produced by
HEASOFT `barycorr` 2.19 on the committed NuSTAR test file with every input pinned:
DE440, ICRS, `ra=294.9107 dec=21.58308`, and `clockfile=NONE`.
`tools/make_test_data.py` records the exact call and can regenerate it.

Measured agreement of the current (PINT) engine against that reference, on 937 events:

| Environment | mean | std | peak-to-peak | max abs |
|---|---|---|---|---|
| `py313-x64` (80-bit `longdouble`) | +116.3 ns | 19.2 ns | 59.6 ns | 149.0 ns |
| `py313` (arm64, 64-bit `longdouble`) | +118.4 ns | 254.6 ns | 1341.1 ns | 804.7 ns |

Two separate effects are visible there.

**The +116 ns offset** is PINT's solar-system Shapiro delay. PINT's formulation carries
an extra `2·T☉·ln(r/AU)` term (with `T☉ = GM☉/c³ = 4.9255 µs`) that `axBary` omits. The
term is an annual modulation of about 100 ns; subtracting it brings the mean residual to
**+21.5 ns**, comfortably inside target. Neither convention is wrong — they differ by a
constant-plus-annual term that is absorbed into a pulsar's spin parameters — but it must
be accounted for when comparing tools.

**The 1.3 µs scatter on arm64** is a precision floor, not a physics difference. PINT
represents absolute times as `numpy.longdouble` MJDs. On x86 that is the 80-bit
extended format, whose 64-bit mantissa gives ~0.54 ns at MJD 57263; on Apple Silicon and
Windows `numpy.longdouble` is just `float64`, whose 53-bit mantissa gives ~1.1 µs at the
same epoch. Subtracting two large MJDs to get a small correction therefore loses the
answer. This is a property of representing an absolute epoch, and disappears if the
correction terms — all of them small numbers of seconds — are computed directly rather
than as a difference of two big ones.

The test suite reflects this: it asserts 200 ns where extended precision is available
and 2 µs where it is not.

### Things that will move the answer by more than 100 ns

When a comparison disagrees, check these before looking for a bug:

| Difference | Size (NuSTAR test file) |
|---|---|
| `RA_OBJ` vs `RA_NOM` (0.1 arcsec) | 172 µs |
| DE430 vs DE440 ephemeris | ~10 µs |
| DE405 vs DE430 | 0.38 µs |
| Clock correction applied before vs after | 1.1 µs |
| PINT's extra Shapiro `2T·ln(r/AU)` term | ~100 ns |
| `numpy.longdouble` being float64 (arm64, Windows) | up to 1.1 µs of scatter |
| GPS→UTC correction on/off | ~0.1 ns |

## Supported missions

| Mission | Orbit file | Clock correction | Status |
|---|---|---|---|
| NuSTAR | `nu<obsid>A.attorb` | `nuCclock*.fits`, `NU_FINE_CLOCK` extension | validated to 100 ns |
| NICER | `ni<obsid>.orb` | none needed | works |
| RXTE | `orbit/FPorbit_*` | HEASOFT `tdc.dat` — **not implemented** | works, clock missing |
| IXPE, Swift | `FPorbit`-style | none needed | works |
| Fermi | FT2 | none needed | works |
| SVOM | `POSITION`/`VELOCITY` in m | to be determined | works |
| ASCA | — | — | `--apply-official` only |
| XMM-Newton | — | — | not supported |
| Chandra | — | — | `--apply-official` only |

## Test data

Everything in `tests/data/` is small on purpose, so that CI can check us against the
official tools without installing HEASOFT, SAS or CIAO.

| File | What it is |
|---|---|
| `dummy_evt.evt` | 937 NuSTAR FPMA events, MJDREF 55197.00076601852 |
| `dummy_orb.fits.gz` | the matching orbit file, 82800 rows at 1 s |
| `dummy_clk.fits` | a NuSTAR clock file with the *old* `CLOCK_CORRECT` extension — the current code reads `NU_FINE_CLOCK` and cannot use it |
| `dummy_par.par` | a minimal timing model for the same position |
| `dummy_evt_bary_DE440_noclk.evt.gz` | the `barycorr` reference described above |
