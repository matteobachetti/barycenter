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

1. The orbit file's `TELESCOP` keyword and `MJDREF` are read (via stingray's
   `high_precision_keyword_read`, which reconstructs `MJDREF` from `MJDREFI`+`MJDREFF`
   without losing digits).
2. `pint.observatory.satellite_obs.get_satellite_observatory` registers the spacecraft
   with PINT as a moving observatory. This is where [`monkeypatch.py`](#the-monkey-patch)
   comes in: it replaces PINT's `load_orbit` so that missions PINT does not know about
   (NuSTAR, SVOM, ...) can be read, and so that several orbit files can be given at once.
   PINT fits cubic splines through the position table; the correction is only valid
   where those splines are.
3. The spline knots give the valid time span. A **grid of TOAs is laid down every
   5 seconds** across that whole span, one `pint.toa.TOA` object per grid point.
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
| `barycenter.py` | Everything: CLI, both workflows, the correction, the clock correction, region extraction, downloads. |
| `utils.py` | FITS I/O that also works on `http(s)://` and `s3://` URLs (`fits_open_including_remote`), column slimming (`slim_down_hdu_list`), and HTML directory listing for the CALDB scrape. |
| `monkeypatch.py` | Replaces two PINT internals so PINT can read orbit files it does not natively support. Imported for its side effects by `barycenter.py`. |

(the-monkey-patch)=
## The monkey patch

`monkeypatch.py` replaces, at import time:

- **`pint.observatory.satellite_obs.load_orbit`** — PINT's version handles Fermi FT2,
  NICER/RXTE/IXPE `FPorbit`, and a couple of others. Ours adds NuSTAR and SVOM, accepts
  a *list* of orbit file names (or an `@metafile`) and stacks them, and applies
  `FPorbit`'s cleanup (sort by time, drop duplicate and zero rows) to every mission
  rather than only to `FPorbit` files.
- **`pint.observatory.satellite_obs.SatelliteObs._check_bounds`** — PINT refuses to
  extrapolate outside the orbit table. In practice an orbit file often stops a fraction
  of a second short of the last event, so ours warns instead of raising.

Every loader in the file is the same function with different column names and units:

| Mission | Extension | Position column | Velocity column | Units |
|---|---|---|---|---|
| Fermi | `SC_DATA` | `SC_POSITION` | `SC_VELOCITY` | m |
| NuSTAR | 1 | `POSITION` | `VELOCITY` | km |
| SVOM | 1 | `POSITION` | `VELOCITY` | m |
| NICER, RXTE, IXPE | `ORBIT` | scalar `X`,`Y`,`Z` | scalar `Vx`,`Vy`,`Vz` | m |

Monkey-patching a library's internals is fragile — it breaks whenever PINT
reorganises — and it is global, so importing `barycenter` changes PINT's behaviour for
everything else in the process. It is on the list to remove.

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
