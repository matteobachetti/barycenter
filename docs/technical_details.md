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
[`core.py`](../src/barycenter/core.py), and it runs as follows.

**Step 1 — fetch the inputs.** `download_locally` pulls the event and orbit files if
they are `http(s)://` or `s3://` URLs. On SciServer (detected from the
`SCISERVER_USER_ID` environment variable or a `/home/jovyan` home directory) the HEASARC
archive is already mounted under `/FTP`, so nothing is downloaded.

A local path is never copied: it comes back as the absolute path of the file where it is,
with a relative path resolved from the caller's directory. It used to be copied into the
current directory, but only when no file of that name was there yet, so reprocessing an
observation silently barycentred the stale copy of the old one, and parallel runs sharing a
working directory read each other's copies. Nothing is ever written next to an input, which
may sit in a read-only directory; anything that has to change a file (the ASCA `timeconv`
path, which also gunzips) works on a private copy in the system temporary directory.
Downloaded `http(s)://` and `s3://` files do still land in the current directory (the
output's directory, on the ASCA path) and are reused when already there, since that is the
cache.

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
  `RA_OBJ`/`DEC_OBJ`, then `RA_TARG`/`DEC_TARG`, then `RA_NOM`/`DEC_NOM`, then
  `RA_PNT`/`DEC_PNT`, then plain `RA`/`DEC`. That order is a deliberate departure from
  HEASOFT's — see [below](#the-coordinate-keyword-order).

A `.par` file is the only thing on this path that needs PINT installed, since it is the
one source of coordinates we do not parse ourselves. Its `EPHEM` also wins over
`--ephem`, because the model was fitted with it, and the output header is stamped with
the one actually used.

**Step 4 — build the barycentric correction function.** This is
`get_barycentric_correction`, and it is the heart of the package. First,
[`orbit.read_orbit`](#the-orbit-reader) reads the orbit file (or the list of them) into
one cleaned table, taking `MJDREF` from `MJDREFI`+`MJDREFF` in extended precision. Then
one of two engines turns it into a callable:

- **`--engine native`** (the default) builds a cubic Hermite spline through the
  spacecraft positions and evaluates [the four terms](#the-native-engine) directly at
  whatever times it is asked for. No grid, no interpolation of the correction itself.
- **`--engine pint`** registers the table with PINT as a moving observatory via
  [`pintengine.TableSatelliteObs`](#the-pint-engine), lays down a **grid of TOAs every
  5 seconds**, and Akima-interpolates between them. PINT fits its own cubic splines
  through the position columns, so the correction is only valid where those splines are.
On the PINT path, `model.get_barycentric_toas(ts)` computes the barycentric arrival time
for each grid point — internally `tdbld - delay`, where PINT's delay chain contains the
geometric (Roemer) delay, the solar-system Shapiro delay, and whatever else the timing
model happened to switch on — and the difference (barycentric MJD − spacecraft MJD) in
seconds is wrapped in a `scipy.interpolate.Akima1DInterpolator` against MET.

So with `--engine pint` the correction is *never* evaluated at the event times
themselves. An Akima spline over a smooth 5 s-sampled function contributes well under a
nanosecond, so that is not a precision problem — but the grid covers the whole orbit
file rather than the events, which is a performance one (see
[Known issues](known_issues.md)). The native engine has no grid, and takes an optional
`dt` if one is ever wanted for speed.

**Step 5 — build the clock correction function.** `clock_correction_fun` is the single
entry point: it is given the mission, whatever the user passed as `-c/--clockfile` and the
`INSTRUME` keyword, and returns the correction function together with the file it came
from, or `(None, None)` for a mission with no clock correction of its own. Keeping that
decision in `clock.py` is what lets `core.py` stay mission-agnostic.

### NuSTAR

If no `-c/--clockfile` is given, `get_latest_clock_file` scrapes
the CALDB HTML directory index, picks the highest-versioned file matching that mission's
`ClockSource.pattern` (`nuCclock*.fits*` for NuSTAR) and caches
it under the user's cache directory — these files are about 12 MB, so they are fetched
once per machine, not once per run. If the index cannot be reached, the newest cached
file is used.

`nustar_clock_correction_fun` reads the `NU_FINE_CLOCK` extension (columns `TIME`,
`CLOCK_OFF_CORR`, `CLOCK_FREQ_CORR`, `CLOCK_ERR_CORR`) and returns a function that
evaluates `interpolate_clock_function` directly at whatever times it is asked for. That
in turn uses `cubic_interpolation`, a numba port of the `cubeterp` routine from HEASOFT's
`seekinterp.c`, so the Hermite interpolation through the tabulated offsets and their
derivatives is HEASOFT's own. On the test observation the correction is 20–29 ms, on a
grid sampled every 1000 s.

Clock files from before 2019 carry a `CLOCK_CORRECT` extension instead — a per-interval
C0/C1/C2 polynomial that HEASOFT documents as accurate only to the millisecond. Those are
**refused**, with an error naming the extension found: a millisecond is four orders of
magnitude worse than the target, and a file corrected with it would look clock-corrected
while being nothing of the kind.

### RXTE

RXTE's coefficients are not in a FITS file at all but in `tdc.dat`, 800 sets of quadratic
coefficients in free-format ASCII. HEASOFT's `barycorr` **ignores its own `clockfile`
parameter** for RXTE and always reads `$LHEA_DATA/tdc.dat`, so reproducing it means
reading the same file. A copy ships with this package, under `src/barycenter/data/`: RXTE
stopped observing in January 2012 and the file is final, it is 53 kB, and bundling it is
what makes an RXTE run work without HEASOFT installed. `TIMING_DIR` and `LHEA_DATA` are
still checked first, so an installation's own copy wins.

`read_tdc_file` is translated from
[`xCC.c`](https://heasarc.gsfc.nasa.gov/docs/xte/abc/xCC.c) by A. Rots, the reader
`axBary` itself uses, and the format is worth spelling out because nothing else documents
it. The file is a stream of rows of four numbers, in two kinds:

* a row whose fourth number is negative starts a block — its first number is the mission
  day the block's polynomials are measured from, its second the value of `TIMEZERO` there;
* any other row gives `C0`, `C1`, `C2` of a quadratic in days since that day, valid until
  the fourth number (also in days since that day).

The first set whose end is later than the time asked for wins. The original reader walks
the file from the top to find it; since the ends never go backwards, that is a
`searchsorted`. The trailing comment block is what terminates the file — the C reader's
`fscanf` loop stops at the first row that fails to parse, which is why the comments are at
the *end* of `tdc.dat` and not the beginning.

Two things are deliberately left out and one is added:

* `TIMEZERO` is not applied here. The event file's header already carries it, and step 6
  folds that in.
* Anything whose `INSTRUME` is not `HEXTE` — in practice the PCA — has a further 16 µs of
  detector delay subtracted, which is what `hdaxbary` does.
* HEASOFT evaluates the correction **once**, at the middle of the observation, and folds
  that one number into `TIMEZERO`. This package evaluates it at each event time instead.
  The coefficients drift by about 25 ns per hour, so on a short observation the difference
  is far below the float64 granularity of the stored times (119 ns at RXTE's 5.4e8 s) and
  on a long one ours is the better answer.

On the test observation the correction is 17.4 µs: small, but 170 times the target.

Passing `--clockfile none` skips all of this. Passing a clock file for RXTE warns and uses
`tdc.dat` anyway, which is also what HEASOFT does with that parameter.

### Swift

Swift's correction is not a fine clock correction at all but the **UTC correction factor**,
the offset between the onboard clock and UTC. It is tens of seconds — −15.56 s on the test
observation — where NuSTAR's is tens of milliseconds and RXTE's tens of microseconds, so a
sign error is not subtle.

The file is `swclockcor*.fits` in `swift/mis/bcf/clock/`, and the numbers live in a
`CLOCK_CORRECT` extension: one row per fit interval, with `TSTART`, `TSTOP` and quadratic
coefficients `C0`, `C1`, `C2`. The correction is

```
x = (met - TSTART) / 86400
correction = -(C0 + C1 x + C2 x^2) * 1e-6
```

microseconds, and negated, because the table gives the clock's excess over UTC while the
correction has to remove it. The sign was settled against the `UTCFINIT` keyword in the
event files themselves rather than reasoned about.

Three properties of the table are worth knowing, because the reader depends on all three:

* **The intervals are contiguous and gap-free** — each `TSTOP` is the next `TSTART`
  exactly — so the row containing a time is found with one `searchsorted` and a boundary
  time belongs to the later row.
* **The polynomial matters.** The clock drifts 4616 µs/day, which is 342 µs across the
  6.4 ks test observation. A reader that took `C0` and stopped would be within 100 ns for
  the first two seconds of an interval and 3400 times the target out by the end of a day.
* **The table steps by −1 s at each leap second**, because the UTCF's destination is UTC
  and UTC is what steps. That is the mirror image of the leap-second term described below,
  which steps +1 s because *its* destination is TT, and the two cancel.

A time the table does not cover raises, rather than extrapolating a quadratic off the end
of it — the same choice made for a NuSTAR observation predating its clock file. The message
says which way out the time fell and offers the two ways forward: fetch a newer CALDB file,
or run with `--clockfile none` and accept the 15 s.

**Step 6 — correct the times.** `TIMEZERO` is folded in first. `TIMEPIXR` deliberately is
not: this step used to add `(0.5 - TIMEPIXR) * TIMEDEL` as well, moving a timestamp from
the start of its bin to its centre. `barycorr` does not do that, and on RXTE PCA data,
where `TIMEDEL` is one 954 ns clock tick and `TIMEPIXR` is 0, it put every event 477 ns
away from the reference — five times the target, from a single line. It also left
`TIMEPIXR` itself untouched in the output header, so the file went on claiming a
convention its times no longer followed. Where a timestamp sits inside its bin is not the
barycentring tool's business. Then `correct_times` computes

```
t_clk = t_in + clock_fun(t_in) + leap_fun(t_in)
t_out = t_clk + bary_fun(t_clk)
```

for every `TIME`, `START`, `STOP`, `TSTART` and `TSTOP` **column**, and every `TSTART`
and `TSTOP` **keyword**, in *every* HDU of the file. Doing all the HDUs matters: GTIs
left behind on the spacecraft clock while the events move to the barycentre would
silently truncate up to ~500 s of data.

:::{important}
**The order matters, and it is not the obvious one.** The clock correction goes on first,
and the barycentric correction is then evaluated *at the clock-corrected time* — which
also means the spacecraft position is looked up there, since `bary_fun` interpolates the
orbit at whatever time it is given. That is what `barycorr` does.

Adding the two corrections independently, `t + clock(t) + bary(t)`, is what this package
used to do, and it is wrong by the clock correction times the rate of change of the
barycentric one. Measured against the reference generated with the clock file on:
**+1146 ns mean, 1878 ns peak**, against **+22 ns mean, 60 ns peak** for the correct
order. It is an order of magnitude above the target, from a line that looks harmless.
:::

### Missions whose MET counts UTC seconds

`leap_fun` above is present for exactly one mission, and is the reason the formula has
three terms rather than two.

Almost every mission counts its mission elapsed time in **TT** seconds, so `MJDREF` is all
you need: `MJD(TT) = MJDREF + MET/86400`. Swift counts **UTC** seconds. Its clock is
effectively held back by one second every time a leap second is inserted, so the number of
TT seconds that have really elapsed since the epoch is larger than the MET by however many
leap seconds fell in between. `utils.leap_seconds_since_mjdref` returns that difference,
measured from the file's own epoch — which is what `MJDREF` means. Swift's
`MJDREFF = 0.00074287037` is 64.184 s, TT − UTC on 2001-01-01, when TAI − UTC was 32 s;
for a December 2015 observation four more leap seconds have been inserted, so the term
is +4 s.

Two things about it are deliberate:

- **It is not a clock correction, and `--clockfile none` does not switch it off.** It is a
  time-system conversion, and it belongs to the file regardless of whether anyone has a
  clock model for it. Leaving it out puts a Swift observation four seconds from
  `barycorr` — forty million times the target — and a flag should not be able to cause
  that. It is opt-in per mission, through `Mission.met_is_utc`.
- **Both the clock term and the leap term are evaluated on the raw time.** Swift's clock
  correction drifts 4616 µs/day, so evaluating its polynomial 4 s later moves every event
  214 ns. That is above the target, and it is not what `barycorr` does.

The leap-second epochs come from ERFA's own table, converted to METs once, so an
observation straddling a leap second gets the step in the right place rather than rounded
to the nearest day — a whole second, silently, in the middle of a file.

:::{note}
**Fermi shares Swift's exact `MJDREF`** (51910.00074287037) and does *not* need this term.
That used to be an open question: the shared value was reason to suspect Fermi's MET also
counted UTC seconds, and being wrong about it is a whole-second error. The `gtbary`
reference settles it. The test observation sits 2.0 s of leap seconds after `MJDREF`, so
applying the term would put us 2 s away from `gtbary`; we are 29.8 ns away without it.
`met_is_utc` is False for Fermi and that is now measured rather than assumed.
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
- ASCA goes through a `timeconv` call. `timeconv` rewrites the file it is given and
  looks for its reference files by name in its current directory, so it runs in a private
  temporary directory holding a copy of the events (decompressed on the way if the input
  is gzipped) and links to `earth.dat` and `frf.orbit.255` under short names. That makes
  the run independent of where it was started, leaves nothing beside the output, and
  keeps every path short of the 80 characters at which FTOOLS truncate file names. The two
  reference files never change (ASCA stopped operating in 2001), so they are fetched once
  per machine into astropy's download cache by `cached_download`, which also repairs the
  broken cache entries that older versions of this package left for exactly these files.

This path exists for cross-checking and for missions we do not yet implement natively.
It requires a working HEASOFT installation and is not exercised in CI.

## Module map

| Module | Role |
|---|---|
| `cli.py` | Argument parsing, the default output file name, and `main_barycenter`. |
| `core.py` | The mission-agnostic workflow: `apply_barycenter_correction`, `correct_times`, region extraction. |
| `orbit.py` | The mission-agnostic orbit file reader: one `OrbitSpec` per mission, one table out. |
| `native.py` | The engine: the correction from astropy + ERFA + a JPL ephemeris. |
| `pintengine.py` | The optional PINT engine, for `.par` models and as an independent cross-check. |
| `clock.py` | Spacecraft clock corrections: NuSTAR's CALDB fine clock files, RXTE's `tdc.dat`, the CALDB fetcher, and `clock_correction_fun`, which looks up which applies. |
| `missions.py` | The `MISSIONS` registry: the only module that knows anything mission-specific. |
| `official.py` | Shelling out to HEASOFT `barycorr` and `timeconv` under `--apply-official`. |
| `remote.py` | `download_locally`: local paths, `https://` and `s3://`; `cached_download` for files that never change. |
| `utils.py` | FITS I/O that also works on `http(s)://` and `s3://` URLs (`fits_open_including_remote`), column slimming (`slim_down_hdu_list`), HTML directory listing for the CALDB scrape, the `MJDREFI`+`MJDREFF` reader, and `splitext_improved`. |

Until this release all of that lived in one 979-line `barycenter.py`. Nothing outside
the package needs to change: `main_barycenter` and the other public names are still
importable from `barycenter` itself, and the `barycenter` command is unaffected. Code
that imported from `barycenter.barycenter` has to be updated.

(the-mission-registry)=
## The mission registry

Everything mission-specific lives in one dict in
[`missions.py`](../src/barycenter/missions.py). Nothing else in the package branches on a
mission name — `core.py` does not mention one, and `clock.py` and `official.py` only look
entries up.

```python
MISSIONS["nustar"] = Mission(
    name="nustar",
    telescop=("nustar",),
    orbit=OrbitSpec(pos="POSITION", vel="VELOCITY", pos_unit=u.km, vel_unit=u.km / u.s),
    clock=nustar_clock_builder,
    official="barycorr",
)
```

| field | meaning |
|---|---|
| `name` | canonical short name; must match the dict key |
| `telescop` | lower-case *substrings* of the `TELESCOP` keyword that identify the mission — substrings because the keyword is written `XTE` and `RXTE`, `NuSTAR` and `NUSTAR`, `AXAF` and `CHANDRA` |
| `orbit` | an `OrbitSpec`, or `None` for no native reader |
| `clock` | `clock(clockfile, instrument) -> (function, path)`, or `None` for a mission that needs no correction |
| `official` | `"barycorr"`, `"timeconv"` or `None` — which branch of `official.py` `--apply-official` takes |
| `official_ephem` | the only ephemeris that tool can manage, if it is stuck on one: DE200 for ASCA's `timeconv`, DE405 for CIAO's `axbary`. The native engine has no such limit, which is the main reason to prefer it for those two. |

### Adding a mission

Step by step, with the traps, in [Adding a mission](adding_a_mission.md). The short
version: the registry entry is usually five lines, and producing a reference file to
check it against is the work. That is why a mission is listed with `orbit=None` — "known
about, not yet validated" — until there is a file to check it against.

`Mission` is a frozen dataclass, so a misspelt field is a `TypeError` at import rather
than a silently ignored setting, and `mission_for` raises with the list of known missions
rather than guessing.

(the-orbit-reader)=
## The orbit reader

Every mission tabulates the same three things — time, geocentric position, geocentric
velocity — and differs only in the dialect. So [`orbit.py`](../src/barycenter/orbit.py)
has one reader driven by a declarative `OrbitSpec`, which is the `orbit` field of a
mission's [registry entry](#the-mission-registry):

| Mission | Extension | Position column | Velocity column | Units in the file |
|---|---|---|---|---|
| Fermi | `SC_DATA` | `SC_POSITION` | `SC_VELOCITY` | m |
| NuSTAR | 1 | `POSITION` | `VELOCITY` | **km** |
| SVOM | 1 | `POSITION` | `VELOCITY` | m |
| NICER, IXPE | `ORBIT` | scalar `X`,`Y`,`Z` | scalar `Vx`,`Vy`,`Vz` | m |
| RXTE | `XTE_PE` (or `ORBIT`) | scalar `X`,`Y`,`Z` | scalar `Vx`,`Vy`,`Vz` | m |
| XMM-Newton | `ORBIT` | scalar `GEI_X`,`GEI_Y`,`GEI_Z` | scalar `VX`,`VY`,`VZ` | **km** |
| Chandra | `ORBITEPHEM` | scalar `X`,`Y`,`Z` | scalar `Vx`,`Vy`,`Vz` | m |
| Swift | `PREFILTER` | `POSITION` | `VELOCITY` | **km** |

The units are declared in the spec, not read from `TUNITn`, because orbit files are
unreliable about that keyword and getting the factor of 1000 wrong is a 20 ms error.
The table that comes out is always in metres and metres per second.

The column names in that table are what the spec asks for, not what the file wrote:
every lookup goes through `utils.column_named`, which matches without regard to case.
FITS column names are case-insensitive by standard and missions use that freedom —
Chandra spells its orbit time `Time` and its velocities `Vx`, `Vy`, `Vz`, and its event
times `time`. The reason to be careful about it is that neither failure is loud: on the
orbit side a velocity that is not found is silently replaced by a numerical derivative of
the position, and on the event side a `time` column that is not found leaves the events
untouched while the capitalised `START`/`STOP` of the `GTI` extension beside them are
corrected, so the file comes out with its events and its good-time intervals on
different time scales. `core.TIME_COLUMNS` lists every column a correction must move:
`TIME`, `START`, `STOP`, `TSTART`, `TSTOP`.

Three cleanups are applied to **every** mission, not just to `FPorbit` files as PINT
does: sort by time, drop rows repeated at the same time (a spline through a repeated
abscissa is undefined, and repeats are common where two files overlap), and drop
all-zero placeholder rows (a zero position is a 6400 km error).

The output table carries both time bases — `MJD_TT`, which is what PINT's
`SatelliteObs` expects, and `MET`, which is what the native engine uses. Reading the
orbit file once, in one place, is what makes the native-versus-PINT comparison
meaningful: the two engines cannot disagree about where the spacecraft was.

(where-the-orbit-file-stops)=
## Where the orbit file stops

An interpolating spline answers every time it is asked about. Past its last knot it
extrapolates, and through an interior gap it coasts, and in neither case does it say so.
That is the behaviour we want for the sub-second shortfalls orbit files routinely have,
and exactly the wrong one for a file that is missing half an orbit: the answer is then a
guess written into an event list, indistinguishable from a measurement.

`OrbitCoverage` in `orbit.py` draws the line. It holds the sample times and the file's
own **cadence** — the median spacing — and `uncovered(times)` returns, per time, how many
seconds it reaches beyond what the file covers: zero between two samples, zero within one
cadence of either end, and positive past that or deep inside a gap.

The cadence allowance is not a nicety. Fermi tabulates the spacecraft position every
30 s, so a rule based on distance to the nearest *sample* would count **33.5 % of a
healthy LAT file** as uncovered purely for sitting between two samples. With the
allowance, the same file reports zero uncovered events out of 143 569 and zero uncovered
boundaries out of 1 308 GTIs — and still isolates the one time that really is outside.

### What happens to a time that is not covered

`enforce_orbit_coverage` tests every time column where the correction is actually
evaluated — after `TIMEZERO`, the leap seconds and the clock correction, since that is
what the orbit file gets asked about — and sorts the failures into three cases, against a
tolerance of `COVERAGE_TOLERANCE_S` (10 s) on top of the cadence allowance:

- **Inside a good time interval** → `ValueError`. There is no honest time to write, and
  silently dropping real events would be worse than refusing the file. A GTI boundary is
  a good time by definition, so a GTI reaching past the orbit file is always an error.
- **Outside every good time interval** → the row is dropped, with a loud warning. It is
  junk the orbit file also happens not to cover.
- **A file with no GTI extension at all** → every time counts as good. Silence about
  which times are trustworthy is not a claim that none of them are.

### `TSTART` and `TSTOP` are a separate case

These two keywords routinely hold the range that was *requested* rather than the one that
was delivered. A Fermi LAT extraction is the clearest example: the server returns
`TSTART` set to the start of the requested window, while the GTIs and the spacecraft file
begin whenever the data really does. On one M82 observation that gap is **1455.6 s**, and
it is why `gtbary` refuses the file outright:

```
Cannot get Fermi spacecraft position for 412041603 Fermi MET (TT):
the time is not covered by spacecraft file ...SC00.fits[SC_DATA]
```

`gtbary` is right, and the time it names is the `TSTART` keyword — not an event. Every
one of the 143 569 events and all 1 308 GTI boundaries in that file are properly covered.

Refusing the whole file over a keyword would be unhelpful, and extrapolating it is what
we used to do: the correction came out 300.4644 s, from a spacecraft position extrapolated
a quarter of an orbit past the end of the file. Chopping the first 49 samples off a real
Fermi spacecraft file and asking it to predict them back measures what that costs:

| extrapolated back | position error | timing error |
|------------------:|---------------:|-------------:|
|             300 s |         1.7 km |     0.002 ms |
|             600 s |          38 km |     0.008 ms |
|            1450 s |        1441 km |     2.3 ms   |

So `clamp_uncovered_keyword` moves such a keyword to the edge of the good time intervals
instead — the first GTI `START` for `TSTART`, the last `STOP` for `TSTOP` — warns loudly,
and corrects that. Where the file has no GTIs the event times stand in, which is the same
intent. Only these two keywords are ever moved, and only when they miss by more than the
tolerance; on the M82 file the events and GTIs come out bit-identical either way, and only
`TSTART` and the `DATE-OBS` derived from it change.

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

## Choosing an engine

`--engine native` is the default. `--engine pint` computes the same thing through PINT's
pulsar timing model. Measured on the committed NuSTAR reference (937 events, DE440,
ICRS, explicit coordinates, no clock correction):

| engine | platform | mean | std | max abs |
|---|---|---|---|---|
| native | arm64 **and** x86 | **+22.0 ns** | 19.5 ns | **59.6 ns** |
| pint | x86 (80-bit longdouble) | +117.2 ns | 19.6 ns | 149.0 ns |
| pint | arm64 (no extended precision) | +129.7 ns | 256.0 ns | 774.9 ns |

The native engine is the same to the last bit on both platforms, because every term it
computes is a small number of seconds. The PINT engine's constant offset is the Shapiro
convention and its arm64 spread is the `longdouble` MJD subtraction; both are explained
below. Keep `--engine pint` for a second opinion, and for a `.par` file whose proper
motion or parallax actually matters — the native engine uses only `RAJ`/`DECJ`.

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

So the 5-second grid is a good choice, and it is what the package now uses above
`core.AUTO_GRID_EVENTS` events.

#### When the grid is used, and what it costs

`apply_barycenter_correction` decides from the file's size, in `core.grid_spacing_for`:
below 100 000 events the correction is evaluated at **every event**, which is exact and
already costs well under a second; above it, a 5 s grid is used. `--dt` overrides the
decision either way, and `--dt 0` forces the exact path whatever the size. The threshold
means every reference comparison in the test suite runs on the unapproximated code, since
the committed test files all hold a few hundred events.

Measured end to end, on `dummy_evt.evt` blown up to 3 000 000 events (858 MB), with
identical code:

| | wall | peak RSS |
|---|---|---|
| `--dt 0`, exact per event | 23.0 s | 3.76 GB |
| `--dt 5` | **2.3 s** | **1.98 GB** |

and the two answers, on those three million events:

* **99.03 % of the stored times are bit-identical**
* the largest difference is 29.802 ns, which is **exactly one float64 step** at this MET
* mean +0.005 ns, std 2.9 ns — and that std is the storage quantisation, not
  interpolation error, which is 1.6 ns at most

So at the precision a FITS `D` column can hold, the grid is indistinguishable from the
exact path. The peak memory nearly halves as well, because the exact path runs
`barycentric_correction` over every event at once and a grid asks it for 16 560 points
instead of three million. Which phase holds that memory — 865 MB of the 905 MB total is
astropy's JPL ephemeris evaluation — is tabulated under
[Performance](known_issues.md#performance).

#### The grid has to be padded by two steps, not one

When the grid is clipped to the events — which is what makes it cheap on a multi-day
orbit file — the events must sit clear of the interpolant's **end conditions**. A spline's
first and last intervals are not the same function as its interior, so a grid padded by
one step puts those intervals exactly where the events are. Clipped against unclipped,
inside the span:

| margin | PINT (`Akima1DInterpolator`) | native (`CubicSpline`) |
|---|---|---|
| 1 step | **46.2 ns** | 0.045 ns |
| 2 steps | 0.000 ns | 0.013 ns |

The whole difference lives at the end of the span, confirming it is the boundary and not
interpolation error. PINT's is 1000× worse because `Akima1DInterpolator` builds the slopes
at its boundary knots from *extrapolated* points, and because its knots carry PINT's own
noise for that construction to amplify, while the native engine's knots hold exact values.
Both engines now pad by `2 * dt`, and `tests/test_native.py::TestGridClipping` and
`tests/test_pintengine.py::test_clipping_the_grid_does_not_move_the_answer` assert that
clipping is free.

The range itself comes from `core.met_range_for_file`: the widest `TSTART`/`TSTOP` over
every extension, padded by `MET_RANGE_PAD_S` (1000 s), and then **shifted by the clock and
leap-second terms**, because the barycentric correction is evaluated at the clock-corrected
time and not the raw one. On Swift that shift is nearly 20 s, so a grid clipped to the raw
span would leave every event outside it. The range is taken from the headers rather than
from the data because reading a strided time column out of a memory-mapped table pages in
the whole file; all five committed reference files keep their times inside their own
`TSTART`/`TSTOP` (the widest slack being Chandra's 823 s), and a file that does not is
caught by a coverage check that logs a warning naming how far outside it went.

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

### The reference file's own granularity

A reference file stores times as float64 seconds since `MJDREF`, so it cannot record a
difference finer than one unit in the last place: **29.8 ns** at NuSTAR's 1.8e8 s and
**119.2 ns** at RXTE's 5.4e8 s. Asserting a flat 100 ns on RXTE would be asserting
something the file is physically unable to express, so `assert_times_agree` in the test
suite adds that step to the tolerance, and reports the mean — which averages the
quantisation away — when it fails.

It also means a difference of tens of nanoseconds between two *derived* quantities can be
invisible. The RXTE clock correction is 17.4 µs on a 5.4e8 s timestamp, and the difference
between two such timestamps can only come out as a multiple of 119 ns, so the test that
pins down `tdc.dat` compares the correction function against the constant HEASOFT froze,
not the output files.

### RXTE

`tests/data/dummy_xte_bary_DE440_{noclk,clk}.evt.gz` are the same thing for RXTE PCA,
made from a decimated PSR B1509-58 observation with DE440, ICRS and explicit coordinates.
Measured on 404 events, native engine:

| reference | mean | std | max abs |
|---|---|---|---|
| `clockfile=NONE` | +25.4 ns | 48.8 ns | 119.2 ns (1 ulp) |
| `tdc.dat` applied | +24.8 ns | 48.4 ns | 119.2 ns (1 ulp) |

The two agreeing to 0.6 ns is the point: the clock correction is reproduced well enough
to leave the solar-system residual untouched.

### XMM-Newton

`tests/data/dummy_xmm_bary_DE430.evt.gz` is an SAS 22.1 `barycen` reference, made with
DE430 and explicit coordinates on 401 EPIC-pn events spanning 7.6 h of observation
0112290201. Measured with the native engine:

| quantity | mean | std | max abs |
|---|---|---|---|
| `TIME` column | **−42.3 ns** | 6.9 ns | 59.6 ns (4 ulp) |
| GTI `START`/`STOP` | −40 ns | — | 44.7 ns (3 ulp) |
| `TSTART`, `TSTOP` keywords | — | — | 373 ns, −179 ns |

The residual on the events is a **constant**: it sits on −3 units in the last place of the
reference's float64 times (14.9 ns each at XMM's 1.06e8 s), with no drift, no annual term
and no correlation with the spacecraft's geocentric distance. The 6.9 ns "scatter" is
entirely that quantisation. So the disagreement with `barycen` is an offset of about
−40 ns, which is below the 100 ns target and, being constant, affects no measured period
or pulse phase at all.

The two keywords are a different matter, and not our doing: SAS writes a floating-point
keyword with **15 significant digits**, which at 1.06e8 s leaves six decimals, so the
reference's `TSTART` is quantised at 1 µs — a hundred times coarser than the binary `TIME`
column beside it. Both keywords come out inside half a step, which is as close as the file
can record.

Ephemeris sensitivity on this dataset, which is also how DE430 was confirmed as the
reference's own choice (`barycen`'s default is DE200, so it has to be asked for):

| against the DE430 reference | offset |
|---|---|
| DE430 | −42 ns |
| DE405 | +1.7 µs |
| DE440 | +95 µs |
| DE200 | +1.8 ms |

### Chandra

`tests/data/dummy_chandra_bary_{DE405,DE200}.evt.gz` are CIAO `axbary` references made on
401 ACIS-S events spanning the 5.6 h of observation 10026, with explicit coordinates.
There are two of them because `axbary` chooses the ephemeris from the reference frame and
can reach only these two combinations — `refframe=ICRS` reads `JPLEPH.405` and
`refframe=FK5` reads `JPLEPH.200`. Measured with the native engine:

| reference | mean | std | max abs |
|---|---|---|---|
| DE405, ICRS | **+47.0 ns** | 24.4 ns | 59.6 ns (1 ulp) |
| DE200, FK5 | **+44.0 ns** | 26.2 ns | 59.6 ns (1 ulp) |
| GTI `START`/`STOP` | +59.6 ns | — | 59.6 ns (1 ulp) |

One unit in the last place is 59.6 ns at Chandra's 3.57e8 s, so the scatter is again the
reference file's own granularity rather than ours.

Having both is what makes the **ephemeris/frame pairing** testable against a real tool
instead of against ourselves. DE200 is referred to FK5 and DE405 onwards to ICRS; the code
pairs them in `native.ephemeris_frame`, and pairing them wrongly on this dataset costs
+11.2 µs (DE405 read in FK5) or −11.1 µs (DE200 read in ICRS) — a hundred times the
target, from a mistake no file will ever warn about. Every other official tool here was
run in one frame only, so no other reference can catch it.

Ephemeris sensitivity, measured against the DE405 reference:

| ephemeris | offset |
|---|---|
| DE405 | +47 ns |
| DE421 | −1.2 µs |
| DE430 | −1.5 µs |
| DE440 | **+79.4 µs** |
| DE200 | +1.8 ms |

The DE440 figure is the one to note, because it is much larger than the ~10 µs DE430→DE440
costs on the NuSTAR dataset, and it is not a Chandra effect: it is where the source is. The
difference between DE405 and DE440 is dominated by the position of the solar-system
barycentre, which moved when Jupiter's and Saturn's masses were revised, and 2009 is over a
decade past the data DE405 was fitted to. Projected on M82's direction that comes to 79 µs
— about 24 km — and it varies by less than one ulp across the exposure, so it is an
offset, not noise. **DE440 is the package default**, so a comparison against an `axbary` product has to
ask for DE405 explicitly; the tests do.

### Swift

`tests/data/dummy_swift_bary_DE440_{noclk,clk}.evt.gz` are HEASOFT `barycorr` references
made on 492 XRT events spanning the 6.4 ks of observation 00037258040, with explicit
coordinates. There are two of them because Swift is the only mission here where the clock
correction and the leap-second term can be separated: `clockfile=NONE` switches the UTCF
off and leaves the +4 s of leap seconds in place. Measured with the native engine:

| reference | mean | std | max abs |
|---|---|---|---|
| no clock file, `TIME` | **+36.6 ns** | 29.0 ns | 59.6 ns (1 ulp) |
| with the UTCF, `TIME` | **+37.4 ns** | 28.8 ns | 59.6 ns (1 ulp) |
| GTI `START`/`STOP` | +29.8 to +59.6 ns | — | 59.6 ns (1 ulp) |
| `TSTART`, `TSTOP` keywords | — | — | 59.6 ns (1 ulp) |

One unit in the last place is 59.6 ns at Swift's 4.7e8 s, so again the scatter is the
reference file's granularity and not ours, and the residual drifts by 0.9 ns per ks across
the exposure — nothing.

The two references agreeing to the same 37 ns is the useful part: it says the UTCF is
reproduced well enough to leave the solar-system residual untouched, even though it is a
15.56 s correction and drifts 342 µs across the observation.

Getting to those numbers took one non-obvious fix, recorded here because it will bite
anyone who writes this kind of term again. The leap-second count has to be read out of
ERFA's table as an integer, never computed as `(epoch.tai.mjd - epoch.utc.mjd) * 86400`:
two MJDs of order 5e4 cannot express 32 s to better than 0.6 µs, so that expression gives
31.999999937, and the resulting **+63 ns** bias on every Swift time is two thirds of the
budget. It showed up as an unexplained 96 ns disagreement with `barycorr` where a prototype
had measured 37 ns.

Ephemeris sensitivity, measured against the DE440 reference:

| ephemeris | offset |
|---|---|
| DE440 | +37 ns |
| DE430 | −274.2 µs |
| DE421 | −273.9 µs |
| DE405 | −274.3 µs |
| DE200 | +2.16 ms |

The three middle rows sitting on top of each other, 274 µs from DE440, is the same
solar-system-barycentre revision that costs 79 µs on the Chandra dataset, and it checks
out exactly: DE440 puts the Earth 116.5 km from where DE405 puts it at this epoch, and
82.2 km of that is along the direction of Mrk 421, which is 274.3 µs of light travel time.
The measurement and the geometry agree to 0.1 µs, so this is the ephemerides differing and
not the code.

(the-coordinate-keyword-order)=
### The coordinate keyword order is deliberately not HEASOFT's

When the position is not given explicitly, we read it from the header in the order
`RA_OBJ` → `RA_TARG` → `RA_NOM` → `RA_PNT` → `RA`. HEASOFT `barycorr` uses
`RA_NOM` → `RA_PNT` → `RA_OBJ` → `RA` (its own `kwfallback` call, `barycorr` 2.19
line 278). Both orders live in `barycenter.core` as `COORDINATE_KEYWORDS` and
`HEASOFT_COORDINATE_KEYWORDS`.

We start from `RA_OBJ` on purpose. `RA_OBJ`/`DEC_OBJ` is the position of the object the
observation was aimed at — for a known pulsar, its catalogue position. `RA_NOM`/`DEC_NOM`
and `RA_PNT`/`DEC_PNT` describe where the spacecraft was pointing, which is the same
direction only to within the pointing accuracy and is often written out rounded. The
barycentric correction refers arrival times to the source, so the source position is the
right input; the pointing is a proxy for it that a mission-planning tool happened to
record.

`RA_TARG`/`DEC_TARG` sits second because it is the same thing as `RA_OBJ` under another
name, and is what Chandra uses: a Chandra event file carries no `RA_OBJ` at all. Leaving
it out would fall straight through to the pointing, and on the ACIS test file `RA_NOM` is
324 arcsec from the target — 0.8 s of Roemer delay, not a rounding error.

The two are not interchangeable: on the NuSTAR test file `RA_OBJ` and `RA_NOM` differ by
0.1 arcsec, which is 172 µs of Roemer delay — a thousand times the accuracy target.
So when the header carries both and they differ by enough to matter (more than a
nanosecond of implied delay), `get_coordinates_from_fits_header` logs a warning naming
the keyword HEASOFT would have used and the size of the difference in microseconds.

**Comparisons against an official tool must therefore pass `--ra` and `--dec`
explicitly**, matched to whatever the reference was made with. Every reference-agreement
test in `tests/` does exactly that, which is why the difference in default order costs
nothing in validation while keeping the more accurate position in everyday use.

### The keywords derived from `TSTART` and `TSTOP`

Correcting the times leaves a handful of keywords that are not times themselves but are
computed from them, and which are therefore wrong the moment `TSTART` and `TSTOP` move.
`barycenter.core.DERIVED_KEYWORDS` lists them and `update_derived_keywords` rewrites
them, per extension, after that extension's times have been corrected. Each is written
only where the input already had it: adding a `DATE-END` to a file that never carried
one would be inventing metadata.

No official tool rewrites all of them, and each rewrites a different subset. Read off the
five committed reference pairs:

| keyword | HEASOFT `barycorr` | SAS `barycen` | CIAO `axbary` | here |
|---|---|---|---|---|
| `TSTART`, `TSTOP` | updated | updated | updated | updated |
| `TELAPSE` | **left stale** | updated | (absent) | updated |
| `DATE-OBS`, `DATE-END` | recomputed | string shifted | recomputed | recomputed |
| `MJD-OBS` | **left stale** | (absent) | recomputed | recomputed |
| `ONTIME`, `LIVETIME`, `EXPOSURE` | left | left | left | left |

`TELAPSE` is `TSTOP − TSTART`, and it moves by however much the corrections at the two
ends differ: 3.4 s over the NuSTAR test observation, 1.6 s over the 7.6 h XMM one.
`barycorr` moves `TSTART` by 309.5 s and `TSTOP` by 306.1 s and leaves `TELAPSE` at
81973.50000756979 — the value it had on the way in. That is a bug, not a convention, and
`barycen` shows what the right answer looks like. `tests/test_barycenter.py` asserts the
staleness of the reference explicitly, so the reason for not copying `barycorr` here is
recorded in a test rather than only in prose.

`DATE-OBS`, `DATE-END` and `MJD-OBS` are **recomputed** from `MJDREF + TSTART/86400`,
not shifted by however far `TSTART` moved. Both `barycorr` and `axbary` recompute, and
recomputing is the self-consistent answer: the output header says `TIMESYS = TDB`, so the
date should be the date of the time the file now records.

The two rules are not equivalent, because several missions write `DATE-OBS` in UTC while
counting their MET in TT seconds since `MJDREF`, so the file's date string and its own
`TSTART` disagree before anything is corrected:

| reference file | `DATE-OBS` − date of `TSTART`, as delivered |
|---|---|
| Chandra ACIS | −0.005 s (consistent) |
| RXTE PCA | +3.816 s |
| XMM EPIC-pn | −62.708 s |
| Swift XRT | −68.207 s |

`barycen` shifts the string and so carries that inconsistency through; recomputing
replaces it. On XMM our `DATE-OBS` therefore differs from `barycen`'s by exactly the
63 s above and by nothing else, which is what
`test_xmm_dates_differ_from_barycen_by_xmms_own_utc_offset` pins down. Both `barycorr`
and `axbary` truncate the string to whole seconds; we keep the milliseconds, since a
package aiming at 100 ns has no business rounding a timestamp to the nearest second, and
the tests compare against our string cut back the same way.

A keyword can also describe a correction the output has now *absorbed*, in which case
there is nothing left for it to describe and it is removed rather than rewritten;
`barycenter.core.ABSORBED_KEYWORDS` lists those. `UTCFINIT` is the one that exists today.
It means "the UTC correction factor at TSTART", and after barycentring TSTART has moved,
`TIMESYS` is TDB, and the factor has been folded into every time -- so applying it again
would move a Swift event a further 15.56 s. `barycorr` deletes it from every extension,
with or without a clock file, and both committed Swift references have it gone.

`ONTIME`, `LIVETIME` and `EXPOSURE` are deliberately left alone, as all three tools
leave them. They are sums of good-time interval lengths rather than differences between
the file's ends, and the corrections at the two edges of one interval differ by
microseconds — far below the precision those keywords are used at.

### `TIERABSO`, how well the clock is known

`TIERABSO` is "the absolute accuracy of the time, in seconds". Unlike everything above it
cannot be derived from the times in the file: it is a property of the clock correction
that was applied, so it is written only on a run that applies one, and each mission's
number needs its own justification. `barycenter.clock` supplies it alongside the
correction itself — every mission's clock-function builder now returns a triple of
`(correction, filename, accuracy)`, where `accuracy(met_start, met_stop)` is a callable
so that a mission whose accuracy varies with time can say so. `core` calls it once per
file, with the uncorrected `TSTART` and `TSTOP`, and writes the result into every
extension, as `hdaxbary` does.

Two of the three are constants, held in `clock.CLOCK_ACCURACY_S`:

| mission | value | where it comes from |
|---|---|---|
| Swift | 10 µs | constant, once the UTCF is applied |
| RXTE | 5 µs | constant, once `tdc.dat` is applied |
| NuSTAR | ~132 µs | the clock file's own `CLOCK_ERR_CORR` column |

The two constants are read off the committed references and match `hdaxbary` exactly.
They are read off rather than derived because the recipe lives in `hdaxbary`'s C source,
which is in `heasarc` — a component the distributed HEASOFT source tarballs do not
include — so the reference files are the only evidence for them that exists here.

NuSTAR's is measured rather than constant. `nustar_clock_accuracy_fun` reads
`CLOCK_ERR_CORR` from the fine clock file and returns the **largest** value anywhere in
the observation, including the two interpolated endpoints. On the test observation that
is 131.8 µs, where `hdaxbary` writes 122.9 µs — the value of the column near `TSTOP`.
Both are the same quantity read two different ways and they differ by 7 %; we take the
maximum because `TIERABSO` is one number describing the whole file, and the worst case
over the file is the honest reading of that. The exact recipe `hdaxbary` uses could not
be reproduced to its seven significant figures — neither linear nor spline interpolation
at the raw or corrected `TSTOP`, at the last event, nor the span's mean, minimum or
maximum lands on it — so the test asserts that ours is inside the column's range, is no
smaller than HEASOFT's, and is within 10 % of it, rather than that it is equal.

With `--clockfile none` nothing is written and any existing value is left untouched.
HEASOFT does the same for NuSTAR, but for Swift it writes `TIERABSO = 100` even when
told to apply no clock file. We deliberately do not copy that: our Swift times still
carry the leap-second term (see [Missions whose MET counts UTC
seconds](#missions-whose-met-counts-utc-seconds)), so the figure that would honestly
describe them is the size of the UTCF we were told not to read — 15.56 s on the test
file, not 100 s — and that is precisely the thing such a run cannot know.

Two related HEASOFT behaviours are knowingly not reproduced. `barycorr` also writes
`TIERRELA` (the *relative* accuracy, 1e-9 for NuSTAR) and writes it even with
`clockfile=NONE`; and it sets `CLOCKAPP = T` for Swift with `clockfile=NONE`, where we
write `F` because no clock file was applied. Both are recorded in the known issues.

### Things that will move the answer by more than 100 ns

When a comparison disagrees, check these before looking for a bug:

| Difference | Size (NuSTAR test file) |
|---|---|
| `RA_OBJ` vs `RA_NOM` (0.1 arcsec) — [our default differs from HEASOFT's](#the-coordinate-keyword-order) | 172 µs |
| `RA_TARG` vs `RA_NOM` (324 arcsec, Chandra ACIS) | 0.8 s |
| DE430 vs DE440 ephemeris | ~10 µs |
| DE405 vs DE430 | 0.38 µs |
| Clock correction applied before vs after | 1.1 µs |
| `(0.5 − TIMEPIXR) · TIMEDEL` half-bin shift (RXTE PCA) | 477 ns |
| RXTE PCA 16 µs detector delay | 16 µs |
| PINT's extra Shapiro `2T·ln(r/AU)` term | ~100 ns |
| `numpy.longdouble` being float64 (arm64, Windows) | up to 1.1 µs of scatter |
| GPS→UTC correction on/off | ~0.1 ns |

## Supported missions

| Mission | Orbit file | Clock correction | Status |
|---|---|---|---|
| NuSTAR | `nu<obsid>A.attorb` | `nuCclock*.fits`, `NU_FINE_CLOCK` extension | validated to 100 ns |
| NICER | `ni<obsid>.orb` | none needed | works |
| RXTE | `orbit/FPorbit_*` | HEASOFT `tdc.dat`, bundled with the package | validated to 100 ns |
| IXPE | `FPorbit`-style | none needed | works |
| Fermi | FT2 `SC_DATA`, `SC_POSITION` in m, timed by `START` | none needed | validated to 100 ns |
| SVOM | `POSITION`/`VELOCITY` in m | to be determined | works |
| XMM-Newton | PPS `P*OBX000ORBTSR*.FTZ`, `GEI_*` in km | none needed | validated to 100 ns — **the only route, see below** |
| Chandra | `primary/orbitf*_eph1.fits`, `ORBITEPHEM` in m | none needed | validated to 100 ns — **the only route, see below** |
| Swift | `auxil/sw<obsid>sao.fits`, `PREFILTER` in km | `swclockcor*.fits`, `CLOCK_CORRECT` extension (the UTCF) | validated to 100 ns |
| ASCA | — | — | `--apply-official` only, DE200 only |

## Test data

Everything in `tests/data/` is small on purpose, so that CI can check us against the
official tools without installing HEASOFT, SAS or CIAO.

| File | What it is |
|---|---|
| `dummy_evt.evt` | 937 NuSTAR FPMA events, MJDREF 55197.00076601852 |
| `dummy_orb.fits.gz` | the matching orbit file, 82800 rows at 1 s |
| `dummy_fine_clk.fits` | 92 rows of `NU_FINE_CLOCK`, trimmed from CALDB `nuCclock20100101v230.fits.gz` (12 MB) to the span of the events plus 5000 s |
| `dummy_clk.fits` | a NuSTAR clock file with the *old* `CLOCK_CORRECT` extension, kept so the test suite can prove it is refused |
| `dummy_par.par` | a minimal timing model for the same position (`EPHEM DE436`) |
| `dummy_evt_bary_DE440_noclk.evt.gz` | the `barycorr` reference described above, clock correction off |
| `dummy_evt_bary_DE440_clk.evt.gz` | the same with `clockfile=dummy_fine_clk.fits`, which is what pins down the order the two corrections are applied in |
| `dummy_xte_evt.evt` | 404 events, every 64th row of PINT's `B1509_RXTE_short.fits` (public RXTE PCA data), so the sample spans the whole hour |
| `dummy_xte_orb.fits.gz` | 78 rows of the matching `FPorbit_Day6223`, the observation plus 600 s either side |
| `dummy_xte_bary_DE440_noclk.evt.gz` | the `barycorr` reference for those events, `clockfile=NONE` |
| `dummy_xte_bary_DE440_clk.evt.gz` | the same with `tdc.dat` applied, which barycorr does whatever `clockfile` says |
| `dummy_xmm_evt.evt` | 401 EPIC-pn events, every 839th row of observation 0112290201, so the sample spans the whole 7.6 h, plus one `STDGTI` extension |
| `dummy_xmm_orb.fits.gz` | 2832 rows of the matching PPS `ORBTSR` file, every 10th second over the events plus 600 s, with **all ten** columns kept |
| `dummy_xmm_bary_DE430.evt.gz` | the SAS 22.1 `barycen` reference for those events, DE430, GTIs corrected too |
| `dummy_chandra_evt.evt` | 401 ACIS-S events, every 164th row of observation 10026, so the sample spans the whole 5.6 h, plus its `GTI` extension |
| `dummy_chandra_orb.fits.gz` | 71 rows of the matching `orbitf*_eph1.fits`, the file's own 300 s sampling over the events plus 1200 s |
| `dummy_chandra_bary_DE405.evt.gz` | the CIAO `axbary` reference for those events, `refframe=ICRS` |
| `dummy_chandra_bary_DE200.evt.gz` | the same with `refframe=FK5`, which is how the ephemeris/frame pairing gets checked against a real tool |
| `dummy_swift_evt.evt` | 492 XRT photon-counting events, every 3rd row of observation 00037258040, so the sample spans the whole 6.4 ks, plus its `GTI` extension |
| `dummy_swift_orb.fits.gz` | 3271 rows of the matching `sw*sao.fits` prefilter, every 2nd second over the events plus 600 s, cut to `TIME`, `POSITION`, `VELOCITY` |
| `dummy_swift_clk.fits` | 15 intervals of CALDB `swclockcor20041120v174.fits`, spanning the 2015-07-01 leap second as well as the observation |
| `dummy_swift_bary_DE440_noclk.evt.gz` | the `barycorr` reference for those events, `clockfile=NONE` — which still carries the +4 s of leap seconds |
| `dummy_swift_bary_DE440_clk.evt.gz` | the same with the UTCF applied, 15.56 s away from its twin |
| `dummy_fermi_evt.evt` | 413 simulated LAT events, every 10th row of the ScienceTools tutorial's `fakepulsar_event.fits`, so the sample spans the whole week, plus all 70 of its `GTI` rows |
| `dummy_fermi_orb.fits.gz` | 20199 rows of the matching `simscdata_1week.fits` FT2 file, at its native 30 s over the events and GTIs plus 600 s, cut to `START`, `STOP`, `SC_POSITION` — **not** decimated, see below |
| `dummy_fermi_bary_DE405.evt.gz` | the Fermi `gtbary` reference for those events, `solareph="JPL DE405"`, GTIs corrected too |

The XMM orbit file keeps its `GSE_*` columns on purpose. The file offers two position
triples of identical length — `GEI_*` is geocentric equatorial and is the one the
ephemeris is referred to, `GSE_*` is the same vector rotated into the Earth-Sun frame —
and reading the wrong one is a 160 ms error that nothing in the units or the column
comments would give away. A committed file that still offers the wrong choice is a
sharper test than a hand-built one.

The Swift prefilter is decimated to 2 s rather than something coarser for a reason worth
recording, since it costs 90 kB: `hdaxbary` refuses to read a prefilter sampled more
coarsely than about 10 s — 15 s and up fail with "no bracketing sample found" on a file
that brackets the time comfortably — and while 5 s and 10 s are read, they move the
reference times by a full float64 ulp (59.6 ns at Swift's MET, most of the budget) against
the native 1 s sampling. 2 s is the coarsest step that is bit-identical to 1 s.

The clock file is trimmed to 15 intervals rather than the one that contains the
observation, so that the tests can check the two things that only show up at an interval
boundary: that the row containing a time is the one used, and that the tabulated correction
steps by exactly one second at a leap second. The boundaries are found in the file itself,
by evaluating each interval's polynomial at its own end and at the next interval's start —
ordinary boundaries agree to a few microseconds, a leap second shows up as a one-second
jump — rather than by looking the leap seconds up and trusting them to line up.

The Chandra files keep their column names exactly as Chandra writes them, for the same
reason: `time` in the events, `Time` in the orbit file, `START`/`STOP` in capitals in the
`GTI` extension beside them. That mixture is the only committed example of the
case-insensitivity the FITS standard grants and most missions never use, and the failure
it guards against is silent — a case-sensitive lookup corrects the GTIs and leaves the
events alone.

`tools/make_test_data.py` regenerates all of them, including the trimming, and it now
allocates its own pseudo-terminal: HEASOFT tasks open `/dev/tty` for their prompts and
abort with `ERROR: Device not configured` without one, and the old `script -q /dev/null`
wrapper only worked when it already had a terminal to start from.

### Why the Fermi orbit file is not decimated

Every other orbit file here is thinned — XMM to every 10th second, Swift to every 2nd.
The Fermi one is not, and that is measured rather than cautious. `simscdata_1week.fits`
carries `SC_POSITION` but no `SC_VELOCITY` column, which older FT2 files routinely do
not, so the velocity has to come from differentiating the position — and a coarser grid
makes a worse derivative. Regenerating the `gtbary` reference at each step and comparing:

| spacecraft sampling | agreement with `gtbary` |
|--------------------:|------------------------:|
|          30 s (native) |    **29.8 ns** (one ulp) |
|                   60 s |                 119.2 ns |
|                  120 s |                 327.8 ns |
|                  240 s |                 4440 ns  |

The tolerance is 129.8 ns, so 60 s would pass — at 92 % of it, with nothing left for a
platform whose longdouble differs. 30 s is exact and costs 305 KB gzipped, so that is
what is committed. Note that this is a property of *this* file rather than of Fermi:
a current FT2 file with an `SC_VELOCITY` column would take a Hermite spline instead and
thin far better.

### Driving SAS `barycen` for the XMM reference

`barycen` is harder to drive than `barycorr`, and the reasons are worth writing down:

* **It has no output parameter.** It edits the table it is given, irreversibly — its own
  documentation says so. `run_barycen` therefore copies the input into a private temporary
  directory, runs there, and moves the result out. Neither the committed input nor the
  source observation is ever written to.
* **It will not take an orbit file on the command line.** The spacecraft position comes
  through SAS's observation access layer, which means an *ingested* ODF. An ingested ODF's
  summary (`*SUM.SAS`) records **absolute** paths, so a summary made elsewhere fails with
  `OrbitFileOpenError` naming a directory that no longer exists. A copy of the ODF is
  re-ingested inside the working directory; that is the only reason `odfingest` appears in
  the script. The orbit it then reads is the ODF's `ROS.ASC`, *not* the PPS `ORBTSR` file
  we read — so the XMM test quietly checks that those two describe the same orbit.
* **No CCF is needed.** Checked by running with `SAS_CCF` and `SAS_CCFPATH` unset and
  comparing the output bit for bit. `setsas.sh` prints a reminder to set them anyway.
* **`setsas.sh` needs HEASOFT initialised first**, exports nothing unless `SAS_DIR` is
  already set, clears the positional parameters of the script that sourced it, and tries to
  raise the stack limit — which fails harmlessly in a sandbox. So it must not be sourced
  under `set -e`, and anything the caller passed has to be saved before sourcing it.
* Unlike HEASOFT, SAS tasks need **no controlling terminal**.

Regenerating the reference also needs the observation itself, which is not committed:
`XMM_OBS` in the script names where it lives on the machine that made it, in the same way
`RXTE_EVENTS` points at PINT's test data and `CALDB_CLOCK` at a local CALDB.

### Driving CIAO `axbary` for the Chandra references

`axbary` is easier than `barycen` — it writes a new file rather than editing in place, and
needs no controlling terminal — but it has one trap and one limit:

* **It fails silently without its calibration data.** It looks for `tai-utc.dat` and
  `JPLEPH.*` under `$TIMING_DIR` or `$ASCDS_CALIB`. With neither set it prints
  `Could not initialize bary stuff`, **exits 0**, and writes the times out unchanged — so
  the output is a perfectly valid file that is bit-for-bit the input. `run_axbary` sets
  `ASCDS_CALIB` and then checks that the times actually moved.
* **It is a shell wrapper around `pset`/`pget`**, so CIAO's `bin` has to be on `PATH`, not
  merely be where the executable came from. Without it the wrapper reads no parameters and
  does nothing, again exiting 0.
* **It can only reach DE200 and DE405**, chosen through `refframe=FK5` and `refframe=ICRS`
  respectively — there is no ephemeris parameter. That is the main reason to prefer the
  native engine on Chandra: DE405 is 79 µs from DE440 on this dataset.

`barycorr` is **not** an option for Chandra, despite the mission appearing in its
documentation. On a Chandra orbit file it reports `no bracketing sample found for time
357377056.00505000` followed by `Invalid Observatory/Spacecraft position vector`, on a file
that brackets that time comfortably — reproduced with both the trimmed and the full,
uncompressed file. `strings` on `hdaxbary` shows readers for `xtescorbit`, `nicerscorbit`,
`swiftscorbit` and NuSTAR, and nothing for Chandra. So `MISSIONS["chandra"].official` is
`None`, not `"barycorr"`.

## Documentation hosting

`.github/workflows/docs.yml` runs `hatch run docs:build` on every push to `main` (so, on every merged pull request) and can also be started by hand from the Actions tab. The warnings-as-errors build output in `docs/_build` is published to the `gh-pages` branch with `peaceiris/actions-gh-pages`, which writes `.nojekyll` itself. The branch is recreated from scratch each time (`force_orphan`), so it holds no history. In the repository settings, Pages must be set to "Deploy from a branch", `gh-pages`, root. The site is at <https://matteobachetti.github.io/barycenter/>.
