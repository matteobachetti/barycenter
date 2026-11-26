# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries are assembled by [towncrier](https://towncrier.readthedocs.io) from the
fragments in [`docs/changes/`](docs/changes/README.md); do not edit the released sections by
hand.

<!-- towncrier release notes start -->

## [1.2.0](https://github.com/matteobachetti/barycenter/tree/v0.1.5.dev48+g2bb5256f9) - 2026-11-26

### Backwards-incompatible changes

- Python 3.12 or newer is now required. Dropping 3.10 and 3.11 also avoids a clash between pytest 9.2 and the pytest-remotedata release that still installs on them. ([#7](https://github.com/matteobachetti/barycenter/pull/7))

### New features

- New `--fill-orbit-gaps` option: fit a Kepler+J2 orbit across long gaps in the orbit file (e.g. the Fermi LAT's South Atlantic Anomaly passages) instead of refusing the events that fall in them. The position there is good to about 100 m (0.4 us of light time); the gaps used are recorded in the output header. Native engine only. ([#3](https://github.com/matteobachetti/barycenter/pull/3))
- Read the Fermi GBM daily position history (`glg_poshist_all_*.fit`), so GBM event files can be barycentred through the South Atlantic Anomaly gaps of the LAT spacecraft file. ([#3](https://github.com/matteobachetti/barycenter/pull/3))
- New `barycenter-apply-gti` command and `barycenter.gti` module: combine good time intervals from separate extensions or a separate file (as SVOM/ECLAIRs ships them) into one `GTI` extension of the event file, so they are barycentred together with the events. ([#4](https://github.com/matteobachetti/barycenter/pull/4))
- XRISM support: the orbit file's `ORBIT` extension is read natively (agreeing with HEASOFT `barycen` to 41 ns), and `--apply-official` runs HEASOFT `barycen` for XRISM files. ([#5](https://github.com/matteobachetti/barycenter/pull/5))
- `barycenter-apply-gti` now filters every event table of a file (SVOM/GRM has one per detector), and `--existing-gti intersect|replace` (`existing_gti=` in `add_gti_extension`) handles a file that already has a `GTI` extension. ([#6](https://github.com/matteobachetti/barycenter/pull/6))
- SVOM/GRM: synthetic `GRM-EVT` test file (three detector tables on the SVOM orbit dummy), a test that each table gets the single-table correction, and a GRM section in the SVOM docs. ([#7](https://github.com/matteobachetti/barycenter/pull/7))

### Bug fixes

- A NuSTAR time more than 1000 s past either end of the clock file now raises an error, telling you to fetch a newer clock file or pass `--clockfile none`. Before, it got a cubic extrapolation that came out at about -1000 s for events a week past the file. Within 1000 s the correction now continues along the last tabulated slope, and times inside the file are unchanged. ([#6](https://github.com/matteobachetti/barycenter/pull/6))

### Documentation

- New SVOM section in the mission guide: which files go together, merging the separate GTI file, the time scale and orbit units, and the file quirks, and what the NEO/PEO/TEO Earth-occultation GTIs mean for timing. The guide to adding a mission now explains how to build a reference for a mission without an official tool. ([#4](https://github.com/matteobachetti/barycenter/pull/4))


## [1.0.0](https://github.com/matteobachetti/barycenter/tree/v1.0.0) - 2026-10-01

### Backwards-incompatible changes

- The single `barycenter.barycenter` module is split into `cli`, `core`, `orbit`, `native`, `pintengine`, `clock`, `official`, `remote` and `utils`; public names are importable from `barycenter` itself, but imports from `barycenter.barycenter` must be updated. ([#2](https://github.com/matteobachetti/barycenter/pull/2))

### New features

- Files above 100 000 events are corrected on a 5 s interpolated grid, cutting a 3-million-event run from 23 s to 2.3 s; `--dt` overrides this and `--dt 0` forces the exact path. ([#2](https://github.com/matteobachetti/barycenter/pull/2))
- A new native barycentring engine (astropy, ERFA and a JPL ephemeris, no PINT) agrees with HEASOFT `barycorr` to about 20 ns, runs some 30 times faster, and is now the default. `--engine pint` keeps the old path, and old ephemerides such as DE200 and DE405 work again. ([#2](https://github.com/matteobachetti/barycenter/pull/2))
- A bare clock-file name is now looked up in `$TIMING_DIR` and `$LHEA_DATA`, and header position keywords fall back to plain `RA`/`DEC` with a warning naming the keyword used. ([#2](https://github.com/matteobachetti/barycenter/pull/2))
- XMM-Newton (validated against SAS `barycen`) and Chandra (validated against CIAO `axbary`) are supported natively, as is Swift (validated against `barycorr` to 37 ns), including its UTC-based mission elapsed time. ([#2](https://github.com/matteobachetti/barycenter/pull/2))
- Times outside the orbit file's coverage are no longer silently extrapolated: inside a good time interval they raise an error, outside they are dropped with a warning, and a `TSTART`/`TSTOP` that was merely requested is moved to the good time edge. ([#2](https://github.com/matteobachetti/barycenter/pull/2))
- Fermi is now validated against `gtbary` to 29.8 ns, which also confirms its mission elapsed time needs no leap-second term, and a Fortran `D`-exponent `MJDREF` is now read. ([#2](https://github.com/matteobachetti/barycenter/pull/2))
- RXTE observations are now clock-corrected from the shipped `tdc.dat`, agreeing with `barycorr` to 25 ns, and `TIERABSO` is written whenever a clock correction is applied. ([#2](https://github.com/matteobachetti/barycenter/pull/2))

### Bug fixes

- `--radecsys` now changes the result on the PINT path, and `TELAPSE`, `DATE-OBS`, `DATE-END` and `MJD-OBS` are rewritten while `UTCFINIT` is dropped from the output. ([#2](https://github.com/matteobachetti/barycenter/pull/2))
- ASCA `timeconv` runs in a private temporary directory with its reference files in astropy's download cache, no longer depending on the working directory. ([#2](https://github.com/matteobachetti/barycenter/pull/2))
- FITS columns are matched case-insensitively and `RA_TARG`/`DEC_TARG` are read, which fixes Chandra files being silently half-barycentred. ([#2](https://github.com/matteobachetti/barycenter/pull/2))
- Local input files are read in place instead of being copied into the current directory, relative paths resolve from the caller's directory, and nothing is written next to the inputs. ([#2](https://github.com/matteobachetti/barycenter/pull/2))
- `--clockfile none` now skips the clock correction instead of failing, the clock correction is applied before the barycentric one as in `barycorr`, and the `TIMEPIXR` half-bin shift is no longer applied. ([#2](https://github.com/matteobachetti/barycenter/pull/2))
- Clipping the correction grid with `met_range` no longer changes the answer by up to 46 ns, downloads no longer corrupt astropy's cache, and `fits_open_remote` no longer raises `UnboundLocalError` on local paths. ([#2](https://github.com/matteobachetti/barycenter/pull/2))

### Documentation

- New documentation covers how the correction is computed, its measured accuracy against `barycorr`, known issues, a per-mission how-to (`docs/missions.md`) and a guide to adding a mission. ([#2](https://github.com/matteobachetti/barycenter/pull/2))
- `tools/benchmarks/` holds the measurements behind every speed and memory claim, and the README now describes the tool instead of a template. ([#2](https://github.com/matteobachetti/barycenter/pull/2))
- The documentation is rebuilt on every push to `main` and published to the `gh-pages` branch, at <https://matteobachetti.github.io/barycenter/>. ([#2](https://github.com/matteobachetti/barycenter/pull/2))

### Internal changes

- Dependencies are now declared properly, with optional extras `remote`, `regions` and `speed` (numba, imported lazily), and `stingray` is no longer needed. ([#2](https://github.com/matteobachetti/barycenter/pull/2))
- The CALDB clock-file fetcher works for any mission via `CLOCK_CALDB`, which also stops a cached NuSTAR clock file being returned for another mission. ([#2](https://github.com/matteobachetti/barycenter/pull/2))
- Everything mission-specific lives in one registry, `barycenter.missions.MISSIONS`, read by a single orbit reader, so adding a mission is one entry; the global monkey-patch of PINT is replaced by a `SatelliteObs` subclass. ([#2](https://github.com/matteobachetti/barycenter/pull/2))
