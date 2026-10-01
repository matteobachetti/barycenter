# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries are assembled by [towncrier](https://towncrier.readthedocs.io) from the
fragments in [`docs/changes/`](docs/changes/README.md); do not edit the released sections by
hand.

<!-- towncrier release notes start -->

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
