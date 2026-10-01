# barycenter

Barycentric correction of X-ray event files, in pure Python.

[![Tests](https://github.com/matteobachetti/barycenter/actions/workflows/test.yml/badge.svg)](https://github.com/matteobachetti/barycenter/actions/workflows/test.yml)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

A photon's arrival time at the spacecraft is not a useful clock. The spacecraft moves
around the Earth, and the Earth around the Sun, so recorded arrival times wander by up to
about 500 seconds over a year. *Barycentring* removes that: it reports each photon as it
would have arrived at the solar system's centre of mass, which is very nearly inertial, so
times can be compared between missions and across decades.

That is the job of HEASOFT `barycorr`, XMM-SAS `barycen` and CIAO `axbary`. `barycenter`
does it with nothing installed but Python, agrees with each of those tools to better than
100 nanoseconds, and is not restricted to the ephemerides they happen to ship.

## Install

```bash
pip install git+https://github.com/matteobachetti/barycenter.git
```

The optional `speed` extra installs [numba](https://numba.pydata.org), which makes the
spacecraft-clock interpolation about six times faster. It changes no answer — the
compiled and pure-`numpy` kernels are bit-identical — and only pays for its compilation
time on very large files:

```bash
pip install "barycenter[speed] @ git+https://github.com/matteobachetti/barycenter.git"
```

## Use it from the command line

You need the event file and the mission's orbit file, and a position on the sky:

```bash
barycenter nu30702012003A06_cl.evt nu30702012003A.attorb \
    --ra 148.9584 --dec 69.6794 --ephem DE440
```

The position can come from a pulsar parameter file instead, which saves copying
coordinates by hand and also picks up the ephemeris the model was fitted with:

```bash
barycenter ni1013010101_0mpu7_cl.evt ni1013010101.orb --parfile Crab.par
```

The default engine takes the static `RAJ`/`DECJ` out of the model. If the source's proper
motion or parallax matters, add `--engine pint`, which hands the whole timing model over
and so applies them.

Several orbit files can be given at once, for an observation split across them, and
`--clockfile` names a spacecraft clock file (NuSTAR's is fetched from the CALDB
automatically if you do not). `barycenter --help` lists the rest; the options worth
knowing about are `--ephem`, `--engine` and `--dt`.

## Use it from Python

The whole command line is one function:

```python
from barycenter import apply_barycenter_correction

outfile = apply_barycenter_correction(
    "nu30702012003A06_cl.evt",
    "nu30702012003A.attorb",
    outfile="bary.evt",
    ra=148.9584,
    dec=69.6794,
    ephem="DE440",
)
```

If you only want the correction itself — to apply to times you already hold, or to plot —
ask for the function of mission elapsed time and call it:

```python
from barycenter import get_barycentric_correction, read_orbit

orbit = "nu30702012003A.attorb"
correction = get_barycentric_correction(orbit, ra=148.9584, dec=69.6794, ephem="DE440")

met = read_orbit(orbit)["MET"].value  # the times the orbit file covers
barycentred = met + correction(met)  # correction is in seconds, about +309 here
```

The returned object is a cubic spline over the orbit file's own time span, so ask it only
for times inside that span: outside it the spline extrapolates, quietly and badly. The
command line checks this for you and warns.

## Missions

| Mission | Orbit file | Clock correction | Agreement with the mission's own tool |
|---|---|---|---|
| NuSTAR | `nu<obsid>A.attorb` | `nuCclock*.fits`, fetched automatically | +22 ns vs `barycorr` |
| RXTE | `orbit/FPorbit_*` | `tdc.dat`, bundled | +25 ns vs `barycorr` |
| Swift | `auxil/sw<obsid>sao.fits` | `swclockcor*.fits` (the UTCF) | +37 ns vs `barycorr` |
| XMM-Newton | PPS `*ORBTSR*.FTZ` | none needed | −42 ns vs SAS `barycen` |
| Chandra | `primary/orbitf*_eph1.fits` | none needed | +47 ns vs CIAO `axbary` |
| NICER | `ni<obsid>.orb` | none needed | +53 ns vs `barycorr`, one-off check |
| Fermi LAT | FT2 spacecraft file | none needed | 30 ns vs `gtbary` |
| Fermi GBM | `glg_poshist_all_*.fit` | none needed | `gtbary` refuses GBM files; identical to the LAT route given the same positions |
| IXPE, SVOM | see the docs | none implemented | orbit file read, never checked against a tool |

The first five rows and Fermi LAT have a reference file committed to the repository, so continuous
integration re-checks them on every change. The NICER figure comes from a single
comparison against a `barycorr` run on a Crab observation too large to commit, and the
last three missions have never been compared against an official tool at all — their
orbit files are read correctly, which is a different claim.

XMM-Newton and Chandra are worth singling out: `barycorr` cannot do either mission, and
`axbary` can only reach DE200 and DE405, so this is the only way to barycentre them
against a modern ephemeris. On the Chandra test data DE405 is 79 µs away from DE440.

Per-mission instructions are in [docs/missions.md](docs/missions.md), and adding a new
mission is [about five lines](docs/adding_a_mission.md) plus a reference file to check it
against.

## Accuracy

The target is 100 nanoseconds against each mission's own tool, and it is met on all five
missions for which an official reference could be produced. Most of the residual scatter
in the table above is not ours: a reference file stores times as 64-bit floats, which
near a mission elapsed time of 5e8 seconds cannot express a difference finer than about
119 ns, so the tests add that floor to their tolerance.

Two engines compute the correction. The default, `--engine native`, uses `astropy`, ERFA
and a JPL ephemeris directly, keeping every term as a small quantity so that it works
identically on any platform. `--engine pint` routes the same job through
[PINT](https://github.com/nanograv/PINT) as an independent second opinion; the two agree
to sub-nanosecond once the differing Shapiro-delay convention is accounted for. On Apple
Silicon and Windows the PINT path is quantised at about 1.1 µs, because those platforms
have no extended-precision float; the native engine is unaffected.

## Documentation

- [Per-mission how-to](docs/missions.md) — where each mission's files come from, and the exact command
- [Adding a mission](docs/adding_a_mission.md)
- [Technical details](docs/technical_details.md) — how it works, and every measurement quoted above
- [Known issues](docs/known_issues.md) — an honest list of what is wrong or missing

## Contributing

Bug reports and patches are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Copyright

- Copyright © 2025 Matteo Bachetti.
- Free software distributed under the [MIT License](./LICENSE).
