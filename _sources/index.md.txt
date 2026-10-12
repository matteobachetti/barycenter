# barycenter

## Overview

Barycentric correction of X-ray event files, in pure Python.

A photon's arrival time at the spacecraft is not a useful clock: the spacecraft moves
around the Earth, and the Earth around the Sun, so recorded arrival times wander by up to
about 500 seconds over a year. `barycenter` reports each photon as it would have arrived
at the solar system's centre of mass — the job of HEASOFT `barycorr`, XMM-SAS `barycen`
and CIAO `axbary` — without needing any of those installations.

```bash
barycenter nu30702012003A06_cl.evt nu30702012003A.attorb \
    --ra 148.9584 --dec 69.6794 --ephem DE440
```

The result agrees with each mission's own tool to better than 100 ns on five missions with
committed reference files; see [Technical details](technical_details.md#accuracy) for the
measurements and for what moves the answer by more than that.

## Where to start

- **[Barycentring, mission by mission](missions.md)** — which files you need for NuSTAR,
  RXTE, Swift, XMM-Newton, Chandra and NICER, where they come from, the exact command,
  and the one thing that catches people out on each.
- **[Adding a mission](adding_a_mission.md)** — the registry entry, and the reference
  file that makes it a supported mission rather than a hopeful one.
- **[Technical details](technical_details.md)** — how it works and every number quoted
  anywhere else.
- **[Known issues](known_issues.md)** — an honest list of what is wrong or missing.

## Documentation

```{toctree}
:maxdepth: 2

missions.md
adding_a_mission.md
technical_details.md
known_issues.md
./api/modules.rst
```

## Copyright

- Copyright © 2025 Matteo Bachetti.
- Free software distributed under the MIT License.
