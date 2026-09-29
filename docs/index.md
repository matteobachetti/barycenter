# barycenter

## Overview

Barycentric correction of X-ray event files, in pure Python.

`barycenter` moves the photon arrival times in a FITS event file from the spacecraft
clock to the solar-system barycentre — the job of HEASOFT `barycorr`, XMM-SAS
`barycen` and CIAO `axbary` — without needing any of those installations.

```bash
barycenter nu30702012003A06_cl.evt nu30702012003A.attorb \
    --ra 148.9584 --dec 69.6794 --ephem DE440
```

The result agrees with `barycorr` to better than 150 ns on the test data; see
[Technical details](technical_details.md#accuracy) for the measurements and for what
moves the answer by more than that.

## Documentation

```{toctree}
:maxdepth: 2

technical_details.md
known_issues.md
./api/modules.rst
```

## Copyright

- Copyright © 2025 Matteo Bachetti.
- Free software distributed under the MIT License.
