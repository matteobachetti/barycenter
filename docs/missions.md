# Barycentring, mission by mission

Every run needs three things: the **event file**, the mission's **orbit file** (where the
spacecraft was), and a **position on the sky**. Some missions need a fourth, a
**spacecraft clock file**. This page says where each of those comes from for each
mission, gives the command, and names the one thing that catches people out.

What is *not* mission-specific is worth saying once. The mission is recognised from the
`TELESCOP` keyword, so you never name it. The position can be given as `--ra`/`--dec` in
degrees, or with `--parfile`, which is usually better because a parameter file carries
proper motion and parallax. `--ephem` selects the JPL ephemeris and defaults to DE440.
Times, good-time intervals and the `TSTART`/`TSTOP` keywords are all corrected, in every
extension of the file — if they were not, the events and the GTIs would end up on
different time scales and up to 500 s of exposure could silently vanish.

For how any of it works, see [Technical details](technical_details.md). For what is
wrong or missing, [Known issues](known_issues.md).

## NuSTAR

| | |
|---|---|
| Orbit file | `nu<obsid>A.attorb` in the observation's `auxil/` directory |
| Clock file | `nuCclock*.fits` — **fetched automatically** |
| Reference | agrees with HEASOFT `barycorr` to **+22 ns** |

```bash
barycenter nu30702012003A06_cl.evt nu30702012003A.attorb \
    --ra 148.9584 --dec 69.6794 --ephem DE440
```

The clock file is the thing to know about. NuSTAR's onboard clock needs a correction of
20–30 milliseconds, which is five orders of magnitude above the 100 ns target, so it is
not optional. If you do not pass `--clockfile`, the newest file is downloaded from the
CALDB and cached under your user cache directory — they are about 12 MB, so this happens
once per machine, not once per run. With no network, the newest already-cached file is
used.

**The catch:** clock files from before 2019 carry a different extension
(`CLOCK_CORRECT`, a quadratic HEASOFT documents as good only to the millisecond) and are
**refused** rather than used. A millisecond is ten thousand times the target, and a file
corrected that way would look corrected while being nothing of the kind. Fetch a current
clock file.

## RXTE

| | |
|---|---|
| Orbit file | `orbit/FPorbit_*` in the observation directory |
| Clock file | `tdc.dat` — **bundled with this package** |
| Reference | agrees with HEASOFT `barycorr` to **+25 ns** |

```bash
barycenter FS37_be7ab98-be8f0ab.evt FPorbit_D06-05-18 \
    --ra 228.4818 --dec -59.1358
```

RXTE's clock coefficients are not in a FITS file but in `tdc.dat`, a 53 kB ASCII table.
HEASOFT ignores its own `clockfile` parameter for RXTE and always reads
`$LHEA_DATA/tdc.dat`, so a copy ships inside the package: RXTE stopped observing in
January 2012 and the file is final. `$TIMING_DIR` and `$LHEA_DATA` are checked first, so
an installation's own copy wins. Naming a clock file on the command line warns and uses
`tdc.dat` anyway — again, what HEASOFT does.

**The catch:** the correction is only 17.4 µs, which sounds negligible and is 170 times
the target. `--clockfile none` skips it, and should only be used deliberately.

## Swift

| | |
|---|---|
| Orbit file | `auxil/sw<obsid>sao.fits` |
| Clock file | `swclockcor*.fits`, from `swift/mis/bcf/clock/` in the CALDB |
| Reference | agrees with HEASOFT `barycorr` to **+37 ns** |

```bash
barycenter sw00037258040xwtw2po_cl.evt sw00037258040sao.fits \
    --ra 182.6358 --dec 39.4058 --clockfile swclockcor20041120v110.fits
```

Swift is the awkward one, for two reasons that both come from the same place: its mission
elapsed time counts **UTC** seconds, not the TT seconds every other mission here uses.

First, the leap seconds since 2001 have to be added — four seconds on a 2015
observation — and this package always does so, whatever `--clockfile` says, because a
flag should not be able to cause a whole-second error.

Second, Swift's "clock file" is not a fine clock correction but the **UTC correction
factor**, the offset between the onboard clock and UTC. It is −15.56 s on the test
observation, and it drifts 4.6 ms per day, so the quadratic in the table has to be
evaluated and not just its constant term.

**The catch:** an observation within about 15 seconds of a leap second may come out a
second wrong. The leap-second step and the clock file's own compensating step are ~15 s
apart in different time bases, and no reference exists to settle which should win.
Leap seconds are rare, observations straddling them rarer; it is written up honestly in
[Known issues](known_issues.md#correctness) rather than guessed at.

## XMM-Newton

| | |
|---|---|
| Orbit file | PPS product `P*OBX000ORBTSR*.FTZ` |
| Clock file | none needed |
| Reference | agrees with SAS `barycen` to **−42 ns** |

```bash
barycenter P0112290201PNS003PIEVLI0000.FTZ P0112290201OBX000ORBTSR0000.FTZ \
    --ra 148.9583 --dec 69.6792 --ephem DE430
```

This is the only route for XMM if you do not have SAS: HEASOFT `barycorr` refuses the
mission outright, despite listing it.

**The catch, for anyone extending the reader:** the `ORBTSR` file tabulates the
spacecraft position **twice**, as `GEI_*` and as `GSE_*`. The first is geocentric
equatorial, which is what is wanted; the second is the same vector rotated into the
Earth–Sun frame. They are exactly the same length as each other, so nothing catches the
mistake, and it is a 160 millisecond error. The committed test file keeps both triples
so that the reader's choice stays under test.

## Chandra

| | |
|---|---|
| Orbit file | `primary/orbitf*_eph1.fits` |
| Clock file | none needed |
| Reference | agrees with CIAO `axbary` to **+47 ns** |

```bash
barycenter acisf10026N003_evt2.fits orbitf301093602N002_eph1.fits \
    --ra 148.9583 --dec 69.6792 --ephem DE440
```

Like XMM, this is the only route without the mission's own software — and unlike XMM
there is a positive reason to prefer it even if you have CIAO. `axbary` picks its
ephemeris from the reference frame and can reach only DE200 (with FK5) and DE405 (with
ICRS). On the test observation **DE405 is 79 µs away from DE440**, so being able to ask
for a modern ephemeris is not a detail.

**The catch:** Chandra writes its column names in mixed case — `time` in the events,
`Time` in the orbit file, `START`/`STOP` in the GTI beside them. FITS says column names
are case-insensitive and this package honours that, but be aware of it if you write
your own reader: a `time` column that is not found leaves the events untouched *while*
the GTIs are corrected, and nothing complains.

## NICER

| | |
|---|---|
| Orbit file | `ni<obsid>.orb` in `auxil/` |
| Clock file | none needed |
| Reference | +53 ns against `barycorr` in a one-off check; no tracked reference |

```bash
barycenter ni1013010101_0mpu7_cl.evt ni1013010101.orb --parfile Crab.par
```

NICER needs no clock correction and its orbit file is the plain `ORBIT`-extension
dialect, which makes it the least fussy mission here. The 100 ns agreement was measured
once, against a `barycorr` run on a Crab observation too large to commit, so continuous
integration does not re-check it the way it re-checks the five missions above.

That comparison did settle one useful thing, because it was run with **DE200 and FK5**
rather than DE440 and ICRS: reading DE200 coordinates in the wrong frame costs 45 µs.
DE200 is referred to FK5 and DE405 onwards to ICRS, `--radecsys` overrides the pairing,
and the default gets it right.

## IXPE, Fermi and SVOM

These three have registry entries and their orbit files are read, but **none has ever
been compared against an official tool**, so treat the output as unverified. IXPE uses
the same `ORBIT`-extension dialect as NICER; Fermi's positions come from the `SC_DATA`
extension of an FT2 file; SVOM's from a `POSITION`/`VELOCITY` pair in metres.

Fermi carries an open question worth knowing about before you trust a Fermi result: its
`MJDREF` is bit-for-bit Swift's, whose fractional part is the signature of a mission
elapsed time counting UTC seconds. If Fermi's does too, it needs Swift's leap-second
term, and leaving it out is a whole-number-of-seconds error. Settling it needs an FT1/FT2
pair and a `gtbary` reference; see [Known issues](known_issues.md#correctness).

## ASCA

ASCA has no native orbit reader. It can only be done by handing the job to HEASOFT
`timeconv`, with `--apply-official`, which also means DE200 only.

## If your mission is not here

Adding one is usually about five lines of registry entry, plus the work of producing a
reference file to check it against. See [Adding a mission](adding_a_mission.md).
