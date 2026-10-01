# Adding a mission

Everything mission-specific lives in one dictionary in
[`missions.py`](../src/barycenter/missions.py). Nothing else in the package branches on a
mission name: `core.py` never mentions one, and `clock.py` and `official.py` only look
entries up. So adding a mission is a registry entry, and then the real work — a reference
file to prove the entry is right.

## 1. Write the registry entry

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
| `name` | canonical short name; must match the dictionary key |
| `telescop` | lower-case *substrings* of the `TELESCOP` keyword that identify the mission — substrings, because the keyword is written `XTE` and `RXTE`, `NuSTAR` and `NUSTAR`, `AXAF` and `CHANDRA` |
| `orbit` | an [`OrbitSpec`](technical_details.md#the-orbit-reader), or `None` for no native reader |
| `clock` | `clock(clockfile, instrument) -> (function, path, accuracy)`, or `None` for a mission needing no correction. `accuracy` is a callable of the MET span returning the clock's absolute accuracy in seconds, which `core.py` writes as `TIERABSO` |
| `official` | `"barycorr"`, `"timeconv"` or `None` — which branch of `official.py` `--apply-official` takes |
| `official_ephem` | the only ephemeris that tool can manage, if it is stuck on one: DE200 for ASCA's `timeconv`. The native engine has no such limit, which is the main reason to prefer it for such a mission. |
| `met_is_utc` | `True` if the mission's elapsed time counts **UTC** seconds rather than TT seconds, so leap seconds must be added. Currently Swift alone; see the warning below. |

`Mission` is a frozen dataclass, so a misspelt field is a `TypeError` at import rather
than a silently ignored setting, and `mission_for` raises with the list of known missions
rather than guessing.

## 2. Describe the orbit file

If the orbit file's layout already matches an existing `OrbitSpec`, reuse it — three
scalar `X`, `Y`, `Z` columns in metres covers NICER, RXTE, Chandra and IXPE between them.
The [dialect table](technical_details.md#the-orbit-reader) lists what each existing
mission needed.

Two things in an `OrbitSpec` are easy to get wrong and quiet when you do:

- **Units are declared, not read** from `TUNITn`, because orbit files are unreliable
  about that keyword. Getting the factor of a thousand wrong is a 20 ms error.
- **A velocity column that is not found is silently replaced** by a numerical derivative
  of the position. That is a deliberate fallback for files that omit velocities, but it
  also means a misspelt velocity column name produces a slightly worse answer instead of
  an exception. Check that the names are found.

If the file offers more than one position triple, make sure you know which frame each one
is in. XMM's `ORBTSR` tabulates `GEI_*` (geocentric equatorial, the one wanted) and
`GSE_*` (the same vector rotated into the Earth–Sun frame) at identical length, and
picking the wrong one is a 160 ms error that nothing in the units or the comments reveals.

## 3. Check whether the clock needs correcting, and in which time base

Two questions, and the second one is the one people forget.

**Does the onboard clock need a correction?** If so, `clock` is a builder returning
`(function, path, accuracy)`: a function of mission elapsed time, the file the correction
came from, and a callable of the MET span giving the clock's absolute accuracy in seconds,
which is written into every extension as `TIERABSO`. A mission whose accuracy is a
constant can use `constant_clock_accuracy`; NuSTAR's is read out of the clock file's own
`CLOCK_ERR_CORR` column. The three existing ones span three orders of magnitude
— NuSTAR tens of milliseconds, Swift tens of seconds, RXTE tens of microseconds — and
all three matter at the 100 ns level.

**Does mission elapsed time count UTC seconds or TT seconds?** Almost every mission
counts TT seconds and needs nothing. Swift counts UTC seconds, so the leap seconds
elapsed since its `MJDREF` have to be added, and leaving that out is a four-second error
on a 2015 observation. The tell-tale is a fractional `MJDREF` encoding TT − UTC at the
reference epoch: Swift's is `51910.00074287037`, whose fraction is 64.184 s.

:::{warning}
**Fermi's `MJDREF` is bit-for-bit Swift's**, and `met_is_utc` is nevertheless `False` for
it, because no Fermi result has ever been compared against `gtbary` here. If you add a
mission whose `MJDREF` fraction looks like a TT − UTC offset, do not guess: an error here
is a whole number of seconds, not a small one. `tests/test_missions.py` asserts the
current set explicitly so it cannot drift without a test failing.
:::

## 4. Produce a reference, and commit it trimmed

This is the actual work, and it is the reason a mission is not called supported until it
is done. `orbit=None` in the registry means "known about, not yet validated" rather than
pretending — Swift, XMM and Chandra all started that way.

[`tools/make_test_data.py`](../tools/make_test_data.py) does the whole job: it runs the
mission's own tool and trims the result to something small enough to commit. Everything
in `tests/data/` is a few hundred events, decimated so that the sample still spans the
whole observation, so that continuous integration can check against the official tools
without HEASOFT, SAS or CIAO installed.

Decimate the *events*, not the span. 401 events taken every 839th row across 7.6 hours
tests the annual and diurnal terms; 401 consecutive events test almost nothing.

The tool already knows how to drive HEASOFT `barycorr`, SAS `barycen`, CIAO `axbary` and
`hdaxbary`, including several traps that cost hours to find: HEASOFT tasks need a
controlling terminal, `barycen` edits its input in place, `axbary` exits 0 on failure.
Those are written up in
[Technical details](technical_details.md#test-data).

## 5. Add the test

The test asserts agreement with the reference to 100 ns, through `assert_times_agree`,
which adds the reference file's own granularity to the tolerance. That matters: a
reference stores times as 64-bit floats, so it cannot express a difference finer than one
unit in the last place — 29.8 ns at NuSTAR's mission elapsed time, 119.2 ns at RXTE's.
Asserting a flat 100 ns on RXTE would be asserting something the file physically cannot
express.

Check the GTIs and the `TSTART`/`TSTOP` keywords too, not only the `TIME` column. Events
corrected while good-time intervals are left on the spacecraft clock is a silent failure
that can truncate up to 500 s of exposure.

## What you do not have to do

Nothing outside `missions.py` should need touching. If you find yourself adding a mission
name to `core.py`, that is a sign the registry is missing a field — add the field. The
one file that does need a manual edit is `docs/api/barycenter.rst`, and only if you add a
whole new *module*: it is maintained by hand, and autodoc skips silently what is not
listed there.
