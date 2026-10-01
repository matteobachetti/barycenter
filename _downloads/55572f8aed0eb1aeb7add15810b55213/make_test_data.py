#!/usr/bin/env python
"""Regenerate the reference files in ``tests/data`` with the official tools.

The reference files record what a mission's own barycentring tool produces for
the small event files committed in this repository.  Committing them means CI
can check us to 100 ns without installing HEASOFT, SAS or CIAO.

Every reference is generated with *every* input pinned: ephemeris, reference
frame, source coordinates and clock file.  A reference made with different
settings will disagree at a level far above 100 ns (DE430 vs DE440 alone is
~10 us for NuSTAR), so the settings below are part of the test, not incidental.

Run this only when a reference needs to change, and commit the result together
with the change that motivated it.

Notes for whoever runs it
-------------------------
* HEASOFT tasks need a controlling terminal, and the environment that
  ``headas-init.sh`` sets up.  :func:`run_heasoft` provides both, so this script can be
  run from anywhere, including a non-interactive shell.
* Each run gets a private PFILES directory, otherwise concurrent runs clobber
  each other's parameter files.
"""

import glob
import os
import pty
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))
from barycenter.clock import SWIFT_CLOCK_EXTENSION  # noqa: E402

HERE = os.path.abspath(os.path.dirname(__file__))
DATA = os.path.join(HERE, os.pardir, "tests", "data")
HEADAS = os.path.expanduser("~/mamba/envs/henv313_x86/heasoft")

#: The RXTE dataset the XTE reference is made with: a PSR B1509-58 PCA observation and
#: the matching FPorbit file, both public HEASARC data that ship with PINT's test suite.
#: Trimmed copies are committed; see :func:`trim_rxte_inputs`.
RXTE_EVENTS = os.path.expanduser("~/devel/pint/tests/datafile/B1509_RXTE_short.fits")
RXTE_ORBIT = os.path.expanduser("~/devel/pint/tests/datafile/FPorbit_Day6223")

#: The CALDB clock file the fine-clock reference is made with.  It is 12 MB, so only the
#: ~90 rows covering the test observation are committed, as ``dummy_fine_clk.fits``; see
#: :func:`trim_clock_file`.
CALDB_CLOCK = os.path.expanduser(
    "~/devel/CALDB/data/nustar/fpm/bcf/clock/nuCclock20100101v230.fits.gz"
)

#: The XMM-Newton dataset the ``barycen`` reference is made with: the EPIC-pn event list of
#: observation 0112290201 (M82, revolution 258), the PPS orbit file beside it, and the ODF
#: that ``barycen`` reads the spacecraft position from.  Nothing under this directory is
#: ever written to: ``barycen`` edits the table it is given *in place*, so every run here
#: works on copies in a private temporary directory.
XMM_OBS = os.path.expanduser("~/tmp/m82_xmm/0112290201")
XMM_EVENTS = os.path.join(XMM_OBS, "event_cl", "xmm0112290201_pn_S003_imaging_cl.evt")
XMM_ORBIT = os.path.join(XMM_OBS, "PPS", "P0112290201OBX000ORBTSR0000.FTZ")
XMM_ODF = os.path.join(XMM_OBS, "event_cl", "odf")

#: The Chandra dataset the ``axbary`` references are made with: the ACIS-S event list of
#: observation 10026 (M82) and the orbit ephemeris beside it, both public Chandra archive
#: data.  ``axbary`` writes a new file rather than editing in place, so nothing here is at
#: risk, but it is still only ever read.
CHANDRA_OBS = os.path.expanduser("~/tmp/m82_acis/10026/primary")
CHANDRA_EVENTS = os.path.join(CHANDRA_OBS, "acisf10026N003_evt2.fits.gz")
CHANDRA_ORBIT = os.path.join(CHANDRA_OBS, "orbitf356961902N001_eph1.fits.gz")

#: The Swift dataset the ``barycorr`` references are made with: the XRT photon-counting
#: event list of observation 00037258040, the prefilter (``sao``) product beside it that
#: carries the spacecraft position, and the CALDB clock file.  Public HEASARC data; a
#: trimmed copy of each is committed.  PC mode rather than the WT-mode file in the same
#: directory, which holds only 50 events over 19 s of exposure.
SWIFT_OBS = os.path.expanduser("~/tmp/swift_00037258040")
SWIFT_EVENTS = os.path.join(SWIFT_OBS, "xrt", "event", "sw00037258040xpcw3po_cl.evt.gz")
SWIFT_ORBIT = os.path.join(SWIFT_OBS, "auxil", "sw00037258040sao.fits.gz")
SWIFT_CLOCK = os.path.join(SWIFT_OBS, "swclockcor20041120v174.fits")

#: The Fermi dataset the ``gtbary`` reference is made with: the simulated pulsar event
#: list and the one-week spacecraft file from the Fermi ScienceTools pulsar tutorial,
#: both public and small enough to fetch directly:
#:
#:     https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/data/Pulsars/fakepulsar_event.fits
#:     https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/data/Pulsars/simscdata_1week.fits
#:
#: Simulated rather than real LAT data on purpose: it is the file the mission's own
#: documentation barycentres, so the reference can be checked against a published one.
FERMI_OBS = os.path.expanduser("~/tmp/fermi_pulsar")
FERMI_EVENTS = os.path.join(FERMI_OBS, "fakepulsar_event.fits")
FERMI_ORBIT = os.path.join(FERMI_OBS, "simscdata_1week.fits")

#: Where the Fermi ScienceTools live.  ``gtbary`` needs ``$TIMING_DIR`` pointing at the
#: directory holding ``JPLEPH.*``; without it it fails outright rather than quietly, which
#: is one thing it does better than ``axbary``.
FERMI_ENV = os.path.expanduser("~/mamba/envs/fermi")

#: How the Fermi reference is made.  ``ra``/``dec`` are the simulated pulsar's position,
#: passed explicitly like everywhere else.  DE405 because that and DE200 are the only two
#: ``gtbary`` offers, and DE405 is the one paired with ICRS.
FERMI_REFERENCE = {
    "infile": "dummy_fermi_evt.evt",
    "orbitfile": "dummy_fermi_orb.fits.gz",
    "outfile": "dummy_fermi_bary_DE405.evt.gz",
    "args": {"ra": "111.11", "dec": "22.22", "solareph": "JPL DE405"},
}

#: Where CIAO lives.  ``axbary`` needs its own leap-second file and JPL ephemerides, which
#: it looks for under ``$TIMING_DIR`` or ``$ASCDS_CALIB``; with neither set it reports
#: "Could not initialize bary stuff" and then writes the times out *unchanged*, so this
#: has to be right or the reference is silently the input.
CIAO = os.path.expanduser("~/mamba/envs/ciao")

#: How the Chandra references are made.  Two of them, because ``axbary`` chooses the
#: ephemeris from the reference frame and can reach only these two combinations:
#: ``FK5`` reads ``JPLEPH.200`` and ``ICRS`` reads ``JPLEPH.405``.  Having both is what
#: makes the ephemeris/frame pairing testable against a real tool -- pairing DE405 with
#: FK5, or DE200 with ICRS, is an 11 us error, and nothing in either file says so.
#: ``ra``/``dec`` are RA_TARG/DEC_TARG, passed explicitly like everywhere else here.
CHANDRA_REFERENCES = {
    "dummy_chandra_bary_DE405.evt.gz": {
        "args": {"ra": "148.959167", "dec": "69.679722", "refframe": "ICRS"},
    },
    "dummy_chandra_bary_DE200.evt.gz": {
        "args": {"ra": "148.959167", "dec": "69.679722", "refframe": "FK5"},
    },
}

#: Where SAS lives.  Globbed rather than pinned, so a SAS update does not silently stop
#: the XMM reference from being regenerable.
SAS_DIRS = sorted(glob.glob(os.path.expanduser("~/devel/SAS/xmmsas_*")))

#: How the XMM reference is made.  ``srcra``/``srcdec`` are the position of M82 X-2, which
#: matches none of the file's own RA_/DEC_ keywords on purpose: a bug where the code reads
#: a header keyword instead of the coordinates it was handed would then show up as a
#: 100 us disagreement rather than as nothing at all.  ``DE430`` is not the task default
#: (DE200 is) and has to be asked for: DE405 would be 1.7 us out and DE200 1.8 ms out.
XMM_REFERENCE = {
    "outfile": "dummy_xmm_bary_DE430.evt.gz",
    "infile": "dummy_xmm_evt.evt",
    "args": {
        "withsrccoordinates": "yes",
        "srcra": "148.96267",
        "srcdec": "69.67931",
        "processgtis": "yes",
        "ephemeris": "DE430",
    },
}

#: Swift's own RA_OBJ/DEC_OBJ, passed explicitly like everywhere else here.  RA_NOM is
#: 116 arcsec away, and barycorr prefers RA_NOM.
SWIFT_RA, SWIFT_DEC = "182.635833", "39.405833"

#: One entry per reference file.  ``args`` are passed to ``barycorr`` verbatim.
REFERENCES = {
    # NuSTAR, no clock correction.  The clock correction is left out on purpose:
    # barycorr applies it *before* evaluating the barycentric correction and the
    # orbit lookup, while we add it afterwards, a ~1.1 us difference that is
    # tracked separately (see docs/technical_details.md).  Keeping it out makes
    # this reference a clean test of the solar-system delays alone.
    "dummy_evt_bary_DE440_noclk.evt.gz": {
        "infile": "dummy_evt.evt",
        "orbitfiles": ["dummy_orb.fits.gz"],
        "args": {
            "clockfile": "NONE",
            "refframe": "ICRS",
            "ephem": "JPLEPH.440",
            # dummy_evt.evt has RA_OBJ=294.91067 but RA_NOM=294.9107; barycorr
            # prefers RA_NOM.  Passing both explicitly removes the ambiguity:
            # 0.1 arcsec is 172 us of Roemer delay.
            "ra": "294.9107",
            "dec": "21.58308",
        },
    },
    # The same thing with the fine clock correction applied.  This is the reference that
    # pins down the *order* of the two corrections: barycorr applies the clock first and
    # then evaluates the barycentric correction, and the spacecraft position, at the
    # clock-corrected time.  The clock correction here is about 25 ms, so getting the
    # order wrong costs ~1.1 us.
    "dummy_evt_bary_DE440_clk.evt.gz": {
        "infile": "dummy_evt.evt",
        "orbitfiles": ["dummy_orb.fits.gz"],
        "extra_inputs": ["dummy_fine_clk.fits"],
        "args": {
            "clockfile": "dummy_fine_clk.fits",
            "refframe": "ICRS",
            "ephem": "JPLEPH.440",
            "ra": "294.9107",
            "dec": "21.58308",
        },
    },
    # RXTE, with and without the fine clock correction.  barycorr ignores its
    # ``clockfile`` parameter for RXTE and reads $LHEA_DATA/tdc.dat instead, so the pair
    # of references is the only way to see what that file contributes (~33 us here).
    "dummy_xte_bary_DE440_noclk.evt.gz": {
        "infile": "dummy_xte_evt.evt",
        "orbitfiles": ["dummy_xte_orb.fits.gz"],
        "args": {
            "clockfile": "NONE",
            "refframe": "ICRS",
            "ephem": "JPLEPH.440",
            # The file has only RA_PNT/DEC_PNT, so there is nothing to be ambiguous
            # about, but pin them anyway: the test passes the same numbers.
            "ra": "228.481995",
            "dec": "-59.136002",
        },
    },
    "dummy_xte_bary_DE440_clk.evt.gz": {
        "infile": "dummy_xte_evt.evt",
        "orbitfiles": ["dummy_xte_orb.fits.gz"],
        "args": {
            "clockfile": "CALDB",
            "refframe": "ICRS",
            "ephem": "JPLEPH.440",
            "ra": "228.481995",
            "dec": "-59.136002",
        },
    },
    # Swift, without and with its clock correction.  Unlike every other mission here the
    # two differ by *seconds*, not microseconds: Swift's clock correction is the mission's
    # UTCF.  Both references still carry the leap-second term, which barycorr applies
    # whatever clockfile says -- that is the whole point of having the pair.
    "dummy_swift_bary_DE440_noclk.evt.gz": {
        "infile": "dummy_swift_evt.evt",
        "orbitfiles": ["dummy_swift_orb.fits.gz"],
        "args": {
            "clockfile": "NONE",
            "refframe": "ICRS",
            "ephem": "JPLEPH.440",
            "ra": SWIFT_RA,
            "dec": SWIFT_DEC,
        },
    },
    "dummy_swift_bary_DE440_clk.evt.gz": {
        "infile": "dummy_swift_evt.evt",
        "orbitfiles": ["dummy_swift_orb.fits.gz"],
        "extra_inputs": ["dummy_swift_clk.fits"],
        "args": {
            "clockfile": "dummy_swift_clk.fits",
            "refframe": "ICRS",
            "ephem": "JPLEPH.440",
            "ra": SWIFT_RA,
            "dec": SWIFT_DEC,
        },
    },
}


def trim_clock_file(source=CALDB_CLOCK, event_file=None, outfile=None, margin=5000.0):
    """Cut a NuSTAR fine clock file down to the span of one event file.

    The CALDB file covers the whole mission in 12 MB; the test observation needs about
    90 of its 477604 rows.  ``margin`` keeps a few samples beyond each end (the table is
    sampled every 1000 s) so the interpolation sees the same neighbours it would in the
    full file.
    """
    import numpy as np
    from astropy.io import fits

    event_file = event_file or os.path.join(DATA, "dummy_evt.evt")
    outfile = outfile or os.path.join(DATA, "dummy_fine_clk.fits")
    times = fits.getdata(event_file, 1)["TIME"]

    with fits.open(source) as hdul:
        hdu = hdul["NU_FINE_CLOCK"]
        t = hdu.data["TIME"]
        keep = (t > times.min() - margin) & (t < times.max() + margin)
        trimmed = fits.BinTableHDU(data=hdu.data[keep], header=hdu.header, name=hdu.name)
        trimmed.header["TSTART"] = float(np.min(t[keep]))
        trimmed.header["TSTOP"] = float(np.max(t[keep]))
        trimmed.header.add_history(
            f"Trimmed from {os.path.basename(source)} to the span of "
            f"{os.path.basename(event_file)} by tools/make_test_data.py"
        )
        fits.HDUList([hdul[0].copy(), trimmed]).writeto(outfile, overwrite=True)
    print(f"    wrote {outfile} ({keep.sum()} of {len(t)} rows)")
    return outfile


def trim_rxte_inputs(nevents=400, margin=600.0):
    """Cut the RXTE event and orbit files down to something committable.

    The events are decimated rather than truncated, so the sample still spans the whole
    observation and the test exercises the orbit interpolation over a full RXTE orbit
    rather than a few seconds of it. The orbit file is cut to the observation plus
    ``margin`` seconds either side.
    """
    import numpy as np
    from astropy.io import fits

    evt_out = os.path.join(DATA, "dummy_xte_evt.evt")
    orb_out = os.path.join(DATA, "dummy_xte_orb.fits")

    with fits.open(RXTE_EVENTS) as hdul:
        events = hdul["XTE_SE"]
        step = max(1, len(events.data) // nevents)
        trimmed = fits.BinTableHDU(data=events.data[::step], header=events.header, name=events.name)
        trimmed.header.add_history(
            f"Every {step}th row of {os.path.basename(RXTE_EVENTS)}, by tools/make_test_data.py"
        )
        # Only the first GTI: the file has two identical copies of the extension.
        out = [hdul[0].copy(), trimmed, hdul["GTI"].copy()]
        tstart = float(events.header["TSTART"])
        tstop = float(events.header["TSTOP"])
        fits.HDUList(out).writeto(evt_out, overwrite=True)
    print(f"    wrote {evt_out} ({len(trimmed.data)} of {step * nevents} rows)")

    with fits.open(RXTE_ORBIT) as hdul:
        orbit = hdul["XTE_PE"]
        t = orbit.data["Time"]
        keep = (t > tstart - margin) & (t < tstop + margin)
        trimmed = fits.BinTableHDU(data=orbit.data[keep], header=orbit.header, name=orbit.name)
        trimmed.header["TSTART"] = float(np.min(t[keep]))
        trimmed.header["TSTOP"] = float(np.max(t[keep]))
        trimmed.header.add_history(
            f"Trimmed from {os.path.basename(RXTE_ORBIT)} by tools/make_test_data.py"
        )
        fits.HDUList([hdul[0].copy(), trimmed]).writeto(orb_out, overwrite=True)
    print(f"    wrote {orb_out} ({keep.sum()} of {len(t)} rows)")
    subprocess.run(["gzip", "-9", "-f", orb_out], check=True)
    return evt_out, orb_out + ".gz"


#: Keywords a freshly built table defines for itself, plus everything indexed by a column
#: number.  Carrying those over from a table with different columns leaves TFIELDS
#: disagreeing with the number of columns, which SAS refuses to open.
_STRUCTURAL = re.compile(r"^(XTENSION|BITPIX|NAXIS\d*|PCOUNT|GCOUNT|TFIELDS|END)$|^T[A-Z]+\d+$")


def subset_table(hdu, rows, columns=None):
    """A copy of one table HDU with a subset of its rows and, optionally, its columns.

    ``fits.BinTableHDU(data=..., header=hdu.header)`` is enough when every column is kept,
    but dropping one leaves the old ``TTYPEn``/``TFORMn`` cards behind.  So the columns are
    rebuilt and only the non-structural keywords are carried over.
    """
    from astropy.io import fits

    names = list(hdu.data.names) if columns is None else list(columns)
    trimmed = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name=name,
                format=hdu.columns[name].format,
                unit=hdu.columns[name].unit,
                array=hdu.data[name][rows],
            )
            for name in names
        ],
        name=hdu.name,
    )
    for card in hdu.header.cards:
        if card.keyword and card.keyword not in ("COMMENT", "HISTORY"):
            if not _STRUCTURAL.match(card.keyword):
                trimmed.header[card.keyword] = (card.value, card.comment)
    for entry in hdu.header.get("HISTORY", []):
        trimmed.header.add_history(entry)
    return trimmed


def trim_xmm_inputs(nevents=400, orbit_step=10, margin=600.0):
    """Cut the XMM event and orbit files down to something committable.

    The event list keeps ``TIME`` and ``PI`` only, decimated so that the sample still spans
    the whole 7.6 h exposure, plus one ``STDGTI`` extension -- which is what makes the
    reference able to check the GTIs as well as the events.

    The orbit file keeps **all ten** of its columns.  The two position triples are the
    point: ``GEI_X/Y/Z`` is geocentric equatorial and is the one to use, ``GSE_X/Y/Z`` is
    geocentric solar-ecliptic, and reading the wrong triple is a 160 ms error, so a
    committed file that still offers both is a sharper test than a hand-built one.  Rows
    are kept every ``orbit_step`` seconds rather than at the file's own 1 s sampling:
    measured against the full-rate file, 10 s costs 0.8 ns and saves 1.8 MB.
    """
    import numpy as np
    from astropy.io import fits

    evt_out = os.path.join(DATA, "dummy_xmm_evt.evt")
    orb_out = os.path.join(DATA, "dummy_xmm_orb.fits")

    with fits.open(XMM_EVENTS) as hdul:
        events = hdul["EVENTS"]
        nrows = len(events.data)
        step = max(1, nrows // nevents)
        rows = slice(None, None, step)
        trimmed = subset_table(events, rows, columns=("TIME", "PI"))
        trimmed.header.add_history(
            f"Every {step}th row of {os.path.basename(XMM_EVENTS)}, by tools/make_test_data.py"
        )
        # Materialised, not a view: it is used after the file is closed.
        times = np.array(events.data["TIME"][rows])
        fits.HDUList([hdul[0].copy(), trimmed, hdul["STDGTI01"].copy()]).writeto(
            evt_out, overwrite=True
        )
    print(f"    wrote {evt_out} ({len(times)} of {nrows} rows)")

    with fits.open(XMM_ORBIT) as hdul:
        orbit = hdul["ORBIT"]
        t = orbit.data["TIME"]
        window = np.flatnonzero((t > times.min() - margin) & (t < times.max() + margin))
        keep = window[::orbit_step]
        trimmed = subset_table(orbit, keep)
        trimmed.header["TSTART"] = float(t[keep].min())
        trimmed.header["TSTOP"] = float(t[keep].max())
        trimmed.header.add_history(
            f"Every {orbit_step}th row of {os.path.basename(XMM_ORBIT)} over the span of "
            f"{os.path.basename(evt_out)}, by tools/make_test_data.py"
        )
        fits.HDUList([hdul[0].copy(), trimmed]).writeto(orb_out, overwrite=True)
    print(f"    wrote {orb_out} ({len(keep)} of {len(t)} rows)")
    subprocess.run(["gzip", "-9", "-f", orb_out], check=True)
    return evt_out, orb_out + ".gz"


def trim_swift_inputs(nevents=400, orbit_step=2, margin=600.0):
    """Cut the Swift event, orbit and clock files down to something committable.

    The event list keeps ``TIME`` and ``PI`` only, plus the ``GTI`` extension.  The
    prefilter file has 31 columns of which we read two, so it is cut to ``TIME``,
    ``POSITION`` and ``VELOCITY``, decimated to one sample every ``orbit_step`` seconds
    from its native 1 s.  2 s was measured, not guessed.  ``hdaxbary`` refuses to read a
    prefilter sampled more coarsely than about 10 s -- 15 s and up fail with "no bracketing
    sample found" -- and while 5 s and 10 s are read, they move the reference times by a
    full float64 ulp (59.6 ns at Swift's MET, most of the 100 ns budget) against the native
    sampling.  2 s is the coarsest step that is bit-identical to 1 s, and halves the file.

    The clock file keeps every interval from two before the most recent leap second back
    through two after the observation.  One interval would be enough to correct these
    events; the wider slice is what lets the tests check the two things that only show up
    at a boundary -- that the row containing a time is the one used, and that the tabulated
    correction steps by exactly one second at a leap second, because the UTCF converts to
    UTC and UTC is what steps.  The leap-second boundaries are found in the file itself
    rather than looked up, by evaluating each interval's polynomial at its own end and at
    the next interval's start: ordinary boundaries agree to a few microseconds, a leap
    second shows up as a one-second jump.
    """
    import numpy as np
    from astropy.io import fits

    evt_out = os.path.join(DATA, "dummy_swift_evt.evt")
    orb_out = os.path.join(DATA, "dummy_swift_orb.fits")
    clk_out = os.path.join(DATA, "dummy_swift_clk.fits")

    with fits.open(SWIFT_EVENTS) as hdul:
        events = hdul["EVENTS"]
        nrows = len(events.data)
        step = max(1, nrows // nevents)
        rows = slice(None, None, step)
        trimmed = subset_table(events, rows, columns=("TIME", "PI"))
        trimmed.header.add_history(
            f"Every {step}th row of {os.path.basename(SWIFT_EVENTS)}, by tools/make_test_data.py"
        )
        # Materialised, not a view: it is used after the file is closed.
        times = np.array(events.data["TIME"][rows])
        fits.HDUList([hdul[0].copy(), trimmed, hdul["GTI"].copy()]).writeto(evt_out, overwrite=True)
    print(f"    wrote {evt_out} ({len(times)} of {nrows} rows)")

    with fits.open(SWIFT_ORBIT) as hdul:
        orbit = hdul["PREFILTER"]
        t = orbit.data["TIME"]
        window = np.flatnonzero((t > times.min() - margin) & (t < times.max() + margin))
        keep = window[::orbit_step]
        trimmed = subset_table(orbit, keep, columns=("TIME", "POSITION", "VELOCITY"))
        trimmed.header["TSTART"] = float(t[keep].min())
        trimmed.header["TSTOP"] = float(t[keep].max())
        trimmed.header.add_history(
            f"Every {orbit_step}th row of {os.path.basename(SWIFT_ORBIT)} over the span of "
            f"{os.path.basename(evt_out)}, by tools/make_test_data.py"
        )
        fits.HDUList([hdul[0].copy(), trimmed]).writeto(orb_out, overwrite=True)
    print(f"    wrote {orb_out} ({len(keep)} of {len(t)} rows)")

    with fits.open(SWIFT_CLOCK) as hdul:
        clock = hdul[SWIFT_CLOCK_EXTENSION]
        starts = np.asarray(clock.data["TSTART"], dtype=float)
        stops = np.asarray(clock.data["TSTOP"], dtype=float)
        c0, c1, c2 = (np.asarray(clock.data[c], dtype=float) for c in ("C0", "C1", "C2"))

        def at(row, met):
            x = (met - starts[row]) / 86400.0
            return c0[row] + c1[row] * x + c2[row] * x**2

        here = int(np.searchsorted(starts, times.min(), side="right") - 1)
        steps = [i for i in range(here) if abs(at(i, stops[i]) - at(i + 1, starts[i + 1])) > 5e5]
        first = max(0, (steps[-1] if steps else here) - 2)
        keep = np.arange(first, min(len(starts), here + 3))
        trimmed = subset_table(clock, keep)
        trimmed.header.add_history(
            f"Intervals {keep[0]}-{keep[-1]} of {os.path.basename(SWIFT_CLOCK)}, "
            "spanning the last leap second before the observation, "
            "by tools/make_test_data.py"
        )
        fits.HDUList([hdul[0].copy(), trimmed]).writeto(clk_out, overwrite=True)
    print(f"    wrote {clk_out} ({len(keep)} of {len(starts)} intervals)")

    for raw in (orb_out,):
        subprocess.run(["gzip", "-9", "-f", raw], check=True)
    return evt_out, orb_out + ".gz", clk_out


def trim_fermi_inputs(nevents=400, margin=600.0):
    """Cut the Fermi event and spacecraft files down to something committable.

    The event list keeps ``TIME`` and ``ENERGY`` only, decimated so that the sample still
    spans the whole week, plus the ``GTI`` extension -- 70 intervals, which is what makes
    the reference able to check the GTIs as well as the events.  The spacecraft file is
    cut to ``START``, ``STOP`` and ``SC_POSITION`` over the span of the events and the
    GTIs together, plus a margin.

    Unlike the other missions here the spacecraft file is **not** decimated, and that was
    measured rather than assumed.  This file carries no ``SC_VELOCITY`` column -- older
    LAT spacecraft files do not, though current ones do -- so the velocity has to be
    obtained by differentiating the position, and a coarser grid makes a worse derivative.
    Against a ``gtbary`` reference regenerated at each step, the native 30 s sampling
    agrees to 29.8 ns, which is one unit in the last place of the reference's float64
    times and therefore the floor; 60 s gives 119.2 ns, 120 s gives 328 ns, and 240 s
    gives 4.4 us.  60 s would still pass, at 92 % of the tolerance and with no margin for
    another platform; 30 s costs 305 KB gzipped and is exact, so it is what is committed.
    """
    import numpy as np
    from astropy.io import fits

    evt_out = os.path.join(DATA, "dummy_fermi_evt.evt")
    orb_out = os.path.join(DATA, "dummy_fermi_orb.fits")

    with fits.open(FERMI_EVENTS) as hdul:
        events = hdul["EVENTS"]
        nrows = len(events.data)
        step = max(1, nrows // nevents)
        rows = slice(None, None, step)
        trimmed = subset_table(events, rows, columns=("TIME", "ENERGY"))
        trimmed.header.add_history(
            f"Every {step}th row of {os.path.basename(FERMI_EVENTS)}, by tools/make_test_data.py"
        )
        gti = hdul["GTI"].copy()
        # Materialised, not a view: both are used after the file is closed.
        times = np.array(events.data["TIME"][rows])
        starts, stops = np.array(gti.data["START"]), np.array(gti.data["STOP"])
        fits.HDUList([hdul[0].copy(), trimmed, gti]).writeto(evt_out, overwrite=True)
    print(f"    wrote {evt_out} ({len(times)} of {nrows} rows, {len(starts)} GTIs)")

    # The GTIs reach past the events at both ends -- they are the requested window, the
    # events are what arrived -- and they get barycentred too, so the spacecraft file has
    # to cover them as well or the new coverage check refuses the file.
    low = min(times.min(), starts.min()) - margin
    high = max(times.max(), stops.max()) + margin
    with fits.open(FERMI_ORBIT) as hdul:
        orbit = hdul["SC_DATA"]
        start = orbit.data["START"]
        keep = np.flatnonzero((start > low) & (start < high))
        trimmed = subset_table(orbit, keep, columns=("START", "STOP", "SC_POSITION"))
        trimmed.header["TSTART"] = float(start[keep].min())
        trimmed.header["TSTOP"] = float(start[keep].max())
        trimmed.header.add_history(
            f"Rows of {os.path.basename(FERMI_ORBIT)} covering the events, "
            "by tools/make_test_data.py"
        )
        fits.HDUList([hdul[0].copy(), trimmed]).writeto(orb_out, overwrite=True)
    print(f"    wrote {orb_out} ({len(keep)} of {len(start)} rows)")
    subprocess.run(["gzip", "-9", "-f", orb_out], check=True)


def trim_chandra_inputs(nevents=400, margin=1200.0):
    """Cut the Chandra event and orbit files down to something committable.

    The event list keeps ``time`` and ``pi`` only, decimated across the whole 5.6 h
    exposure, plus the ``GTI`` extension.  The lower-case column names are kept exactly as
    Chandra writes them -- ``time`` in the events, ``Time`` in the orbit file, against
    ``START``/``STOP`` in the GTI -- because that mixture is the thing worth having a
    committed file for: a case-sensitive column lookup leaves the events untouched while
    correcting the GTIs, which is a wrong answer rather than an error.

    The orbit file is kept at its own 300 s sampling.  Measured against the full file,
    interpolating on that grid costs ~1e-11 s at Chandra's 86000-113600 km, so there is
    nothing to gain from decimating and nothing to lose by not.
    """
    import numpy as np
    from astropy.io import fits

    evt_out = os.path.join(DATA, "dummy_chandra_evt.evt")
    orb_out = os.path.join(DATA, "dummy_chandra_orb.fits")

    with fits.open(CHANDRA_EVENTS) as hdul:
        events = hdul["EVENTS"]
        nrows = len(events.data)
        step = max(1, nrows // nevents)
        rows = slice(None, None, step)
        trimmed = subset_table(events, rows, columns=("time", "pi"))
        trimmed.header.add_history(
            f"Every {step}th row of {os.path.basename(CHANDRA_EVENTS)}, by tools/make_test_data.py"
        )
        # Materialised, not a view: it is used after the file is closed.
        times = np.array(events.data["time"][rows])
        fits.HDUList([hdul[0].copy(), trimmed, hdul["GTI"].copy()]).writeto(evt_out, overwrite=True)
    print(f"    wrote {evt_out} ({len(times)} of {nrows} rows)")

    with fits.open(CHANDRA_ORBIT) as hdul:
        orbit = hdul["ORBITEPHEM"]
        t = orbit.data["Time"]
        keep = np.flatnonzero((t > times.min() - margin) & (t < times.max() + margin))
        trimmed = subset_table(orbit, keep)
        trimmed.header["TSTART"] = float(t[keep].min())
        trimmed.header["TSTOP"] = float(t[keep].max())
        trimmed.header.add_history(
            f"Rows of {os.path.basename(CHANDRA_ORBIT)} spanning "
            f"{os.path.basename(evt_out)}, by tools/make_test_data.py"
        )
        fits.HDUList([hdul[0].copy(), trimmed]).writeto(orb_out, overwrite=True)
    print(f"    wrote {orb_out} ({len(keep)} of {len(t)} rows)")
    subprocess.run(["gzip", "-9", "-f", orb_out], check=True)
    return evt_out, orb_out + ".gz"


def run_axbary(infile, orbitfile, outfile, args):
    """Run CIAO ``axbary`` on ``infile``, writing an uncompressed ``outfile``.

    Unlike ``barycorr`` this needs no HEASOFT environment and no terminal, but it does
    need ``ASCDS_CALIB`` pointing at the directory holding ``tai-utc.dat`` and
    ``JPLEPH.*``.  Without it the task prints "Could not initialize bary stuff" and
    *still exits 0*, having copied the times through unchanged -- so the output is checked
    for having actually moved.
    """
    import numpy as np
    from astropy.io import fits

    workdir = tempfile.mkdtemp(prefix="axbary_")
    try:
        local_in = os.path.join(workdir, os.path.basename(infile))
        shutil.copy(infile, local_in)
        shutil.copy(orbitfile, workdir)
        pfiles = os.path.join(workdir, "pfiles")
        os.makedirs(pfiles)

        cmd = [
            os.path.join(CIAO, "bin", "axbary"),
            f"infile={os.path.basename(local_in)}",
            f"orbitfile={os.path.basename(orbitfile)}",
            f"outfile={os.path.basename(outfile)}",
            "clobber=yes",
            "mode=h",
        ]
        cmd += [f"{k}={v}" for k, v in args.items()]
        env = dict(os.environ)
        env["ASCDS_CALIB"] = os.path.join(CIAO, "data")
        env["PFILES"] = pfiles + ";" + os.path.join(CIAO, "param")
        # `axbary` is a shell wrapper around `pset`/`pget`, so CIAO's bin has to be on
        # PATH as well as being where the executable came from.  Without it the wrapper
        # reads no parameters at all and exits 0 having done nothing.
        env["PATH"] = os.path.join(CIAO, "bin") + os.pathsep + env.get("PATH", "")
        result = subprocess.run(
            cmd, cwd=workdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        output = result.stdout.decode(errors="replace")
        if result.returncode != 0 or "Could not" in output or "command not found" in output:
            raise RuntimeError(f"axbary failed:\n{output}")
        print(output.rstrip())

        produced = os.path.join(workdir, os.path.basename(outfile))
        with fits.open(local_in) as before, fits.open(produced) as after:
            moved = np.abs(after["EVENTS"].data["time"] - before["EVENTS"].data["time"])
        if moved.max() < 1.0:
            raise RuntimeError(
                f"axbary moved the times by at most {moved.max():g} s: it ran without "
                "an ephemeris and the output is the input"
            )
        print(f"    times moved by {moved.mean():.3f} s")
        shutil.move(produced, outfile)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_gtbary(infile, orbitfile, outfile, args):
    """Run Fermi ``gtbary`` on ``infile``, writing an uncompressed ``outfile``.

    ``gtbary`` needs no terminal and no ScienceTools init script, but it does need
    ``$TIMING_DIR`` (or ``$LHEA_DATA``) pointing at the directory holding ``JPLEPH.405``.
    Without it the task says so and exits non-zero, so unlike ``axbary`` there is no
    silent-passthrough failure mode to guard against -- but the output is still checked
    for having moved, because a reference that is its own input would pass every test.
    """
    import numpy as np
    from astropy.io import fits

    workdir = tempfile.mkdtemp(prefix="gtbary_")
    try:
        local_in = os.path.join(workdir, os.path.basename(infile))
        shutil.copy(infile, local_in)
        shutil.copy(orbitfile, workdir)
        pfiles = os.path.join(workdir, "pfiles")
        os.makedirs(pfiles)

        fermi_share = os.path.join(FERMI_ENV, "share", "fermitools")
        cmd = [
            os.path.join(FERMI_ENV, "bin", "gtbary"),
            f"evfile={os.path.basename(local_in)}",
            f"scfile={os.path.basename(orbitfile)}",
            f"outfile={os.path.basename(outfile)}",
            "tcorrect=BARY",
            "clobber=yes",
            "mode=h",
        ]
        cmd += [f"{k}={v}" for k, v in args.items()]
        env = dict(os.environ)
        env["PFILES"] = pfiles + ";" + os.path.join(fermi_share, "syspfiles")
        env["TIMING_DIR"] = os.path.join(fermi_share, "refdata", "fermi", "jplephem")
        result = subprocess.run(
            cmd, cwd=workdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        output = result.stdout.decode(errors="replace")
        if result.returncode != 0 or "Caught" in output:
            raise RuntimeError(f"gtbary failed:\n{output}")
        print(output.rstrip())

        produced = os.path.join(workdir, os.path.basename(outfile))
        with fits.open(local_in) as before, fits.open(produced) as after:
            moved = np.abs(after["EVENTS"].data["TIME"] - before["EVENTS"].data["TIME"])
        if moved.max() < 1.0:
            raise RuntimeError(
                f"gtbary moved the times by at most {moved.max():g} s: the output is the input"
            )
        print(f"    times moved by {moved.mean():.3f} s")
        shutil.move(produced, outfile)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_barycen(infile, outfile, args):
    """Run SAS ``barycen`` on a copy of ``infile``, writing an uncompressed ``outfile``.

    Two things make this more involved than ``barycorr``:

    * ``barycen`` has no output parameter -- it edits the table it is given, irreversibly.
      So the input is copied into a private working directory and the result moved out, and
      neither the committed input nor the observation directory is ever written to.
    * It will not take an orbit file on the command line.  It reaches the spacecraft
      position through SAS's observation access layer, which means an *ingested* ODF, and
      an ingested ODF's summary records absolute paths -- so a copy of the ODF is
      re-ingested inside the working directory.  That is the only reason ``odfingest``
      appears here.  No CCF is needed: checked by running with ``SAS_CCF`` and
      ``SAS_CCFPATH`` unset and comparing the output bit for bit.
    """
    if not SAS_DIRS:
        raise RuntimeError("No SAS installation found under ~/devel/SAS/xmmsas_*")
    workdir = tempfile.mkdtemp(prefix="barycen_")
    try:
        local_in = os.path.join(workdir, os.path.basename(outfile))
        shutil.copy(infile, local_in)
        odf = os.path.join(workdir, "odf")
        shutil.copytree(XMM_ODF, odf)
        # The summary is what carries the stale absolute paths; odfingest rewrites it.
        for stale in glob.glob(os.path.join(odf, "*SUM.SAS")):
            os.remove(stale)

        table = f"{os.path.basename(local_in)}:EVENTS"
        cmd = ["barycen", "withtable=yes", f"table={table}", "timecolumn=TIME"]
        cmd += [f"{k}={v}" for k, v in args.items()]
        run_sas(cmd, workdir, odf)
        shutil.move(local_in, outfile)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_sas(cmd, cwd, odf):
    """Run a SAS task, after ingesting the ODF in ``odf`` so it can be found.

    ``setsas.sh`` needs HEASOFT initialised before it, refuses to run unless ``SAS_DIR``
    is exported, and tries to raise the stack limit -- which fails harmlessly in a
    sandbox, so the script must not be run under ``set -e``.
    """
    script = (
        f'export HEADAS={shlex.quote(HEADAS)}; . "$HEADAS/headas-init.sh"; '
        f'export SAS_DIR={shlex.quote(SAS_DIRS[-1])}; . "$SAS_DIR/setsas.sh"; '
        f"export SAS_ODF={shlex.quote(odf)}; "
        f"odfingest withodfdir=no outdir={shlex.quote(odf)} "
        "usecanonicalname=yes writepath=yes -w 1 -V 2 && "
        f'export SAS_ODF="$(ls {shlex.quote(odf)}/*SUM.SAS)" && '
        + " ".join(shlex.quote(part) for part in cmd)
    )
    result = subprocess.run(
        ["bash", "-c", script], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    output = result.stdout.decode(errors="replace")
    if result.returncode != 0 or "error (" in output:
        raise RuntimeError(f"{cmd[0]} failed:\n{output}")
    print("\n".join(line for line in output.splitlines() if ":- " in line or "warning" in line))


def run_barycorr(infile, orbitfiles, outfile, args, extra_inputs=()):
    """Run HEASOFT barycorr on ``infile``, writing an uncompressed ``outfile``."""
    workdir = tempfile.mkdtemp(prefix="barycorr_")
    try:
        local_in = os.path.join(workdir, os.path.basename(infile))
        shutil.copy(infile, local_in)
        orbnames = []
        for orb in orbitfiles:
            shutil.copy(orb, workdir)
            orbnames.append(os.path.basename(orb))
        # Clock files and anything else named in args, which barycorr opens by name.
        for extra in extra_inputs:
            shutil.copy(extra, workdir)

        pfiles = os.path.join(workdir, "pfiles")
        os.makedirs(pfiles)

        cmd = [
            "barycorr",
            f"infile={os.path.basename(local_in)}",
            f"outfile={os.path.basename(outfile)}",
            f"orbitfiles={','.join(orbnames)}",
        ]
        cmd += [f"{k}={v}" for k, v in args.items()]
        run_heasoft(cmd, workdir, pfiles)
        shutil.move(os.path.join(workdir, os.path.basename(outfile)), outfile)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_heasoft(cmd, cwd, pfiles):
    """Run a HEASOFT task, giving it the controlling terminal it insists on.

    HEASOFT tasks open ``/dev/tty`` to redirect their prompts and abort with
    ``ERROR: Device not configured`` when there is none -- which there is not, in a
    non-interactive shell.  Wrapping the call in ``script -q /dev/null`` only works when
    *that* has a terminal to start from, so instead we make one: a pseudo-terminal, with
    the child put in its own session and the pty made its controlling terminal.

    The environment has to come from ``headas-init.sh``; setting ``HEADAS`` alone leaves
    the libraries unfindable and the task refuses to start.
    """
    import fcntl
    import termios

    master, slave = pty.openpty()

    def become_session_leader():
        os.setsid()
        fcntl.ioctl(slave, termios.TIOCSCTTY, 0)

    quoted = " ".join(shlex.quote(part) for part in cmd)
    script = (
        f'export HEADAS={shlex.quote(HEADAS)}; . "$HEADAS/headas-init.sh"; '
        f"export PFILES={shlex.quote(pfiles + ';' + HEADAS + '/syspfiles')}; {quoted}"
    )
    try:
        result = subprocess.run(
            ["bash", "-c", script],
            cwd=cwd,
            stdin=slave,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=become_session_leader,
        )
    finally:
        os.close(slave)
        os.close(master)

    output = result.stdout.decode(errors="replace")
    # HEASOFT tasks are fond of exiting 0 after saying "terminating with status 1".
    if result.returncode != 0 or "terminating with status" in output:
        raise RuntimeError(f"{cmd[0]} failed:\n{output}")
    print(output.rstrip())


def main():
    if not os.path.exists(os.path.join(DATA, "dummy_fine_clk.fits")):
        print("--- dummy_fine_clk.fits")
        trim_clock_file()
    if not os.path.exists(os.path.join(DATA, "dummy_xte_evt.evt")):
        print("--- dummy_xte_evt.evt, dummy_xte_orb.fits.gz")
        trim_rxte_inputs()
    if not os.path.exists(os.path.join(DATA, "dummy_xmm_evt.evt")):
        print("--- dummy_xmm_evt.evt, dummy_xmm_orb.fits.gz")
        trim_xmm_inputs()
    if not os.path.exists(os.path.join(DATA, "dummy_chandra_evt.evt")):
        print("--- dummy_chandra_evt.evt, dummy_chandra_orb.fits.gz")
        trim_chandra_inputs()
    if not os.path.exists(os.path.join(DATA, "dummy_swift_evt.evt")):
        print("--- dummy_swift_evt.evt, dummy_swift_orb.fits.gz, dummy_swift_clk.fits")
        trim_swift_inputs()
    if not os.path.exists(os.path.join(DATA, "dummy_fermi_evt.evt")):
        print("--- dummy_fermi_evt.evt, dummy_fermi_orb.fits.gz")
        trim_fermi_inputs()
    for name, spec in REFERENCES.items():
        target = os.path.join(DATA, name)
        raw = target[: -len(".gz")] if target.endswith(".gz") else target
        print(f"--- {name}")
        run_barycorr(
            os.path.join(DATA, spec["infile"]),
            [os.path.join(DATA, o) for o in spec["orbitfiles"]],
            raw,
            spec["args"],
            extra_inputs=[os.path.join(DATA, e) for e in spec.get("extra_inputs", ())],
        )
        if target.endswith(".gz"):
            subprocess.run(["gzip", "-9", "-f", raw], check=True)
        print(f"    wrote {target}")

    for name, spec in CHANDRA_REFERENCES.items():
        target = os.path.join(DATA, name)
        raw = target[: -len(".gz")]
        print(f"--- {name}")
        run_axbary(
            os.path.join(DATA, "dummy_chandra_evt.evt"),
            os.path.join(DATA, "dummy_chandra_orb.fits.gz"),
            raw,
            spec["args"],
        )
        subprocess.run(["gzip", "-9", "-f", raw], check=True)
        print(f"    wrote {target}")

    target = os.path.join(DATA, FERMI_REFERENCE["outfile"])
    raw = target[: -len(".gz")]
    print(f"--- {FERMI_REFERENCE['outfile']}")
    run_gtbary(
        os.path.join(DATA, FERMI_REFERENCE["infile"]),
        os.path.join(DATA, FERMI_REFERENCE["orbitfile"]),
        raw,
        FERMI_REFERENCE["args"],
    )
    subprocess.run(["gzip", "-9", "-f", raw], check=True)
    print(f"    wrote {target}")

    target = os.path.join(DATA, XMM_REFERENCE["outfile"])
    raw = target[: -len(".gz")]
    print(f"--- {XMM_REFERENCE['outfile']}")
    run_barycen(os.path.join(DATA, XMM_REFERENCE["infile"]), raw, XMM_REFERENCE["args"])
    subprocess.run(["gzip", "-9", "-f", raw], check=True)
    print(f"    wrote {target}")


if __name__ == "__main__":
    sys.exit(main())
