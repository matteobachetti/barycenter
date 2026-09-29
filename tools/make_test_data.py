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

import os
import pty
import shlex
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.abspath(os.path.dirname(__file__))
DATA = os.path.join(HERE, os.pardir, "tests", "data")
HEADAS = os.path.expanduser("~/mamba/envs/henv313_x86/heasoft")

#: The CALDB clock file the fine-clock reference is made with.  It is 12 MB, so only the
#: ~90 rows covering the test observation are committed, as ``dummy_fine_clk.fits``; see
#: :func:`trim_clock_file`.
CALDB_CLOCK = os.path.expanduser(
    "~/devel/CALDB/data/nustar/fpm/bcf/clock/nuCclock20100101v230.fits.gz"
)

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


if __name__ == "__main__":
    sys.exit(main())
