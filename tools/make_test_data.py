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
* HEASOFT tasks need a controlling terminal.  From a non-interactive shell they
  die with ``ERROR: Device not configured``, so every call is wrapped in
  ``script -q /dev/null``.
* Each run gets a private PFILES directory, otherwise concurrent runs clobber
  each other's parameter files.
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.abspath(os.path.dirname(__file__))
DATA = os.path.join(HERE, os.pardir, "tests", "data")
HEADAS = os.path.expanduser("~/mamba/envs/henv313_x86/heasoft")

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
}


def run_barycorr(infile, orbitfiles, outfile, args):
    """Run HEASOFT barycorr on ``infile``, writing an uncompressed ``outfile``."""
    workdir = tempfile.mkdtemp(prefix="barycorr_")
    try:
        local_in = os.path.join(workdir, os.path.basename(infile))
        shutil.copy(infile, local_in)
        orbnames = []
        for orb in orbitfiles:
            shutil.copy(orb, workdir)
            orbnames.append(os.path.basename(orb))

        pfiles = os.path.join(workdir, "pfiles")
        os.makedirs(pfiles)
        env = dict(os.environ)
        env["HEADAS"] = HEADAS
        env["PFILES"] = f"{pfiles};{HEADAS}/syspfiles"

        cmd = [
            os.path.join(HEADAS, "bin", "barycorr"),
            f"infile={os.path.basename(local_in)}",
            f"outfile={os.path.basename(outfile)}",
            f"orbitfiles={','.join(orbnames)}",
        ]
        cmd += [f"{k}={v}" for k, v in args.items()]
        # HEASOFT insists on a tty for its prompts; "script" provides one.
        subprocess.run(
            ["script", "-q", "/dev/null"] + cmd, cwd=workdir, env=env, check=True
        )
        shutil.move(os.path.join(workdir, os.path.basename(outfile)), outfile)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    for name, spec in REFERENCES.items():
        target = os.path.join(DATA, name)
        raw = target[: -len(".gz")] if target.endswith(".gz") else target
        print(f"--- {name}")
        run_barycorr(
            os.path.join(DATA, spec["infile"]),
            [os.path.join(DATA, o) for o in spec["orbitfiles"]],
            raw,
            spec["args"],
        )
        if target.endswith(".gz"):
            subprocess.run(["gzip", "-9", "-f", raw], check=True)
        print(f"    wrote {target}")


if __name__ == "__main__":
    sys.exit(main())
