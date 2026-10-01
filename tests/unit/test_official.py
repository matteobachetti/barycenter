"""Output-file handling in the ``--apply-official`` path.

The tools themselves need HEASOFT and are never run in CI, so these cover only the two
steps around them: refusing to clobber, and delivering the finished file.
"""

import os
import sys

import numpy as np
import pytest
from astropy.io import fits

from barycenter.official import (
    _copy_decompressing,
    _deliver,
    _refuse_to_clobber,
    apply_mission_specific_barycenter_correction,
)


def write_event_file(path, telescope):
    """A FITS file with just enough for ``mission_for`` to recognise the telescope."""
    primary = fits.PrimaryHDU()
    events = fits.BinTableHDU.from_columns(
        [fits.Column(name="TIME", format="D", array=np.array([1.0, 2.0, 3.0]))],
        name="EVENTS",
    )
    events.header["TELESCOP"] = telescope
    fits.HDUList([primary, events]).writeto(path, overwrite=True)
    return str(path)


class TestRefusingToClobber:
    def test_an_existing_file_is_refused(self):
        """Without overwrite=True, an existing output must stop the run."""
        with pytest.raises(FileExistsError, match="already exists"):
            _refuse_to_clobber(__file__, overwrite=False)

    def test_overwrite_allows_it(self):
        """With overwrite=True the same file passes."""
        _refuse_to_clobber(__file__, overwrite=True)

    def test_the_renamed_destination_is_rechecked(self, tmp_path):
        """ASCA's tool forces DE200, rewriting the output name -- recheck the new one.

        The first guard ran against the name the caller passed, so the rewritten
        destination used to be clobbered by the final move whatever ``overwrite`` said.
        """
        infile = write_event_file(tmp_path / "asca.evt", "ASCA")
        (tmp_path / "bary_DE200.evt").write_text("precious")
        with pytest.raises(FileExistsError, match="DE200"):
            apply_mission_specific_barycenter_correction(
                infile,
                orbfile="unused.orbit",
                outfile=str(tmp_path / "bary_DE440.evt"),
                ra=10.0,
                dec=20.0,
                ephem="DE440",
                overwrite=False,
            )
        assert (tmp_path / "bary_DE200.evt").read_text() == "precious"


class TestDelivering:
    def test_the_finished_file_lands_at_the_output_name(self, tmp_path):
        """The whole temporary file is moved when no column selection is asked for."""
        temp = write_event_file(tmp_path / "temp.evt", "NuSTAR")
        outfile = str(tmp_path / "out.evt")
        assert _deliver(temp, outfile) == outfile
        assert os.path.exists(outfile)
        assert not os.path.exists(temp)

    def test_delivery_survives_a_cross_device_temporary_directory(self, tmp_path, monkeypatch):
        """The temporary file is in /tmp, which is a separate filesystem on many machines.

        ``os.rename`` raises EXDEV across filesystems, so the delivery must not use it;
        this makes every ``os.rename`` fail the way a tmpfs /tmp would.
        """

        def no_rename(*args, **kwargs):
            raise OSError(18, "Invalid cross-device link")

        monkeypatch.setattr(os, "rename", no_rename)
        temp = write_event_file(tmp_path / "temp.evt", "NuSTAR")
        outfile = str(tmp_path / "out.evt")
        _deliver(temp, outfile)
        assert os.path.exists(outfile)


class TestCopyDecompressing:
    """``timeconv`` edits its file in place, so it gets a private, uncompressed copy."""

    @pytest.mark.parametrize("gzipped", [True, False])
    def test_the_input_directory_is_left_alone(self, tmp_path, gzipped):
        """The copy holds the uncompressed bytes; the (read-only) input dir is untouched."""
        import gzip

        indir = tmp_path / "in"
        indir.mkdir()
        payload = b"pretend FITS bytes"
        name = "asca.evt.gz" if gzipped else "asca.evt"
        (indir / name).write_bytes(gzip.compress(payload) if gzipped else payload)
        indir.chmod(0o555)
        dest = tmp_path / "copy.evt"
        try:
            _copy_decompressing(str(indir / name), str(dest))
        finally:
            indir.chmod(0o755)
        assert dest.read_bytes() == payload
        assert os.listdir(indir) == [name]


class TestRunningTimeconv:
    """ASCA's ``timeconv`` must work from any directory and leave only the output behind."""

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="the stand-in is a shell script, and HEASOFT does not run on Windows anyway",
    )
    def test_timeconv_runs_in_isolation(self, tmp_path, monkeypatch):
        """A stand-in ``timeconv`` checks that every file it is handed exists where it runs.

        The reference files used to be downloaded next to the output but named relative to
        the current directory, so a run only worked when those two were the same.
        """
        stub_dir = tmp_path / "bin"
        stub_dir.mkdir()
        log = tmp_path / "timeconv.log"
        stub = stub_dir / "timeconv"
        stub.write_text(
            "#!/bin/sh\n"
            f'echo "$@" > {log}\n'
            'for f in "$1" "$5" "$6"; do [ -s "$f" ] || exit 1; done\n'
        )
        stub.chmod(0o755)
        monkeypatch.setenv("PATH", f"{stub_dir}{os.pathsep}{os.environ['PATH']}")

        refs = tmp_path / "refs"
        refs.mkdir()
        (refs / "geo").write_text("earth")
        (refs / "orb").write_text("orbit")
        monkeypatch.setattr(
            "barycenter.official._asca_reference_files",
            lambda: (str(refs / "geo"), str(refs / "orb")),
        )

        infile = write_event_file(tmp_path / "asca.evt", "ASCA")
        outdir = tmp_path / "out"
        outdir.mkdir()
        workdir = tmp_path / "work"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        out = apply_mission_specific_barycenter_correction(
            infile,
            orbfile=None,
            outfile=str(outdir / "bary_DE200.evt"),
            ra=10.0,
            dec=20.0,
            ephem="DE200",
        )

        assert os.listdir(outdir) == ["bary_DE200.evt"]
        assert os.listdir(workdir) == []
        assert "/" not in log.read_text()
        with fits.open(out) as hdul:
            assert hdul[1].header["TIMESYS"] == "TDB"
