"""Output-file handling in the ``--apply-official`` path.

The tools themselves need HEASOFT and are never run in CI, so these cover only the two
steps around them: refusing to clobber, and delivering the finished file.
"""

import os

import numpy as np
import pytest
from astropy.io import fits

from barycenter.official import (
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
