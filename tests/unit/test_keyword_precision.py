"""Reading a reference epoch that an old Fortran tool wrote.

`MJDREF` decides where every time in a file sits, so a header the reader cannot parse is
not a cosmetic problem: the keywords computed from `TSTART` and `TSTOP` are then left
saying something the corrected times contradict. Fermi's own tutorial file is written this
way, which is what these tests are drawn from.
"""

import numpy as np
import pytest
from astropy.io import fits

from barycenter.utils import high_precision_keyword_read, high_precision_mjdref

#: What Fermi's `fakepulsar_event.fits` actually holds: a *string*, in Fortran's
#: D-exponent notation, where a FITS float is expected.
FERMI_MJDREFF = "7.428703703703703D-4"


class TestFortranExponents:
    """A `D` exponent is a real number, and has to be read as one."""

    def test_a_d_exponent_is_read(self):
        """The Fermi case: a string keyword whose exponent marker is D rather than E."""
        header = {"MJDREFI": 51910.0, "MJDREFF": FERMI_MJDREFF}
        assert high_precision_keyword_read(header, "MJDREF") == pytest.approx(
            51910.00074287037, abs=1e-12
        )

    def test_the_fractional_part_keeps_its_precision(self):
        """Summing in longdouble is the whole point of the split, so it must survive parsing."""
        header = {"MJDREFI": 51910.0, "MJDREFF": FERMI_MJDREFF}
        value = high_precision_keyword_read(header, "MJDREF")
        assert isinstance(value, np.longdouble)
        # The fraction is recoverable to far better than a float64 MJD could express.
        assert float(value - np.longdouble(51910.0)) == pytest.approx(7.428703703703703e-4)

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("1D0", 1.0),
            ("1.5D+3", 1500.0),
            ("-2.5d-2", -0.025),
            (".5D1", 5.0),
            ("3.0E-2", 0.03),  # ordinary E notation still works
            ("42", 42.0),  # and a plain integer string
        ],
    )
    def test_the_forms_that_turn_up(self, text, expected):
        """Case, sign and a missing leading digit all appear in real Fortran output."""
        assert high_precision_keyword_read({"MJDREF": text}, "MJDREF") == pytest.approx(expected)

    def test_a_numeric_keyword_is_untouched(self):
        """The overwhelmingly common case must not go anywhere near the string path."""
        assert high_precision_keyword_read({"MJDREF": 51910.5}, "MJDREF") == 51910.5

    def test_nonsense_is_still_refused(self):
        """Substituting D for E must not turn an unparseable keyword into a plausible number."""
        with pytest.raises(ValueError):
            high_precision_keyword_read({"MJDREF": "not a number"}, "MJDREF")

    def test_a_date_like_string_is_not_read_as_a_number(self):
        """`51910-01-02` must not become a number; a wrong MJDREF is a half-hour error."""
        with pytest.raises(ValueError):
            high_precision_keyword_read({"MJDREF": "51910-01-02"}, "MJDREF")


class TestThroughAFitsHeader:
    """The same, through the object the code actually gets handed."""

    def test_a_real_fits_header_is_read(self):
        """astropy hands back a str for such a card, which is where the failure came from."""
        header = fits.Header()
        header["MJDREFI"] = 51910.0
        header["MJDREFF"] = FERMI_MJDREFF
        assert high_precision_mjdref(header) == pytest.approx(51910.00074287037, abs=1e-12)

    def test_a_header_with_no_mjdref_still_raises(self):
        """Guessing a reference epoch is never right, and that must not have changed."""
        with pytest.raises(ValueError, match="no MJDREF"):
            high_precision_mjdref(fits.Header())
