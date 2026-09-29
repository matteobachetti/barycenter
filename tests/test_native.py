"""Tests for the native (astropy + ERFA + JPL) barycentring engine.

The headline test is :func:`TestAgainstBarycorr.test_agrees_with_barycorr`, which is
the whole point of the engine: it must reproduce HEASOFT ``barycorr`` to 100 ns.
"""

import os

import numpy as np
import pytest
from astropy.io import fits

from barycenter.native import (
    T_SUN,
    barycentric_correction,
    ephemeris_frame,
    source_unit_vector,
    spacecraft_interpolator,
)

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")

#: True where numpy's longdouble is wider than float64. The native engine does not
#: care, but PINT does, so the cross-check against it can only run where PINT itself
#: is precise.
HAS_EXTENDED_PRECISION = np.finfo(np.longdouble).eps < np.finfo(np.float64).eps

#: The science target. The native engine reaches it with room to spare: the residual
#: against the reference is 0, 1 or 2 units in the last place of the stored float64
#: times (29.8 ns each), so the reference cannot resolve any difference finer than
#: that in the first place.
TOLERANCE_S = 1e-7

#: Coordinates and ephemeris the reference was generated with. See
#: tools/make_test_data.py.
REF_RA, REF_DEC = 294.9107, 21.58308
REF_EPHEM = "de440"
#: MJDREFI + MJDREFF of the NuSTAR test file, as an exact decimal.
MJDREF = np.longdouble("55197") + np.longdouble("0.00076601852")


def nustar_orbit(fname):
    """Position and velocity from a NuSTAR orbit file, converted from km to metres."""
    with fits.open(fname) as hdul:
        data = hdul[1].data
        met = np.asarray(data["TIME"], dtype=np.float64)
        pos = np.asarray(data["POSITION"], dtype=np.float64) * 1000.0
        vel = np.asarray(data["VELOCITY"], dtype=np.float64) * 1000.0
    return met, pos, vel


class TestAgainstBarycorr:
    @classmethod
    def setup_class(cls):
        cls.evfile = os.path.join(datadir, "dummy_evt.evt")
        cls.orbfile = os.path.join(datadir, "dummy_orb.fits.gz")
        cls.reffile = os.path.join(datadir, "dummy_evt_bary_DE440_noclk.evt.gz")

        cls.met = fits.getdata(cls.evfile, 1)["TIME"].astype(np.float64)
        cls.ref = fits.getdata(cls.reffile, 1)["TIME"].astype(np.float64)
        cls.sc = spacecraft_interpolator(*nustar_orbit(cls.orbfile))

    def correction(self, **kwargs):
        kwargs.setdefault("ephem", REF_EPHEM)
        return barycentric_correction(self.met, MJDREF, REF_RA, REF_DEC, self.sc, **kwargs)

    def test_agrees_with_barycorr(self):
        """The native correction reproduces HEASOFT barycorr to better than 100 ns.

        This is the reason the engine exists; everything else in this file explains
        the number.
        """
        diff = self.met + self.correction() - self.ref
        assert np.max(np.abs(diff)) < TOLERANCE_S, (
            f"max |difference| = {np.max(np.abs(diff)) * 1e9:.1f} ns "
            f"(mean {diff.mean() * 1e9:+.1f} ns)"
        )

    def test_residual_is_only_rounding_of_the_stored_times(self):
        """What disagreement is left is float64 granularity, not physics.

        The reference times are around 1.8e8 s, where one float64 step is 29.8 ns, so
        the difference can only ever be a whole number of those steps. Seeing exactly
        that means there is no residual structure left to chase.
        """
        diff = self.met + self.correction() - self.ref
        ulp = np.spacing(self.ref).mean()
        in_ulps = np.unique(np.round(diff / ulp, 6))
        assert np.array_equal(in_ulps, np.round(in_ulps)), in_ulps
        assert in_ulps.max() <= 2

    def test_result_does_not_depend_on_extended_precision(self):
        """Nothing here needs an 80-bit longdouble.

        Every term is a small number of seconds computed in float64, so the answer
        is identical on Apple Silicon and on x86. Rounding MJDREF to a plain float64
        -- which loses about a microsecond of the absolute epoch, the worst a
        platform without extended precision can do to it -- moves the correction by
        only 7 picoseconds, because it only shifts the point at which smooth
        functions are evaluated. The PINT engine, which subtracts two absolute MJDs,
        loses 1.1 us to the same thing.
        """
        exact = self.correction()
        degraded = barycentric_correction(
            self.met, np.float64(MJDREF), REF_RA, REF_DEC, self.sc, ephem=REF_EPHEM
        )
        assert np.max(np.abs(exact - degraded)) < 1e-10

    def test_shapiro_convention_is_the_axbary_one(self):
        """The two Shapiro conventions differ by 2*T_sun*ln(r/AU), about 96 ns here.

        PINT normalises the logarithm by the astronomical unit and axBary by the
        observer's distance from the Sun. The difference is an annual term, and it is
        the whole reason the PINT engine sits 100 ns away from barycorr.
        """
        axbary = self.correction(shapiro="axbary")
        pint_like = self.correction(shapiro="pint")
        assert np.allclose(pint_like - axbary, 9.5e-8, atol=5e-9)
        # ... and the axBary one is the one that matches the reference.
        assert abs(np.mean(self.met + axbary - self.ref)) < abs(
            np.mean(self.met + pint_like - self.ref)
        )

    def test_shapiro_is_not_negligible(self):
        """Dropping the Shapiro term costs ~4.7 us, far above the target."""
        with_it = self.correction(shapiro="axbary")
        without = self.correction(shapiro="none")
        assert 1e-6 < np.mean(with_it - without) < 1e-5

    def test_a_coarse_grid_is_good_enough(self):
        """The correction may be evaluated on a 5 s grid and interpolated.

        It is smooth on the scale of the spacecraft orbit, so a spline through a
        coarse grid costs a nanosecond or two and saves evaluating the ephemeris once
        per event.
        """
        from scipy.interpolate import CubicSpline

        exact = self.correction()
        grid = np.arange(self.met.min() - 5, self.met.max() + 10, 5.0)
        on_grid = barycentric_correction(grid, MJDREF, REF_RA, REF_DEC, self.sc, ephem=REF_EPHEM)
        assert np.max(np.abs(CubicSpline(grid, on_grid)(self.met) - exact)) < 5e-9

    @pytest.mark.skipif(
        not HAS_EXTENDED_PRECISION,
        reason="PINT is quantised at 1.1 us without an 80-bit longdouble, so it "
        "cannot be compared with anything at the nanosecond level",
    )
    def test_matches_pint_once_the_conventions_agree(self):
        """Native and PINT agree to about a nanosecond on the same ephemeris.

        Two independent implementations of the same physics; the only systematic
        between them is the Shapiro convention, which this test removes.
        """
        pint_toa = pytest.importorskip("pint.toa")
        import astropy.units as u
        from pint.models import StandardTimingModel

        from barycenter.orbit import read_orbit
        from barycenter.pintengine import TableSatelliteObs

        model = StandardTimingModel
        model.RAJ.quantity = REF_RA * u.deg
        model.DECJ.quantity = REF_DEC * u.deg
        model.DM.quantity = 0.0
        model.EPHEM.value = "DE440"
        # The same orbit table the native engine above was given, registered with PINT
        # by our own SatelliteObs subclass: no patching of PINT's internals.
        TableSatelliteObs("nustar", read_orbit(self.orbfile), overwrite=True)

        mjds = np.longdouble(self.met) / 86400 + MJDREF
        toas = [pint_toa.TOA(np.float64(m), obs="nustar", scale="tt") for m in mjds]
        ts = pint_toa.get_TOAs_list(
            toas, ephem="DE440", include_bipm=False, planets=False, tdb_method="default"
        )
        bats = model.get_barycentric_toas(ts)
        used = np.array([t.mjd_long for t in ts.get_mjds(high_precision=True)], dtype=np.longdouble)
        pint_corr = np.asarray((bats.to_value(u.d) - used) * 86400, dtype=np.float64)

        diff = self.correction(shapiro="pint") - pint_corr
        assert np.max(np.abs(diff)) < 5e-9, f"max {np.max(np.abs(diff)) * 1e9:.2f} ns"


class TestPieces:
    def test_ephemeris_frame(self):
        """DE200 is an FK5 ephemeris; DE405 and later are ICRF-aligned."""
        assert ephemeris_frame("DE200") == "fk5"
        assert ephemeris_frame("de440") == "icrs"
        assert ephemeris_frame("/some/path/de200.bsp") == "fk5"

    def test_frame_mismatch_is_worth_tens_of_microseconds(self):
        """FK5 and ICRS differ by ~20 mas, which is ~45 us of Roemer delay.

        Small enough to look like a rounding error in the coordinates, large enough to
        ruin the answer; this is why the frame is matched to the ephemeris rather than
        assumed.
        """
        n_icrs = source_unit_vector(83.6331, 22.0145, frame="fk5", ephem_frame="icrs")
        n_fk5 = source_unit_vector(83.6331, 22.0145, frame="fk5", ephem_frame="fk5")
        angle = np.arccos(np.clip(n_icrs @ n_fk5, -1, 1))
        earth_orbit_m = 1.496e11
        assert 1e-5 < angle * earth_orbit_m / 2.998e8 < 1e-3

    def test_no_rotation_when_the_frames_match(self):
        """Coordinates already in the ephemeris's frame are left alone exactly."""
        a = source_unit_vector(294.9107, 21.58308, frame="icrs", ephem_frame="icrs")
        b = source_unit_vector(294.9107, 21.58308, frame="ICRS", ephem_frame="ICRS")
        assert np.array_equal(a, b)
        assert np.isclose(np.linalg.norm(a), 1.0)

    def test_t_sun(self):
        """GM_sun / c**3 is 4.9255 us; the Shapiro term is a multiple of it."""
        assert np.isclose(T_SUN, 4.925490947e-6, rtol=1e-6)

    def test_interpolator_drops_bad_rows(self):
        """Repeated, out-of-order and all-zero orbit rows are discarded.

        Real orbit files contain all three, and a spline through a repeated abscissa
        is undefined.
        """
        met = np.array([0.0, 10.0, 10.0, 5.0, 20.0])
        pos = np.array([[0.0, 0, 0], [1e6, 0, 0], [1e6, 0, 0], [5e5, 0, 0], [2e6, 0, 0]])
        fun = spacecraft_interpolator(met, pos)
        # The all-zero row at t=0 is dropped, so the spline starts at t=5.
        assert np.allclose(fun([5.0, 10.0, 20.0])[:, 0], [5e5, 1e6, 2e6])

    def test_velocity_makes_the_interpolation_hermite(self):
        """Given velocities, the spline reproduces the tabulated slope exactly.

        That is what distinguishes the Hermite spline from a plain one, and it is why
        we pass the velocity column through: the orbit file already knows the slope,
        so there is no reason to let the fit guess it.
        """
        met = np.arange(0.0, 100.0, 10.0)
        zero = np.zeros_like(met)
        # cos, not sin: a sample whose position is exactly (0, 0, 0) is treated as a
        # dropped row, which is right for a spacecraft but wrong for a toy sine.
        pos = np.column_stack([1e6 * np.cos(met / 50), zero, zero])
        vel = np.column_stack([-1e6 / 50 * np.sin(met / 50), zero, zero])

        with_v = spacecraft_interpolator(met, pos, vel)
        assert np.allclose(with_v.derivative()(met)[:, 0], vel[:, 0])

        without = spacecraft_interpolator(met, pos)
        assert not np.allclose(without.derivative()(met)[:, 0], vel[:, 0])
