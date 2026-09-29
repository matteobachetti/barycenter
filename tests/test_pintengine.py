"""Tests for the optional PINT engine.

The point of these is that removing the monkey patch changed nothing: our own
``SatelliteObs`` subclass has to put the spacecraft in exactly the same place PINT's own
loader does, or the cross-check between the two engines means nothing.
"""

import os

import numpy as np
import pytest
from astropy.time import Time

from barycenter.orbit import read_orbit

pytest.importorskip("pint")

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "data")

NUSTAR_ORBIT = os.path.join(datadir, "dummy_orb.fits.gz")


@pytest.fixture(scope="module")
def orbit_table():
    return read_orbit(NUSTAR_ORBIT)


def test_same_spacecraft_position_as_pints_own_loader(orbit_table):
    """Our subclass puts the spacecraft where PINT's own NuSTAR loader does.

    PINT grew a NuSTAR loader of its own, so this comparison needs no patching, and it
    is the test that licenses deleting the monkey patch. The agreement is not exact, and
    cannot be: PINT indexes the orbit by absolute MJD in float64, whose step at MJD
    57263 is 0.6 us, and the two paths reach those MJDs by slightly different
    arithmetic. At 7.6 km/s that is about 5 mm of spacecraft motion -- 17 picoseconds of
    Roemer delay, six orders of magnitude below the 100 ns target.
    """
    from pint.observatory.satellite_obs import get_satellite_observatory

    from barycenter.pintengine import TableSatelliteObs

    theirs = get_satellite_observatory("nustar_pint", NUSTAR_ORBIT, overwrite=True)
    ours = TableSatelliteObs("nustar_ours", orbit_table, overwrite=True)

    # Sample away from the very ends, where the two may clip differently.
    mjd = np.linspace(orbit_table["MJD_TT"][2], orbit_table["MJD_TT"][-3], 50)
    t = Time(mjd, format="mjd", scale="tt")

    pos_diff = np.abs((ours.get_gcrs(t) - theirs.get_gcrs(t)).to_value("m"))
    assert np.max(pos_diff) < 0.01, f"max {np.max(pos_diff):.3e} m"

    ours_pv, theirs_pv = ours.posvel_gcrs(t), theirs.posvel_gcrs(t)
    assert np.max(np.abs(ours_pv.pos.to_value("m") - theirs_pv.pos.to_value("m"))) < 0.01
    # Velocity differs by the acceleration (about 8 m/s^2) times the same 0.6 us.
    assert np.max(np.abs(ours_pv.vel.to_value("m/s") - theirs_pv.vel.to_value("m/s"))) < 1e-4


def test_extrapolating_past_the_orbit_file_warns_instead_of_raising(orbit_table, caplog):
    """A sub-second shortfall must not abort the run; a big gap must be reported loudly.

    Orbit files routinely stop a fraction of a second before the last event. PINT
    raises ``ValueError`` there, which is useless: refusing to process a file over a
    sub-second extrapolation of a smooth orbit helps nobody. A gap five times the
    allowance is different -- the file is missing a chunk -- so it is logged as an error.
    """
    from barycenter.pintengine import TableSatelliteObs

    obs = TableSatelliteObs("nustar_bounds", orbit_table, maxextrap=2, overwrite=True)
    last = orbit_table["MJD_TT"][-1]

    obs._check_bounds(Time(last + 1.0 / 86400, format="mjd", scale="tt"))
    assert not [r for r in caplog.records if r.levelname in ("WARNING", "ERROR")]

    with caplog.at_level("WARNING"):
        obs._check_bounds(Time(last + 3.0 / 1440, format="mjd", scale="tt"))
    assert any(r.levelname == "WARNING" for r in caplog.records)

    caplog.clear()
    with caplog.at_level("WARNING"):
        obs._check_bounds(Time(last + 30.0 / 1440, format="mjd", scale="tt"))
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_met_range_clips_the_toa_grid(orbit_table):
    """Only the span asked for is computed, not the whole orbit file.

    Without this a multi-day orbit file, or a stack of them, costs tens of thousands of
    PINT TOAs that are then never used.
    """
    import astropy.units as u
    from pint.models import StandardTimingModel

    from barycenter.pintengine import pint_barycentric_correction

    model = StandardTimingModel
    model.RAJ.quantity = 294.9107 * u.deg
    model.DECJ.quantity = 21.58308 * u.deg
    model.DM.quantity = 0.0
    model.EPHEM.value = "DE440"

    met = np.asarray(orbit_table["MET"].value, dtype=np.float64)
    start = met.min() + 100.0
    fun = pint_barycentric_correction(orbit_table, model, dt=5.0, met_range=(start, start + 20.0))
    grid = fun.x
    assert grid.min() >= met.min()
    assert grid.max() - grid.min() < 40.0
    # And the correction it returns is a light-travel time, a few hundred seconds.
    assert 100.0 < abs(fun(start + 10.0)) < 600.0
