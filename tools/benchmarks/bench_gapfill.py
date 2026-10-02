"""How well can a spacecraft position be filled in across a long gap in an orbit file?

Real data: one day of the Fermi GBM position history (1 s, inertial J2000 frame, metres).
It is thinned to 30 s, the cadence of a LAT spacecraft file, and positions only (the
pipeline does not use the LAT file's velocity either).  A gap of ``G`` minutes is cut out
of the thinned samples and each model is fitted to ``WINDOW`` seconds of samples on each
side of it.  The error is the distance to the true 1 s position at every second inside
the gap.  Dividing metres by c gives the worst-case light-travel-time error.

Models
------
spline     what ``native.py`` does today: a cubic spline through the remaining samples.
harmonic   least squares of  sum_k (a_k + b_k t) cos(k w t) + (c_k + d_k t) sin(k w t),
           k = 1..HARMONICS, with the orbital frequency ``w`` scanned for the best fit.
kepler     a 6-parameter orbit (position and velocity at the start of the window)
           integrated numerically under point-mass gravity only.
j2         the same with the Earth's flattening (J2) included.

Usage: python bench_gapfill.py [poshist.fit] [out.csv]
"""

import sys
import time

import numpy as np
from astropy.io import fits
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares, minimize_scalar

GM = 3.986004418e14
J2 = 1.08262668e-3
RE = 6378137.0
C = 299792458.0

CADENCE = 30
WINDOW = 95 * 60  # seconds of samples kept on each side of the gap
GAPS_MIN = [5, 10, 20, 40]
N_START = 25
HARMONICS = 3
MODELS = ("spline", "harmonic", "kepler", "j2")


def load(fname):
    with fits.open(fname) as hdul:
        d = hdul["GLAST POS HIST"].data
        t = np.asarray(d["SCLK_UTC"], dtype=float)
        pos = np.stack([d["POS_X"], d["POS_Y"], d["POS_Z"]], axis=1).astype(float)
    ok = np.all(np.isfinite(pos), axis=1) & (np.linalg.norm(pos, axis=1) > 6e6)
    return t[ok] - t[ok][0], pos[ok]


def spline_model(t, p):
    return CubicSpline(t, p, axis=0)


def harmonic_model(t, p, t0):
    def design(w, tt):
        x = np.asarray(tt) - t0
        cols = [np.ones_like(x), x]
        for k in range(1, HARMONICS + 1):
            c, s = np.cos(k * w * x), np.sin(k * w * x)
            cols += [c, x * c, s, x * s]
        return np.stack(cols, axis=1)

    def rss(w):
        a, *_ = np.linalg.lstsq(design(w, t), p, rcond=None)
        return np.sum((design(w, t) @ a - p) ** 2)

    w0 = 2 * np.pi / 5700.0
    ws = np.linspace(0.97 * w0, 1.03 * w0, 61)
    i = int(np.argmin([rss(w) for w in ws]))
    bounds = (ws[max(i - 1, 0)], ws[min(i + 1, len(ws) - 1)])
    w = minimize_scalar(rss, bounds=bounds, method="bounded", options={"xatol": 1e-12}).x
    a, *_ = np.linalg.lstsq(design(w, t), p, rcond=None)
    return lambda tt: design(w, tt) @ a


def _acceleration(r, with_j2):
    rn = np.linalg.norm(r)
    a = -GM * r / rn**3
    if with_j2:
        z2 = (r[2] / rn) ** 2
        f = 1.5 * J2 * GM * RE**2 / rn**5
        a = a + f * r * np.array([5 * z2 - 1, 5 * z2 - 1, 5 * z2 - 3])
    return a


def orbit_model(t, p, with_j2):
    """Fit position and velocity at ``t[0]`` and return a predictor for times > t[0]."""
    t0 = t[0]

    def propagate(state, tt):
        sol = solve_ivp(
            lambda _, y: np.concatenate([y[3:], _acceleration(y[:3], with_j2)]),
            (t0, tt[-1]),
            state,
            t_eval=tt,
            rtol=1e-10,
            atol=1e-5,
            method="DOP853",
        )
        return sol.y[:3].T

    scale = np.array([1e3] * 3 + [1.0] * 3)
    s0 = np.concatenate([p[0], (p[1] - p[0]) / (t[1] - t[0])]) / scale
    sol = least_squares(
        lambda s: ((propagate(s * scale, t) - p) / 10.0).ravel(),
        s0,
        method="lm",
        xtol=1e-13,
        ftol=1e-13,
    )
    state = sol.x * scale
    return lambda tt: propagate(state, tt)


def trial(t_all, p_all, start, gap_s):
    """Errors in metres, per model, for one gap beginning at ``start`` seconds."""
    thin = np.arange(len(t_all)) % CADENCE == 0
    t30, p30 = t_all[thin], p_all[thin]
    lo, hi = start - WINDOW, start + gap_s + WINDOW
    in_gap = (t30 > start) & (t30 < start + gap_s)
    use = (t30 >= lo) & (t30 <= hi) & ~in_gap
    t, p = t30[use], p30[use]
    truth = (t_all > start) & (t_all < start + gap_s)
    tt, pt = t_all[truth], p_all[truth]
    out = {}
    out["spline"] = np.linalg.norm(spline_model(t, p)(tt) - pt, axis=1)
    out["harmonic"] = np.linalg.norm(harmonic_model(t, p, t.mean())(tt) - pt, axis=1)
    for name, j2 in (("kepler", False), ("j2", True)):
        pred = orbit_model(t, p, j2)
        # Evaluate gap times in one integration together with the sample times.
        times = np.unique(np.concatenate([t[:1], tt]))
        pos = pred(times)
        lookup = dict(zip(times, pos))
        out[name] = np.linalg.norm(np.array([lookup[x] for x in tt]) - pt, axis=1)
    return out


def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else "glg_poshist_all_240315_v00.fit"
    csv = sys.argv[2] if len(sys.argv) > 2 else None
    t, p = load(fname)
    rng = np.random.default_rng(1)
    rows = []
    t_begin = time.time()
    for gap in GAPS_MIN:
        starts = np.sort(rng.uniform(WINDOW + 60, t[-1] - WINDOW - gap * 60 - 60, N_START))
        for s in starts:
            err = trial(t, p, s, gap * 60.0)
            for m in MODELS:
                rows.append((gap, s, m, err[m].max(), np.sqrt(np.mean(err[m] ** 2))))
        print(f"gap {gap} min done, {time.time() - t_begin:.0f} s", flush=True)

    print("\nMax error inside the gap [m]  (light-travel time = m / c, 300 m = 1 us)")
    print(f"{'gap':>5} {'model':>9} {'median':>10} {'worst':>10} {'worst [us]':>11}")
    for gap in GAPS_MIN:
        for m in MODELS:
            v = np.array([r[3] for r in rows if r[0] == gap and r[2] == m])
            print(
                f"{gap:>4}m {m:>9} {np.median(v):>10.1f} {v.max():>10.1f}"
                f" {v.max() / C * 1e6:>11.3f}"
            )
    if csv:
        np.savetxt(
            csv,
            np.array(rows, dtype=object),
            fmt="%s",
            delimiter=",",
            header="gap_min,start_s,model,max_err_m,rms_err_m",
        )


if __name__ == "__main__":
    main()
