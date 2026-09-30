"""Can numpy close the gap on numba if the expression is written to avoid temporaries?

The naive translation allocates about a dozen intermediate arrays. Horner form with
in-place operations needs four.
"""

import timeit

import numpy as np
from numba import vectorize


def naive(x, xtab0, xtab1, ytab0, ytab1, yptab0, yptab1):
    dx = x - xtab0
    xs = xtab1 - xtab0
    ys = ytab1 - ytab0
    dx = dx / xs
    y0 = ytab0
    yp0 = yptab0 * xs
    yp1 = yptab1 * xs
    a = y0
    b = yp0
    c = 3 * ys - 2 * yp0 - yp1
    d = yp0 + yp1 - 2 * ys
    return a + dx * (b + dx * (c + dx * d))


def horner(x, xtab0, xtab1, ytab0, ytab1, yptab0, yptab1):
    xs = np.subtract(xtab1, xtab0)
    dx = np.subtract(x, xtab0)
    dx = np.divide(dx, xs, out=np.asarray(dx, dtype=np.float64))

    ys = np.subtract(ytab1, ytab0)
    yp0 = np.multiply(yptab0, xs)
    yp1 = np.multiply(yptab1, xs)

    # d = yp0 + yp1 - 2*ys, built in place into a fresh buffer
    out = np.add(yp0, yp1)
    acc = np.subtract(out, 2.0 * ys)
    # c = 3*ys - 2*yp0 - yp1  ==  ys - d - yp0 + ... keep it explicit instead
    c = 3.0 * ys
    c -= 2.0 * yp0
    c -= yp1
    acc *= dx
    acc += c
    acc *= dx
    acc += yp0
    acc *= dx
    acc += ytab0
    return acc


nb = vectorize("float64(float64, float64, float64, float64, float64, float64, float64)")(naive)


def args(n):
    rng = np.random.default_rng(42)
    x = np.sort(rng.random(n)) * 100
    xtab0 = np.floor(x)
    return (x, xtab0, xtab0 + 1.0, rng.random(n), rng.random(n), rng.random(n), rng.random(n))


print(f"{'N':>10} {'numba':>11} {'naive np':>11} {'horner np':>11} {'horner/numba':>13}  max|err|")
for n in (1_000, 10_000, 100_000, 1_000_000, 10_000_000):
    a = args(n)
    nb(*a)
    reps = max(3, min(500, int(1e7 / n)))
    t_nb = min(timeit.repeat(lambda: nb(*a), number=reps, repeat=5)) / reps
    t_na = min(timeit.repeat(lambda: naive(*a), number=reps, repeat=5)) / reps
    t_ho = min(timeit.repeat(lambda: horner(*a), number=reps, repeat=5)) / reps
    err = np.max(np.abs(horner(*a) - np.asarray(nb(*a))))
    print(
        f"{n:>10} {t_nb * 1e3:>9.3f}ms {t_na * 1e3:>9.3f}ms {t_ho * 1e3:>9.3f}ms "
        f"{t_ho / t_nb:>13.2f}  {err:.2e}"
    )
