"""Is the numba @vectorize on _cubic_interpolation worth a hard dependency?

Compares the compiled version against the identical expression in plain numpy, on
array sizes spanning what a real clock table interpolation sees, plus the scalar
case where numba normally wins.
"""

import timeit

import numpy as np
from numba import vectorize


def _body(x, xtab0, xtab1, ytab0, ytab1, yptab0, yptab1):
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


numba_fun = vectorize("float64(float64, float64, float64, float64, float64, float64, float64)")(
    _body
)
numpy_fun = _body


def args(n):
    rng = np.random.default_rng(42)
    if n == 1:
        return tuple(float(v) for v in rng.random(7))
    x = np.sort(rng.random(n)) * 100
    xtab0 = np.floor(x)
    xtab1 = xtab0 + 1.0
    return (x, xtab0, xtab1, rng.random(n), rng.random(n), rng.random(n), rng.random(n))


print(f"numpy {np.__version__}")
print(f"{'N':>10} {'numba':>12} {'numpy':>12} {'numpy/numba':>12}  max|diff|")
for n in (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000):
    a = args(n)
    numba_fun(*a)  # warm the JIT so compilation is not timed
    reps = max(3, min(2000, int(2e7 / max(n, 1))))
    t_nb = min(timeit.repeat(lambda: numba_fun(*a), number=reps, repeat=5)) / reps
    t_np = min(timeit.repeat(lambda: numpy_fun(*a), number=reps, repeat=5)) / reps
    diff = np.max(np.abs(np.asarray(numba_fun(*a)) - np.asarray(numpy_fun(*a))))
    print(f"{n:>10} {t_nb * 1e6:>10.3f}us {t_np * 1e6:>10.3f}us {t_np / t_nb:>12.2f}  {diff:.3e}")
