# MIT License
#
# Copyright (c) 2025 Matteo Bachetti
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Barycentric time corrections for X-ray event files, in pure Python.

A photon's arrival time at the spacecraft is not a useful clock: the spacecraft is
moving, around the Earth and with the Earth around the Sun, so arrival times wander by
up to about 500 seconds over a year. Barycentring moves them to the solar system's
centre of mass, where they can be compared between missions and across years.

The workflow lives in :mod:`barycenter.core`, the command line in
:mod:`barycenter.cli`, the physics in :mod:`barycenter.native` (with
:mod:`barycenter.pintengine` as an optional second opinion), the orbit file dialects in
:mod:`barycenter.orbit`, and the spacecraft clock in :mod:`barycenter.clock`.
"""

from ._version import __version__
from .cli import main_barycenter
from .core import (
    apply_barycenter_correction,
    correct_times,
    get_barycentric_correction,
)
from .missions import MISSIONS, Mission, mission_for
from .native import barycentric_correction
from .orbit import OrbitSpec, read_orbit

__all__ = [
    "MISSIONS",
    "Mission",
    "OrbitSpec",
    "__version__",
    "apply_barycenter_correction",
    "barycentric_correction",
    "correct_times",
    "get_barycentric_correction",
    "main_barycenter",
    "mission_for",
    "read_orbit",
]
