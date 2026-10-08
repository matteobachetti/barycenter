"""Good time intervals that come in a file of their own.

Most missions put the good time intervals (GTIs) in an extension of the event file, and
:mod:`barycenter.core` corrects them with everything else. Some do not. SVOM/ECLAIRs
ships them in a separate file with one extension per criterion -- stable attitude,
South Atlantic Anomaly, telemetry, and no, partial or total Earth occultation of the field
of view -- and leaves it to the user to decide which criteria apply and to combine them.

This module does that combining, and writes the result into a copy of the event file as
an ordinary ``GTI`` extension, so that barycentring then moves the GTIs together with the
events. It knows nothing about any mission: extensions are named by the caller.

The command line is ``barycenter-apply-gti``; without ``-e`` it lists the GTI extensions
a file offers::

    barycenter-apply-gti events.fits gtis.fits
    barycenter-apply-gti events.fits gtis.fits -e GTICAL-STA,GTICAL-NSA,GTICAL-NEO --filter-events
"""

import logging as logger
import os

import numpy as np
from astropy.io import fits

from .utils import column_named, high_precision_keyword_read, splitext_improved

__all__ = [
    "add_gti_extension",
    "gti_extensions",
    "intersect_gtis",
    "main_apply_gti",
    "read_gtis",
    "union_gtis",
]

#: Keywords copied from the event extension into the new GTI extension, so that its
#: START/STOP are on the time scale the events are on, and say so.
TIMING_KEYWORDS = (
    "TELESCOP",
    "INSTRUME",
    "TIMESYS",
    "TIMEREF",
    "TIMEUNIT",
    "MJDREF",
    "MJDREFI",
    "MJDREFF",
    "TIMEZERO",
)

#: How far two reference epochs may differ and still count as the same, in seconds.
#: SVOM writes MJDREFF as 0.000800740741 in the events and 0.000800740740999999 in
#: the GTI file: the same epoch to 1e-13 s, which an exact comparison would refuse.
EPOCH_TOLERANCE_S = 1e-6


def _as_intervals(gtis):
    gtis = np.asarray(gtis, dtype=np.float64).reshape(-1, 2)
    return gtis[np.argsort(gtis[:, 0], kind="stable")]


def union_gtis(gti_lists):
    """Time that is good in *any* of the lists, as sorted, non-overlapping intervals.

    Intervals that overlap or touch are merged.

    Parameters
    ----------
    gti_lists : list of array-like of shape (N, 2)

    Returns
    -------
    numpy.ndarray of shape (M, 2)
    """
    if not len(gti_lists):
        return np.zeros((0, 2))
    gtis = _as_intervals(np.concatenate([_as_intervals(g) for g in gti_lists]))
    merged = []
    for start, stop in gtis:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return np.array(merged, dtype=np.float64).reshape(-1, 2)


def intersect_gtis(gti_lists):
    """Time that is good in *every* one of the lists.

    Each list is first merged with itself, so overlapping intervals inside one list do
    not count twice. The result can be empty, with shape ``(0, 2)``.

    Parameters
    ----------
    gti_lists : list of array-like of shape (N, 2)

    Returns
    -------
    numpy.ndarray of shape (M, 2)
    """
    if not len(gti_lists):
        return np.zeros((0, 2))
    result = union_gtis([gti_lists[0]])
    for other in gti_lists[1:]:
        other = union_gtis([other])
        pieces = []
        for start, stop in result:
            lo = np.maximum(start, other[:, 0])
            hi = np.minimum(stop, other[:, 1])
            keep = hi > lo
            pieces.extend(zip(lo[keep], hi[keep]))
        result = np.array(pieces, dtype=np.float64).reshape(-1, 2)
    return result


def _is_gti(hdu):
    data = getattr(hdu, "data", None)
    if not isinstance(hdu, fits.BinTableHDU | fits.TableHDU) or data is None:
        return False
    return (
        column_named(data, "START") is not None
        and column_named(data, "STOP") is not None
        and column_named(data, "TIME") is None
    )


def gti_extensions(hdul):
    """The extensions of an open file that hold GTIs: ``START`` and ``STOP``, no ``TIME``."""
    return [hdu for hdu in hdul[1:] if _is_gti(hdu)]


def _intervals_of(hdu):
    data = hdu.data
    start, stop = column_named(data, "START"), column_named(data, "STOP")
    return np.column_stack([data[start], data[stop]]).astype(np.float64)


def _pick(hdul, spec):
    """The extension named by ``spec``, or by the first present of ``A|B|C``."""
    available = {hdu.name.upper(): hdu for hdu in gti_extensions(hdul)}
    for name in spec.split("|"):
        if name.strip().upper() in available:
            return available[name.strip().upper()]
    raise KeyError(
        f"None of {spec!r} is a GTI extension of this file; it has "
        f"{', '.join(available) or 'none'}."
    )


def read_gtis(gti_file, extensions):
    """Read the named GTI extensions of a file.

    Parameters
    ----------
    gti_file : str
    extensions : list of str
        Extension names, in any case. An entry ``"A|B|C"`` means the first of these the
        file has, for one criterion that different files name differently
        (``"GTI|STDGTI"``). Do not use it to fall back between different criteria.

    Returns
    -------
    list of numpy.ndarray of shape (N, 2)
        One per entry of ``extensions``, in the same order.

    Raises
    ------
    KeyError
        If an entry matches no GTI extension. A constraint that is silently dropped is
        worse than one that stops the run.
    """
    with fits.open(gti_file) as hdul:
        return [_intervals_of(_pick(hdul, spec)) for spec in extensions]


def _epoch_s(header):
    """``MJDREF`` in seconds plus ``TIMEZERO``, or ``None`` if there is no ``MJDREF``."""
    mjdref = high_precision_keyword_read(header, "MJDREF")
    if mjdref is None:
        return None
    return mjdref * 86400 + header.get("TIMEZERO", 0.0)


def _event_hdu(hdul):
    for hdu in hdul[1:]:
        if column_named(getattr(hdu, "data", None), "TIME") is not None:
            return hdu
    raise ValueError("No extension with a TIME column: this is not an event file.")


def _check_same_epoch(event_header, gti_file, extensions):
    ours = _epoch_s(event_header)
    with fits.open(gti_file) as hdul:
        for spec in extensions:
            hdu = _pick(hdul, spec)
            theirs = _epoch_s(hdu.header)
            if theirs is None:
                theirs = _epoch_s(hdul[0].header)
            if ours is None or theirs is None:
                logger.warning(
                    f"Cannot check that {hdu.name} counts time from the events' MJDREF: "
                    "one of the two has no MJDREF."
                )
                continue
            if abs(float(ours - theirs)) > EPOCH_TOLERANCE_S:
                raise ValueError(
                    f"{hdu.name} counts time from a different MJDREF/TIMEZERO than the "
                    f"events ({float(theirs - ours):.6f} s apart): its intervals would be "
                    "misplaced by that much."
                )


def add_gti_extension(
    event_file,
    gti_file,
    extensions,
    outfile=None,
    combine="intersect",
    filter_events=False,
    overwrite=False,
):
    """Write a copy of an event file with the chosen GTIs merged into a ``GTI`` extension.

    Parameters
    ----------
    event_file : str
    gti_file : str
        The file holding the GTI extensions. May be the event file itself.
    extensions : list of str
        Which extensions to use; see :func:`read_gtis`.
    outfile : str, optional
        Default: ``gti_<event file name>`` next to the event file.
    combine : ``"intersect"`` or ``"union"``
        ``"intersect"`` (the default) keeps time that passes every criterion, which is
        what a set of independent quality criteria means. ``"union"`` keeps time good in
        any of them, as for per-CCD GTIs.
    filter_events : bool
        Also drop events outside the merged GTIs. By default they are kept, and only the
        ``GTI`` extension records which are good.
    overwrite : bool

    Returns
    -------
    str
        The output file name.

    Raises
    ------
    ValueError
        If the event file already has a ``GTI`` extension, or if a GTI extension counts
        time from a different reference epoch than the events.
    """
    if combine not in ("intersect", "union"):
        raise ValueError(f"combine must be 'intersect' or 'union', not {combine!r}")
    if outfile is None:
        path, fname = os.path.split(event_file)
        root, extension = splitext_improved(fname)
        outfile = os.path.join(path, "gti_" + root + extension)

    gtis = read_gtis(gti_file, extensions)
    merged = intersect_gtis(gtis) if combine == "intersect" else union_gtis(gtis)

    with fits.open(event_file) as hdul:
        if any(hdu.name.upper() == "GTI" for hdu in hdul):
            raise ValueError(
                f"{event_file} already has a GTI extension; refusing to add a second one."
            )
        events = _event_hdu(hdul)
        _check_same_epoch(events.header, gti_file, extensions)

        if filter_events:
            times = events.data[column_named(events.data, "TIME")]
            keep = np.zeros(len(times), dtype=bool)
            for start, stop in merged:
                keep |= (times >= start) & (times <= stop)
            logger.info(f"Keeping {keep.sum()} of {keep.size} events inside the GTIs")
            events.data = events.data[keep]

        gti = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="START", format="D", unit="s", array=merged[:, 0]),
                fits.Column(name="STOP", format="D", unit="s", array=merged[:, 1]),
            ],
            name="GTI",
        )
        for key in TIMING_KEYWORDS:
            if key in events.header:
                gti.header[key] = events.header[key]
        gti.header["HDUCLASS"] = "OGIP"
        gti.header["HDUCLAS1"] = "GTI"
        gti.header["HDUCLAS2"] = "STANDARD"
        gti.header["ONTIME"] = (float(np.sum(merged[:, 1] - merged[:, 0])), "[s] Sum of GTIs")
        gti.header.add_history(
            f"{combine} of {', '.join(extensions)} from {os.path.basename(gti_file)}"
        )
        hdul.append(gti)
        hdul.writeto(outfile, overwrite=overwrite)
    return outfile


def main_apply_gti(args=None):
    """Command line: ``barycenter-apply-gti EVENTS GTIFILE [-e EXT1,EXT2|EXT3]``."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Merge good time intervals from separate extensions (possibly in a separate "
            "file) into one GTI extension of an event file, before barycentring. "
            "Without -e, list the GTI extensions of GTIFILE."
        )
    )
    parser.add_argument("file", help="Event file")
    parser.add_argument("gtifile", help="File with the GTI extensions (may be the event file)")
    parser.add_argument(
        "-e",
        "--extensions",
        default=None,
        help=(
            "Comma-separated GTI extension names to combine. 'A|B' means the first of A "
            "and B present in the file (quote it: '|' is a pipe to the shell)."
        ),
    )
    parser.add_argument(
        "--union",
        action="store_true",
        default=False,
        help="Keep time good in any extension, instead of in all of them",
    )
    parser.add_argument(
        "--filter-events",
        action="store_true",
        default=False,
        help=(
            "Also drop the events outside the merged GTIs. Needed by tools that read "
            "only the event list and ignore the GTI extension"
        ),
    )
    parser.add_argument("-o", "--outfile", default=None, help="Default: gti_<file>")
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args(args)

    if args.extensions is None:
        with fits.open(args.gtifile) as hdul:
            for hdu in gti_extensions(hdul):
                intervals = _intervals_of(hdu)
                exposure = np.sum(intervals[:, 1] - intervals[:, 0])
                print(f"{hdu.name:20s} {len(intervals):6d} intervals {exposure:12.1f} s")
        return None

    return add_gti_extension(
        args.file,
        args.gtifile,
        [e.strip() for e in args.extensions.split(",")],
        outfile=args.outfile,
        combine="union" if args.union else "intersect",
        filter_events=args.filter_events,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main_apply_gti()
