"""The command line interface."""

import os

from .core import (
    AUTO_GRID_DT_S,
    AUTO_GRID_EVENTS,
    ENGINES,
    apply_barycenter_correction,
)
from .utils import splitext_improved

__all__ = ["main_barycenter"]


def _default_out_file(args):
    path, fname = os.path.split(args.file)
    root, extension = splitext_improved(fname)

    outfile = os.path.join(path, "bary_" + root)

    if args.only_columns is not None:
        outfile += "_slim"
    if args.clockfile == "none":
        outfile += "_noclk"
    if args.source_region_deg is not None:
        if args.source_region_deg >= 1.0:
            region_str = f"{args.source_region_deg:g}deg"
        else:
            region_str = f"{args.source_region_deg * 3600:g}asec".replace(".", "d")
        outfile += f"_src{region_str}"
    outfile += extension

    return outfile


def main_barycenter(args=None):
    import argparse

    description = "Apply the barycenter correction to X-ray event files"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("file", help="Uncorrected event file")
    parser.add_argument("orbitfile", help="Orbit file", nargs="+")
    parser.add_argument(
        "-p",
        "--parfile",
        help="Parameter file in TEMPO/TEMPO2/PINT format (for precise coordinates)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--ra", help="Right ascension (deg) if no parfile", default=None, type=float
    )
    parser.add_argument("--dec", help="Declination (deg) if no parfile", default=None, type=float)
    parser.add_argument(
        "--source-region-deg", help="Source region radius (deg)", default=None, type=float
    )
    parser.add_argument(
        "--radecsys",
        help="Coordinate system (default ICRS for DE4XX, FK5 for DE200)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--ephem", help="Solar system ephemeris (default DE440)", default="DE440", type=str
    )

    parser.add_argument(
        "-o", "--outfile", default=None, help="Output file name (default bary_<opts>.evt)"
    )
    parser.add_argument(
        "-c",
        "--clockfile",
        default=None,
        help=(
            "Clock correction file. If not provided, the latest one is fetched from the "
            "CALDB for the missions that publish it there (NuSTAR and Swift); RXTE always "
            "uses the tdc.dat shipped with the package. Specify 'none' to skip the clock "
            "correction."
        ),
    )
    parser.add_argument(
        "--overwrite", help="Overwrite existing data", action="store_true", default=False
    )
    parser.add_argument(
        "--apply-official",
        help="Use mission-specific official barycenter correction (e.g., heasoft barycorr)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--engine",
        default="native",
        choices=ENGINES,
        help=(
            "Which implementation computes the correction. 'native' (the default) uses "
            "astropy, ERFA and a JPL ephemeris directly; 'pint' goes through PINT's "
            "timing model, as an independent cross-check."
        ),
    )
    parser.add_argument(
        "--only-columns",
        type=str,
        default=None,
        help="Only keep these additional columns in the output file, "
        "in addition to the TIME column. It is a comma separated list, like PI,PRIOR",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help=(
            f"Interpolate the correction on a grid of this spacing, in seconds, instead "
            f"of evaluating it at every event. The default decides from the file's size: "
            f"above {AUTO_GRID_EVENTS} events a {AUTO_GRID_DT_S} s grid is used, which is "
            f"about 50 times faster and worth about 1.6 ns. Pass 0 to always evaluate the "
            f"correction at every event."
        ),
    )

    args = parser.parse_args(args)

    outfile = args.outfile
    if outfile is None:
        outfile = _default_out_file(args)

    if args.radecsys is None:
        if args.ephem == "DE200":
            args.radecsys = "FK5"
        else:
            args.radecsys = "ICRS"

    orbitfiles = args.orbitfile
    if len(orbitfiles) == 1:
        orbitfiles = orbitfiles[0]

    return apply_barycenter_correction(
        args.file,
        orbitfiles,
        parfile=args.parfile,
        outfile=outfile,
        overwrite=args.overwrite,
        clockfile=args.clockfile,
        ephem=args.ephem,
        radecsys=args.radecsys,
        ra=args.ra,
        dec=args.dec,
        source_region_deg=args.source_region_deg,
        only_columns=args.only_columns.split(",") if args.only_columns else None,
        apply_official=args.apply_official,
        engine=args.engine,
        dt=args.dt,
    )


if __name__ == "__main__":
    main_barycenter()
