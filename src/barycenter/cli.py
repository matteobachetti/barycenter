"""The command line interface."""

import os

from .core import apply_barycenter_correction
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
            "Clock correction file. If not provided, the latest clock file will be used for NuSTAR."
            " Specify 'none' to skip clock correction."
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
        "--only-columns",
        type=str,
        default=None,
        help="Only keep these additional columns in the output file, "
        "in addition to the TIME column. It is a comma separated list, like PI,PRIOR",
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
    )


if __name__ == "__main__":
    main_barycenter()
