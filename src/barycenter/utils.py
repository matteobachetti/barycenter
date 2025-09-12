from astropy.io import fits
import logging as logger


__all__ = ["fits_open_including_remote", "fits_open_remote"]


def fits_open_remote(filename, **kwargs):
    """Open a remote FITS file.

    This function attempts to open a FITS file using `astropy.io.fits.open`. If a
    `PermissionError` is raised and the filename appears to be a remote URL,
    it retries opening the file with fsspec.

    Requires the `botocore` package to be installed.

    Parameters
    ----------
    filename : str
        The path or URL to the FITS file to open. Can be a local file path or a remote URL.
    **kwargs
        Additional keyword arguments passed to `astropy.io.fits.open`.

    Returns
    -------
    hdulist : astropy.io.fits.HDUList
        The opened FITS file as an HDUList object.

    Raises
    ------
    PermissionError
        If the file cannot be opened and anonymous access is not possible or fails.

    """
    import botocore
    import botocore.exceptions

    try:
        # This will work for local files and remote files with proper permissions
        hdul = fits.open(filename, **kwargs)
    except (PermissionError, botocore.exceptions.NoCredentialsError) as e:
        if "://" in filename:
            logger.info(f"Permission denied for {filename}, trying with fsspec.")
            hdul = fits.open(filename, use_fsspec=True, fsspec_kwargs={"anon": True}, **kwargs)

    # print(hdul[1].data["TIME"])
    return hdul


def fits_open_including_remote(filename, **kwargs):
    """Open a FITS file, including remote files with anonymous access if needed.

    If the filename appears to be a remote URL, it calls `fits_open_remote` to handle
    potential permission issues. Otherwise, it opens the file directly with
    `astropy.io.fits.open`.

    Parameters
    ----------
    filename : str
        The path or URL to the FITS file to open. Can be a local file path or a remote URL.
    **kwargs
        Additional keyword arguments passed to `astropy.io.fits.open`.

    Returns
    -------
    hdulist : astropy.io.fits.HDUList
        The opened FITS file as an HDUList object.

    """

    if "://" in filename:
        return fits_open_remote(filename, **kwargs)
    hdul = fits.open(filename, **kwargs)
    print(filename, hdul[1].data["TIME"][0], hdul[1].data["TIME"][-1])
    return hdul


def slim_down_hdu_list(hdul, additional_cols=None, ext=1):
    """Reduce a FITS HDUList size by only keeping few columns in the specified extension.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        Input HDUList.

    Other Parameters
    ----------------

    additional_cols : list of str, optional
        Additional column names to keep in the output file, in addition to the "TIME" column
    ext: int or str or List
        Extension(s) to slim down. Default is 1.
    """

    data = hdul[1].data
    cols = [data.columns["TIME"]]
    for col in additional_cols or []:
        if col in data.columns.names:
            cols.append(data.columns[col])
    if isinstance(ext, (int, str)):
        ext = [ext]

    for e in ext:
        hdu = hdul[e]
        if hdu.data is None:
            continue
        if "TIME" not in hdu.data.names:
            raise ValueError(f"Extension {e} does not contain a TIME column.")
        logger.info(f"Slimming down extension {e} to columns {[c.name for c in cols]}")
        hdul[e].data = fits.BinTableHDU.from_columns(cols).data

    return hdul


def slim_down_file(file, outfile, additional_cols=None, ext=1):
    """Reduce a FITS file size only keeping few columns in the specified extension.

    Parameters
    ----------
    file : str
        Input FITS file path
    outfile : str
        Output FITS file path.

    Other Parameters
    ----------------
    additional_cols : list of str, optional
        Additional column names to keep in the output file, in addition to the "TIME" column
    ext: int or str or List
        Extension(s) to slim down. Default is 1.
    """
    hdul = slim_down_hdu_list(
        fits_open_including_remote(file), additional_cols=additional_cols, ext=ext
    )

    hdul.writeto(outfile)
