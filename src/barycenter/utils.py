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
        return fits.open(filename, **kwargs)
    except (PermissionError, botocore.exceptions.NoCredentialsError) as e:
        if "://" in filename:
            logger.info(f"Permission denied for {filename}, trying with fsspec.")
            return fits.open(filename, use_fsspec=True, fsspec_kwargs={"anon": True}, **kwargs)
        raise e


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

    return fits.open(filename, **kwargs)
