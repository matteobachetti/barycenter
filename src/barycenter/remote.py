"""Getting the input files onto the local disk.

Event and orbit files may be given as local paths, ``https://`` URLs or ``s3://`` URLs
(HEASARC publishes its archive both ways). On SciServer the archive is already
mounted, so nothing is downloaded there; see
:func:`barycenter.core.apply_barycenter_correction`.
"""

import contextlib
import logging as logger
import os
import shutil
from collections.abc import Iterable

__all__ = ["download_locally"]


@contextlib.contextmanager
def _do_in_other_directory(x):
    if x == "":
        x = "."
    d = os.getcwd()

    # This could raise an exception, but it's probably
    # best to let it propagate and let the caller
    # deal with it, since they requested x
    os.chdir(x)

    try:
        yield

    finally:
        # This could also raise an exception, but you *really*
        # aren't equipped to figure out what went wrong if the
        # old working directory can't be restored.
        os.chdir(d)


def download_locally(fname, outdir="."):
    """Download a remote file locally if needed.
    Manages S3 and HTTP(s) URLs. For S3, only public buckets are supported at the moment

    A local path is never copied: it comes back as an absolute path to the file where it
    is. A copy only ever duplicated the input, went stale when the observation was
    reprocessed (the stale copy was then read without warning), and collided between runs
    sharing a working directory. Nothing is written next to a local input either, so a
    read-only input directory is fine.

    Parameters
    ----------
    fname : str
        Input file path or URL.
    outdir : str
        Where downloaded files go. A relative local path is still read from the caller's
        current directory, not from here.

    Returns
    -------
    local_fname : str
        Local file path.
    """

    if not isinstance(fname, str) and isinstance(fname, Iterable):
        return [download_locally(f, outdir=outdir) for f in fname]

    if not fname.startswith(("http://", "https://", "s3://")):
        local_fname = os.path.abspath(fname)
        if not os.path.exists(local_fname):
            raise FileNotFoundError(f"No such file: {local_fname}")
        return local_fname

    with _do_in_other_directory(outdir):
        if fname.startswith("http://") or fname.startswith("https://"):
            from astropy.utils.data import download_file

            local_fname = os.path.basename(fname)
            if os.path.exists(local_fname):
                logger.info(f"{local_fname} already exists, skipping download.")
            else:
                # cache=False, deliberately. The local copy made just below is already
                # the cache -- the branch above skips the download when it is there --
                # and with cache=True the file astropy hands back lives *inside* its
                # download cache, so moving it away leaves the cache index claiming a
                # URL whose contents have walked off. astropy's own
                # ``check_download_cache`` reports that as CacheDamaged, and it breaks
                # every later download_file call in the same environment, ours or
                # anyone else's. With cache=False the file is a temporary one that is
                # ours to move.
                cache_file = download_file(fname, cache=False)
                shutil.move(cache_file, local_fname)
                logger.info(f"Downloaded remote file {fname} to local file {local_fname}")
        else:
            from urllib.parse import urlparse

            import boto3
            import botocore

            # Parse S3 URL
            parsed = urlparse(fname)
            bucket_name = parsed.netloc
            config = botocore.client.Config(signature_version=botocore.UNSIGNED)
            s3_resource = boto3.resource("s3", config=config)
            s3_client = s3_resource.meta.client
            path = fname.replace(f"s3://{bucket_name}/", "")
            # Deliberately a prefix match rather than an exact key: HEASARC stores its
            # event files gzipped, and observation logs and papers quote the uncompressed
            # name, so asking for ``..._cl.evt`` has to find ``..._cl.evt.gz``. The local
            # name below is taken from the key that was actually found, so the suffix that
            # arrives is the suffix on disk. The cost is that a prefix matching several
            # keys silently takes the first; give the full key when that matters.
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=path)
            objects = response.get("Contents", [])
            if len(objects) == 0:
                raise FileNotFoundError(f"No objects found at S3 path {fname}")
            key = objects[0]["Key"]
            path2 = "/".join(path.strip("/").split("/")[:-1])
            dest = key[len(path2) + 1 :]
            if os.path.exists(dest):
                logger.info(f"{dest} already exists, skipping download.")
            else:
                s3_client.download_file(bucket_name, key, dest)
            logger.info(f"Downloaded remote file {fname} to local file {dest}")
            local_fname = dest

        fname = os.path.abspath(local_fname)

    return fname
