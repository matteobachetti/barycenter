"""Getting a remote file onto the local disk without breaking anything else.

These tests serve the "remote" file from a throwaway HTTP server bound to the loopback
interface, so they need no network and are not marked ``remote_data``.
"""

import http.server
import threading

import pytest
from astropy.config import set_temp_cache
from astropy.utils.data import check_download_cache, get_cached_urls

from barycenter.remote import download_locally

PAYLOAD = b"SIMPLE  =                    T / a file pretending to be FITS\n"


@pytest.fixture
def cache_dir(tmp_path):
    """A throwaway astropy cache, so a test can never touch the user's real one."""
    path = tmp_path / "cache"
    path.mkdir()
    return path


@pytest.fixture
def served_file(tmp_path):
    """A one-file HTTP server on the loopback interface; yields the file's URL."""
    root = tmp_path / "served"
    root.mkdir()
    (root / "orbit.fits").write_bytes(PAYLOAD)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, *args):  # keep the test output quiet
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/orbit.fits"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestDownloadLocally:
    """A download must leave astropy's own download cache usable."""

    def test_the_file_arrives_intact(self, served_file, cache_dir, tmp_path):
        """The bytes on disk afterwards are the bytes that were served."""
        outdir = tmp_path / "out"
        outdir.mkdir()
        with set_temp_cache(cache_dir):
            local = download_locally(served_file, outdir=str(outdir))
        assert open(local, "rb").read() == PAYLOAD

    def test_astropys_cache_is_left_undamaged(self, served_file, cache_dir, tmp_path):
        """Downloading does not leave the cache index pointing at a missing file.

        ``download_locally`` used to fetch with ``cache=True`` and then ``shutil.move``
        the file out of the cache directory, so the index still claimed to hold a URL
        whose file had walked away. ``check_download_cache`` is astropy's own
        consistency check and raises when that has happened.
        """
        outdir = tmp_path / "out"
        outdir.mkdir()
        with set_temp_cache(cache_dir):
            download_locally(served_file, outdir=str(outdir))
            check_download_cache()
            assert served_file not in get_cached_urls()

    def test_an_existing_local_copy_is_not_downloaded_again(self, served_file, cache_dir, tmp_path):
        """The local copy is the cache here, so a second call must not refetch."""
        outdir = tmp_path / "out"
        outdir.mkdir()
        (outdir / "orbit.fits").write_bytes(b"already here")
        with set_temp_cache(cache_dir):
            local = download_locally(served_file, outdir=str(outdir))
        assert open(local, "rb").read() == b"already here"
