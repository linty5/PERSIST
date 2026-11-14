"""Verified GitHub downloads for the default model fallback.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

import hashlib
import os
import tempfile
from http.client import HTTPException
from pathlib import Path
from typing import Dict, Tuple
from urllib.request import Request, urlopen


DEFAULT_RELEASE = "v1.1.0"
RELEASE_REPO = "linty5/PERSIST"
RELEASE_BASE_URL = "https://github.com/linty5/PERSIST/releases/download/v1.1.0"
DOWNLOAD_FILES = {
    "config.json": {
        "bytes": 3739,
        "sha256": "9c0b649b13a4422afcd548a329233527f46c6f4ae9daaf901881aa85f3dba6e0",
    },
    "model.safetensors": {
        "bytes": 52829520,
        "sha256": "2842fb2af7628765fcbc4cdf32a0712580714b18b10550ab7a09df988be10287",
    },
}


def offline_mode(local_files_only: bool = False) -> bool:
    """Also honour HF_HUB_OFFLINE when it was set after the Hub client was imported."""
    value = os.environ.get("HF_HUB_OFFLINE", "").strip().lower()
    return bool(local_files_only or value not in ("", "0", "false", "no", "off"))


def _verified(path: Path, expected: Dict) -> bool:
    try:
        if path.stat().st_size != expected["bytes"]:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest() == expected["sha256"]
    except OSError:
        return False


def _fetch(path: Path, expected: Dict) -> None:
    """Publish only a verified complete download; preserve any valid concurrent result."""
    temporary = None
    try:
        fd, name = tempfile.mkstemp(prefix="." + path.name + "-", suffix=".part", dir=str(path.parent))
        temporary = Path(name)
        os.close(fd)
        request = Request(RELEASE_BASE_URL + "/" + path.name,
                          headers={"User-Agent": "persist_sbd/1.1.0"})
        digest, size = hashlib.sha256(), 0
        with urlopen(request, timeout=30) as response, temporary.open("wb") as stream:
            for block in iter(lambda: response.read(1 << 20), b""):
                size += len(block)
                if size > expected["bytes"]:
                    raise RuntimeError("GitHub model file exceeds its expected size: " + path.name)
                stream.write(block)
                digest.update(block)
        if size != expected["bytes"]:
            raise RuntimeError("GitHub model download is incomplete: " + path.name)
        if digest.hexdigest() != expected["sha256"]:
            raise RuntimeError("GitHub model SHA-256 verification failed: " + path.name)
        if not _verified(path, expected):
            os.replace(str(temporary), str(path))
    except (OSError, HTTPException) as exc:
        raise RuntimeError("could not download " + path.name + " from GitHub Release "
                           + DEFAULT_RELEASE + ": " + str(exc)) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # keep the original download error


def release_files(cache_dir: str, local_files_only: bool = False) -> Tuple[str, str]:
    """Resolve verified config and weights in a cache separate from Hub snapshots."""
    cache = Path(cache_dir) / "persist_sbd" / "github" / DEFAULT_RELEASE
    missing = [(cache / name, expected) for name, expected in DOWNLOAD_FILES.items()
               if not _verified(cache / name, expected)]
    if missing and offline_mode(local_files_only):
        names = ", ".join(path.name for path, _ in missing)
        raise RuntimeError("the official model is missing or corrupt in the local cache (" + names
                           + "); offline mode prevents a GitHub download. Run persist_sbd --download "
                           "once with network access, or pass a local folder with --model")
    if missing:
        try:
            cache.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError("cannot create the model cache: " + str(exc)) from exc
        for path, expected in missing:
            if not _verified(path, expected):  # another process may have completed this file
                _fetch(path, expected)
    return str(cache / "config.json"), str(cache / "model.safetensors")
