"""Download model files into the container on startup."""
import logging
import shutil
import urllib.request
from pathlib import Path

from .config import settings

logger = logging.getLogger(__name__)


def download_if_missing(url: str, dest_path: str, timeout: int = 300) -> Path:
    """Download ``url`` to ``dest_path`` if it does not exist or is empty.

    Returns the resolved destination path.
    """
    dest = Path(dest_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Model file already present, skipping download: %s", dest)
        return dest

    logger.info("Downloading model from %s -> %s", url, dest)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        if tmp_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded file is empty: {url}")
        tmp_path.replace(dest)
        logger.info("Downloaded model (%d bytes): %s", dest.stat().st_size, dest)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        logger.error("Failed to download model from %s: %s", url, e)
        raise
    return dest


def ensure_models_downloaded() -> None:
    """Download all configured models on startup.

    Raises if a configured model cannot be downloaded, which fails the
    application startup with a clear message.
    """
    downloads = settings.get_model_downloads()
    if not downloads:
        logger.info("No model download URLs configured; skipping download step")
        return

    for item in downloads:
        url = item.get("url")
        rel_file = item.get("file", "")
        if not url:
            continue
        if not rel_file:
            raise ValueError(
                f"Model download URL configured for '{item.get('name')}' "
                "but no model_file is set in its metadata"
            )
        # model_file already includes the models/ prefix and is resolved the
        # same way get_detection_model_path()/get_recognition_model_path() do.
        dest = str(Path(rel_file).resolve())
        download_if_missing(url, dest, timeout=settings.model_download_timeout)
