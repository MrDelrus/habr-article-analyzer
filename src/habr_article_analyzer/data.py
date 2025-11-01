import io
import logging
import os
from pathlib import Path

import jsonlines
import pandas as pd
import requests
import zstandard as zstd
from tqdm import tqdm

from habr_article_analyzer.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_FULL_PATH = settings.data_dir / "raw/habr.jsonl.zst"
DEFAULT_TINY_PATH = settings.data_dir / "tiny/habr.csv"
URL_FULL = "https://huggingface.co/datasets/IlyaGusev/habr/resolve/main/habr.jsonl.zst"


def download_dataset(local_path: Path = DEFAULT_FULL_PATH, url: str = URL_FULL) -> None:
    """Download full dataset from Hugging Face if it does not exist."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path):
        logger.info(f"Dataset already exists at {local_path}, skipping download.")
        return

    logger.info(f"Downloading dataset from {url} to {local_path}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        chunk_size = 8192

        with (
            open(local_path, "wb") as f,
            tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading"
            ) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))
    logger.info("Download completed!")


def load_dataset_from_zst(
    local_path: Path = DEFAULT_FULL_PATH, rows_num: int = None
) -> pd.DataFrame:
    """Load dataset from disk into a pandas DataFrame."""
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"{local_path} not found. Run download_dataset() first."
        )

    logger.info(f"Loading dataset from {local_path}")
    data = []
    with open(local_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
            for obj in tqdm(jsonlines.Reader(text_reader), desc="Reading records"):
                data.append(obj)
                if rows_num is not None and len(data) >= rows_num:
                    break

    logger.info(f"Loaded {len(data)} records")
    return pd.DataFrame(data)


def load_tiny_dataset(local_path: Path = DEFAULT_TINY_PATH) -> pd.DataFrame:
    """Load a small dataset for tests / CI."""
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"{local_path} not found. You can create a tiny CSV manually."
        )
    logger.info(f"Loading tiny dataset from {local_path}")
    df = pd.read_csv(local_path)
    logger.info(f"Loaded {len(df)} records from tiny dataset")
    return df
