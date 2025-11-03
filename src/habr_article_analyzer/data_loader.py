import io
from pathlib import Path
from typing import Iterator, List, Optional, Union

import jsonlines
import pandas as pd
import zstandard as zstd
from tqdm import tqdm


class HabrDataset:
    """
    Lazy loader for Habr dataset stored in jsonl.zst format.

    Example usage:
        dataset = HabrDataset(
            path="data/raw/habr.jsonl.zst",
            columns=["id", "language", "title", "author", "statistics"],
            batch_size=50_000
        )

        for batch_df in dataset:
            print(batch_df.head())
    """

    # Default columns to exclude (largest fields)
    DEFAULT_EXCLUDE_COLUMNS = ["text_html", "comments"]

    def __init__(
        self,
        path: Union[str, Path],
        columns: Optional[List[str]] = None,
        batch_size: int = 50_000,
        exclude_columns: Optional[List[str]] = None,
    ):
        """
        Initialize the lazy dataset.

        :param path: Path to the jsonl.zst file
        :param columns: List of columns to load (None = all)
        :param batch_size: Number of rows per batch
        :param exclude_columns: List of columns to exclude
        """
        self.path = Path(path)
        self.batch_size = batch_size
        self.columns = columns
        self.exclude_columns = exclude_columns or self.DEFAULT_EXCLUDE_COLUMNS

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """
        Iterator over batches of DataFrames.
        """
        batch = []
        with open(self.path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text_reader = io.TextIOWrapper(reader, encoding="utf-8")
                for obj in tqdm(jsonlines.Reader(text_reader), desc="Reading dataset"):
                    # Remove excluded columns
                    for col in self.exclude_columns:
                        obj.pop(col, None)

                    # Keep only requested columns
                    if self.columns is not None:
                        obj = {k: v for k, v in obj.items() if k in self.columns}

                    batch.append(obj)
                    if len(batch) >= self.batch_size:
                        yield pd.DataFrame(batch)
                        batch = []

        # Yield remaining rows
        if batch:
            yield pd.DataFrame(batch)

    def __len__(self) -> int:
        """
        Length cannot be determined lazily without reading the entire file.
        Implementing this would require a full pass through the dataset.
        """
        raise NotImplementedError(
            "len(dataset) is not supported for lazy datasets. "
            "Iterate over batches instead."
        )
