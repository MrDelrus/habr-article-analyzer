from pathlib import Path

import jsonlines
import pandas as pd
import pytest
import zstandard as zstd

from ml_training.data.habr_dataset import HabrDataset


@pytest.fixture
def small_dataset_file(tmp_path: Path) -> Path:
    data = [
        {
            "id": 1,
            "language": "ru",
            "title": "Title 1",
            "author": "Author 1",
            "text_markdown": "Some long text",
            "text_html": "<p>Some long text</p>",
            "comments": [],
            "statistics": {"readingCount": 100},
        },
        {
            "id": 2,
            "language": "en",
            "title": "Title 2",
            "author": "Author 2",
            "text_markdown": "Another long text",
            "text_html": "<p>Another long text</p>",
            "comments": [],
            "statistics": {"readingCount": 200},
        },
    ]

    temp_jsonl = tmp_path / "temp.jsonl"
    with jsonlines.open(temp_jsonl, "w") as writer:
        for row in data:
            writer.write(row)

    file_path = tmp_path / "test_dataset.jsonl.zst"
    with open(temp_jsonl, "rb") as src, open(file_path, "wb") as dst:
        cctx = zstd.ZstdCompressor()
        dst.write(cctx.compress(src.read()))

    return file_path


def test_habr_dataset_iteration(small_dataset_file: Path) -> None:
    dataset = HabrDataset(path=small_dataset_file, batch_size=1)
    batches = []
    for batch in dataset:
        batches.append(batch)
        # Check default behaviour
        assert isinstance(batch, pd.DataFrame)
        assert "text_markdown" in batch.columns
        assert "text_html" not in batch.columns
        assert "comments" not in batch.columns
        assert "id" in batch.columns
        assert "author" in batch.columns
    assert len(batches) == 2


def test_habr_dataset_columns_selection(small_dataset_file: Path) -> None:
    columns = ["id", "author", "statistics"]
    dataset = HabrDataset(path=small_dataset_file, columns=columns, batch_size=2)
    batch = next(iter(dataset))
    assert list(batch.columns) == columns
    assert batch.shape == (2, 3)
