import io
from pathlib import Path

import jsonlines
import pandas as pd
import pytest
import zstandard as zstd

from ml_training.data import loaders

# Fixtures


@pytest.fixture
def tmp_tiny_csv(tmp_path: Path) -> Path:
    """Create a temporary tiny CSV file for testing."""
    dataset_path = tmp_path / "tiny.csv"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_csv(dataset_path, index=False)
    return dataset_path


# Unit tests


def test_load_tiny_dataset_reads_csv(tmp_tiny_csv: Path) -> None:
    df = loaders.load_tiny_dataset(local_path=tmp_tiny_csv)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_load_tiny_dataset_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        loaders.load_tiny_dataset(local_path=missing)


def test_load_dataset_rows_num_limits_reading(tmp_path: Path) -> None:
    """Test that rows_num parameter correctly limits the number of rows read."""
    # Create fake compressed jsonlines file with 5 records
    data_list = [{"id": i, "value": f"row_{i}"} for i in range(5)]
    raw_str = io.StringIO()
    with jsonlines.Writer(raw_str) as writer:
        for rec in data_list:
            writer.write(rec)
    compressed = zstd.ZstdCompressor().compress(raw_str.getvalue().encode("utf-8"))
    fpath = tmp_path / "fake.jsonl.zst"
    fpath.write_bytes(compressed)

    # Test reading only 3 rows
    df = loaders.load_dataset_from_zst(local_path=fpath, rows_num=3)
    assert len(df) == 3
    assert list(df["id"]) == [0, 1, 2]

    # Test reading all rows when rows_num > actual rows
    df_all = loaders.load_dataset_from_zst(local_path=fpath, rows_num=10)
    assert len(df_all) == 5


def test_load_dataset_rows_num_none_reads_all(tmp_path: Path) -> None:
    """Test that rows_num=None reads all available rows."""
    # Create fake compressed jsonlines file with 3 records
    data_list = [{"id": i} for i in range(3)]
    raw_str = io.StringIO()
    with jsonlines.Writer(raw_str) as writer:
        for rec in data_list:
            writer.write(rec)
    compressed = zstd.ZstdCompressor().compress(raw_str.getvalue().encode("utf-8"))
    fpath = tmp_path / "fake.jsonl.zst"
    fpath.write_bytes(compressed)

    # Test with rows_num=None (default behavior)
    df_none = loaders.load_dataset_from_zst(local_path=fpath, rows_num=None)
    assert len(df_none) == 3

    # Test with rows_num=1 for comparison
    df_one = loaders.load_dataset_from_zst(local_path=fpath, rows_num=1)
    assert len(df_one) == 1


# Mocked integration tests


def test_download_dataset_skips_if_exists(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    file_path = tmp_path / "habr.jsonl.zst"
    file_path.write_text("already here")
    loaders.download_dataset(local_path=file_path)
    assert "skipping download" in caplog.text.lower()


def test_load_dataset_reads_zstd_jsonlines(tmp_path: Path) -> None:
    # Create fake compressed jsonlines file
    data_list = [{"x": 1}, {"x": 2}]
    raw_str = io.StringIO()
    with jsonlines.Writer(raw_str) as writer:
        for rec in data_list:
            writer.write(rec)
    compressed = zstd.ZstdCompressor().compress(raw_str.getvalue().encode("utf-8"))
    fpath = tmp_path / "fake.jsonl.zst"
    fpath.write_bytes(compressed)

    df = loaders.load_dataset_from_zst(local_path=fpath)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "x" in df.columns
