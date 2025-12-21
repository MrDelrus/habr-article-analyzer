import logging
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_training.data.habr_dataset import HabrDataset
from ml_training.data.loaders import DEFAULT_FULL_PATH
from ml_training.ml_training.settings import data_settings, settings
from ml_training.ml_training.utils import save_jsonl_zst

logger = logging.getLogger("split_dataset")


def main() -> None:
    logger.info("Starting to split dataset")

    raw_path = settings.raw_data_dir / "habr.jsonl.zst"
    train_path = settings.raw_data_dir / "train.jsonl.zst"
    test_path = settings.raw_data_dir / "test.jsonl.zst"

    columns = ["id", "text_markdown", "hubs"]
    dataset = HabrDataset(
        path=raw_path, columns=columns, batch_size=data_settings.batch_size
    )

    all_rows = []
    for batch_df in dataset:
        batch_df = batch_df.rename(columns={"text_markdown": "text"})
        all_rows.append(batch_df)

    df = pd.concat(all_rows, ignore_index=True)

    logger.info("Splitting the dataset into train/test")
    train_df, test_df = train_test_split(
        df,
        test_size=data_settings.test_size,
        random_state=data_settings.random_seed,
        shuffle=True,
    )

    logger.info("Saving train dataset to %s", train_path)
    save_jsonl_zst(train_df, train_path)

    logger.info("Saving test dataset to %s", test_path)
    save_jsonl_zst(test_df, test_path)

    logger.info("Done")


def split_dataset(
    input_path: Path = DEFAULT_FULL_PATH,
    train_path: Path = settings.raw_data_dir / "train.jsonl.zst",
    val_path: Path = settings.raw_data_dir / "val.jsonl.zst",
    test_path: Path = settings.raw_data_dir / "test.jsonl.zst",
    val_size: float = data_settings.val_size,
    test_size: float = data_settings.test_size,
    columns: List[str] = ["id", "text_markdown", "hubs"],
    reader_batch_size: int = data_settings.batch_size,
    random_state: int = data_settings.random_seed,
) -> None:
    df = HabrDataset(
        path=input_path, columns=columns, batch_size=reader_batch_size
    ).get_dataframe()

    train_df, test_val_df = train_test_split(
        df, test_size=test_size + val_size, random_state=random_state, shuffle=True
    )
    val_df, test_df = train_test_split(
        test_val_df,
        test_size=test_size / (test_size + val_size),
        random_state=random_state,
        shuffle=True,
    )
    for dataset, path in [
        (train_df, train_path),
        (val_df, val_path),
        (test_df, test_path),
    ]:
        logger.info("Saving to {}".format(path))
        save_jsonl_zst(dataset, path)


if __name__ == "__main__":
    main()
