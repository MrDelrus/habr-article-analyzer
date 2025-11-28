import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from habr_article_analyzer.data_loader import HabrDataset
from habr_article_analyzer.settings import data_settings, settings
from habr_article_analyzer.utils import save_jsonl_zst

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


if __name__ == "__main__":
    main()
