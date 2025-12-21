import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ml_training.data.habr_dataset import HabrDataset
from ml_training.ml_training.settings import data_settings, settings
from ml_training.ml_training.utils import save_jsonl_zst

logger = logging.getLogger("generate_negatives")


def main(input_path: Optional[Path] = None, output_path: Optional[Path] = None) -> None:
    logger.info("Starting to generate negatives")

    input_path = input_path or (settings.raw_data_dir / "train.jsonl.zst")
    output_path = output_path or (
        settings.raw_data_dir / "train_with_negatives.jsonl.zst"
    )

    columns = ["id", "text", "hubs"]
    logger.info("Loading dataset")
    dataset = HabrDataset(
        path=input_path, columns=columns, batch_size=data_settings.batch_size
    )

    all_rows: List[pd.DataFrame] = []
    hub_counts: Dict[str, int] = {}
    for batch_df in dataset:
        for hubs in batch_df["hubs"]:
            for hub in hubs:
                hub_counts[hub] = hub_counts.get(hub, 0) + 1
        all_rows.append(batch_df)

    df = pd.concat(all_rows, ignore_index=True)
    top_hubs = [
        hub
        for hub, _ in sorted(hub_counts.items(), key=lambda x: -x[1])[
            : data_settings.top_hubs_count
        ]
    ]

    new_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating negatives"):
        text = row["text"]
        positive_hubs = row["hubs"]
        num_pos = min(len(positive_hubs), data_settings.max_positives)
        sampled_positives = list(
            np.random.choice(positive_hubs, num_pos, replace=False)
        )

        negative_candidates = [h for h in top_hubs if h not in positive_hubs]
        num_neg = min(len(negative_candidates), data_settings.num_negatives)
        sampled_negatives = list(
            np.random.choice(negative_candidates, num_neg, replace=False)
        )

        for hub in sampled_positives:
            new_rows.append({"text": text, "hub": hub, "label": 1})
        for hub in sampled_negatives:
            new_rows.append({"text": text, "hub": hub, "label": 0})

    new_df = pd.DataFrame(new_rows)
    save_jsonl_zst(new_df, output_path)
    logger.info("Saved the result: %s", output_path)


if __name__ == "__main__":
    main()
