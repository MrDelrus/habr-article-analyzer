import io
from pathlib import Path

import jsonlines
import pandas as pd
import zstandard as zstd


def save_jsonl_zst(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=3)
    with open(path, "wb") as f:
        with cctx.stream_writer(f) as compressor:
            writer = jsonlines.Writer(io.TextIOWrapper(compressor, encoding="utf-8"))
            writer.write_all(df.to_dict(orient="records"))
            writer.close()
