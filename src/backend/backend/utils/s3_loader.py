import os

import boto3

from backend.config import settings

_s3_client = boto3.client(
    "s3",
    region_name=settings.AWS_REGION,
)


def list_models() -> set[str]:
    paginator = _s3_client.get_paginator("list_objects_v2")

    models: set[str] = set()

    for page in paginator.paginate(Bucket=settings.S3_BUCKET_NAME):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(settings.MODEL_EXTENSION):
                models.add(key)

    return models


def download_model(model_key: str) -> str:
    local_dir = os.path.join("models_cache")
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, os.path.basename(model_key))

    if not os.path.exists(local_path):
        _s3_client.download_file(
            Bucket=settings.S3_BUCKET_NAME, Key=model_key, Filename=local_path
        )
    return local_path
