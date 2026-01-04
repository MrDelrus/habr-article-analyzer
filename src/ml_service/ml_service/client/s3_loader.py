import os

import boto3

from ml_service import settings

_s3_client = boto3.client(
    "s3",
    region_name=settings.AWS_REGION,
)


def get_local_dir(dir_name: str = "models_cache") -> str:
    local_dir = os.path.join("models_cache")
    os.makedirs(local_dir, exist_ok=True)
    return local_dir


def list_models() -> set[str]:
    local_dir = get_local_dir()
    local_models = set(os.listdir(local_dir))

    remote_models: set[str] = set()
    try:
        paginator = _s3_client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=settings.S3_BUCKET_NAME):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(settings.MODEL_EXTENSION):
                    remote_models.add(key)

    except Exception:
        ...

    return local_models.union(remote_models)


def download_model(model_key: str) -> str:
    local_dir = get_local_dir()
    local_path = os.path.join(local_dir, os.path.basename(model_key))

    if not os.path.exists(local_path):
        _s3_client.download_file(
            Bucket=settings.S3_BUCKET_NAME, Key=model_key, Filename=local_path
        )
    return local_path
