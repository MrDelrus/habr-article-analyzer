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
