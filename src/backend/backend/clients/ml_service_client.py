from collections.abc import AsyncGenerator

import httpx

from backend import settings
from core.schemas.api import ForwardRequest, ForwardResponse, ModelListResponse

ML_ENDPOINTS: dict[str, str] = {
    "models": f"{settings.ML_SERVICE_URL}/v0/models",
    "forward": f"{settings.ML_SERVICE_URL}/v0/forward",
}


class MLServiceClient:
    _x_internal_key: str

    def __init__(self, x_internal_key: str, endpoints: dict[str, str]) -> None:
        self._x_internal_key = x_internal_key
        self._endpoints = endpoints
        self._client = httpx.AsyncClient(timeout=10)

    async def forward(self, request: ForwardRequest) -> ForwardResponse:
        endpoint = self._endpoints["forward"]
        raw_response = await self._client.post(
            endpoint,
            headers={"x-internal-key": self._x_internal_key},
            json=request.model_dump(),
        )

        if raw_response.status_code != 200:
            raise RuntimeError("ML service error")

        response = ForwardResponse.model_validate(raw_response.json())
        if response.result:
            response.result.sort(key=lambda x: x.score, reverse=True)

        return response

    async def models(self) -> ModelListResponse:
        endpoint = self._endpoints["models"]
        raw_response = await self._client.get(
            endpoint, headers={"x-internal-key": self._x_internal_key}
        )

        return ModelListResponse.model_validate(raw_response.json())

    async def close(self) -> None:
        await self._client.aclose()


async def get_ml_service_client() -> AsyncGenerator[MLServiceClient, None]:
    ml_service_client = MLServiceClient(
        x_internal_key=settings.INTERNAL_API_KEY,
        endpoints=ML_ENDPOINTS,
    )
    try:
        yield ml_service_client
    finally:
        await ml_service_client.close()
