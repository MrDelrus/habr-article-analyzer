from pydantic import BaseModel


class StatsResponse(BaseModel):
    total_queries: int
    text_length_mean: float
    latency_mean: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
