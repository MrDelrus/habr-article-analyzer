from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from backend.clients.database_client import engine
from backend.routes import auth_router, forward_router, history_router, models_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield
    await engine.dispose()


app = FastAPI(title="Backend API", lifespan=lifespan)

app.include_router(auth_router)
app.include_router(forward_router)
app.include_router(models_router)
app.include_router(history_router)
