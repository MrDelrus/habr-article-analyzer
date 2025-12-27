from fastapi import FastAPI

from backend.routes import forward_router, history_router, models_router

app = FastAPI(title="Backend API")

app.include_router(forward_router)
app.include_router(models_router)
app.include_router(history_router)
