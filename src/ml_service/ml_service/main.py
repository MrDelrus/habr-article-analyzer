from fastapi import FastAPI

from ml_service.routes import forward_router, models_router

app = FastAPI(title="Internal ML Service")
app.include_router(forward_router)
app.include_router(models_router)
