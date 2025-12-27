from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from core.schemas.api.forward import ForwardResponse
from ml_service.routes import forward_router, models_router

app = FastAPI(title="Internal ML Service")
app.include_router(forward_router)
app.include_router(models_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    payload = ForwardResponse(result=None, error=str(exc.detail)).model_dump()
    return JSONResponse(status_code=exc.status_code, content=payload)
