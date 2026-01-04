from backend.routes.auth_routes import router as auth_router
from backend.routes.forward_routes import router as forward_router
from backend.routes.history_routes import router as history_router
from backend.routes.models_routes import router as models_router

__all__ = ["auth_router", "forward_router", "history_router", "models_router"]
