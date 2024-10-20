from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

from captylize.logger import get_logger
from captylize.app.routers import analyses, generations
from captylize.app.http_client import async_session
from captylize.ml.manager import model_manager
from captylize.ml.models.registry import register_models

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up")
    # Before yield is on startup
    register_models(model_manager)
    # model_manager.load_default_models() # Uncomment to load all default models
    yield
    # After yield is on shutdown
    model_manager.unload_all_models()
    logger.info("Shutting down")
    await async_session.close()


app = FastAPI(root_path="/api/v1", lifespan=lifespan)
app.include_router(analyses.router)
app.include_router(generations.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
