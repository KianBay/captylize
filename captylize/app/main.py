from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

from captylize.app.routers import analyses
from captylize.app.http_client import async_session
from captylize.ml.manager import ModelManager, get_device

app = FastAPI(root_path="/api/v1")
app.include_router(analyses.router)


DEVICE = get_device()
model_manager = ModelManager(cache_dir="./model_cache", device=DEVICE)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Before yield is on startup
    model_manager.load_models()
    yield
    # After yield is on shutdown
    model_manager.unload_models()
    await async_session.close()


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
