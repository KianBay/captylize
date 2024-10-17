from fastapi import FastAPI
import uvicorn

from captylize.app.routers import analyses

app = FastAPI(root_path="/api/v1")
app.include_router(analyses.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
