from fastapi import FastAPI
from app.routes.model_routes import router as model_router

app = FastAPI(
    title="Mistral Text Generation API",
    description="API for generating text using the Mistral model.",
    version="1.0.0"
)

# Include the model routes
app.include_router(model_router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Mistral Text Generation API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)