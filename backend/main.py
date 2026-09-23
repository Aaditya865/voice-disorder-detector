import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.api import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = FastAPI(
    title="AI-Based Multi-Class Laryngeal Voice Disorder Classification System",
    description=(
        "Academic research backend for voice disorder screening using "
        "pretrained Wav2Vec 2.0 representations and Random Forest classification."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration for local development and frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include main router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
