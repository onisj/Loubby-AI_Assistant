from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
from app import dynamic_memory_app

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for JS and CSS)
app.mount("/static", StaticFiles(directory="templates"), name="static")


# Serve index.html at root
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open(Path("templates/index.html")) as f:
        return HTMLResponse(content=f.read())

# Health check endpoint
@app.get("/healthz")
def health_check():
    return {"status": "healthy"}

# Your existing chat endpoint (no changes needed)
@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    question = body.get("question")
    history = body.get("history", [])
    state = {"question": question, "history": history}
    response = dynamic_memory_app.invoke(state)
    return response
