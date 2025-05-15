from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from api.routes import router
import os
from dotenv import load_dotenv

load_dotenv()  # Already there

app = FastAPI(title="Loubby AI Navigator - RAG Implementation")
app.include_router(router)

@app.get("/")
async def root():
    try:
        with open("static/index.html") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Static index.html file not found.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))