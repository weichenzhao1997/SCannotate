from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 1. Allow React (Vite) to talk to Python during development
# CORS: browser security rule - relax it locally so the frontend can talk to the backend
    # during development, React runs on one origin (often http://localhost:5173)
    # FastAPI runs on another origin (often http://localhost:8000)
    # Browsers block cross-origin requests by default
    # CORS tells the browser: this backend allows requests from that frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GraphRequest(BaseModel):
    """
        request body is parsed into a GraphRequest
        if JSON contains invalid data, FastAPI returns an error
    
    """
    k: float


@app.post("/calculate")
async def calculate_graph(req: GraphRequest):
    """
    Receives a graph request. k: float
    Returns a point: 
    {
    "data": [
        {"x": 0, "y": 0},
        {"x": 1, "y": kx},
        ...
        {"x": 20, "y": kx}
        ]
    }
    """
    points = [{"x": x, "y": req.k * x} for x in range(0, 21)]
    return {"data": points}

# 2. THE HOSTING PART: Serve the 'dist' folder from Vite
app.mount("/", StaticFiles(directory="dist", html=True), name="static")