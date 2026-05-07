from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from agents.briefing_agent import run_briefing_agent
from agents.risk_agent import run_risk_agent
from agents.efir_agent import run_efir_agent
from pydantic import BaseModel
from typing import Optional
import traceback
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class BriefingRequest(BaseModel):
    touristId: str
    name: str
    phone: str
    place: str
    checkIn: str
    checkOut: str
    purpose: Optional[str] = "leisure"

class EFIRRequest(BaseModel):
    touristId: str
    name: str
    nationality: str
    idNumber: str
    phone: str
    place: str
    hotelName: str
    emergencyName: str
    emergencyPhone: str
    checkIn: str
    checkOut: str

@app.get("/health")
def health():
    groq_key = os.getenv("GROQ_API_KEY")
    weather_key = os.getenv("OPENWEATHER_API_KEY")
    return {
        "status": "ok",
        "groq_key_loaded": bool(groq_key),
        "weather_key_loaded": bool(weather_key),
    }

@app.post("/agent/briefing")
async def briefing(req: BriefingRequest):
    try:
        result = await run_briefing_agent(req.dict())
        return {"briefing": result}
    except Exception as e:
        print("BRIEFING AGENT ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/risk")
async def risk():
    try:
        result = await run_risk_agent()
        return {"report": result}
    except Exception as e:
        print("RISK AGENT ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/efir")
async def efir(req: EFIRRequest):
    try:
        result = await run_efir_agent(req.dict())
        return {"efir": result}
    except Exception as e:
        print("EFIR AGENT ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))