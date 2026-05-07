import os
import requests
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from typing import TypedDict, Optional

class BriefingState(TypedDict):
    tourist: dict
    weather: Optional[str]
    briefing: Optional[str]

def get_model():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

def fetch_weather(state: BriefingState) -> BriefingState:
    place = state["tourist"].get("place", "Northeast India")
    first_place = place.split(",")[0].strip()
    api_key = os.getenv("OPENWEATHER_API_KEY")
    try:
        res = requests.get(
            "http://api.openweathermap.org/data/2.5/weather",
            params={"q": f"{first_place},IN", "appid": api_key, "units": "metric"},
            timeout=5
        )
        data = res.json()
        if res.status_code == 200:
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            humidity = data["main"]["humidity"]
            weather_str = f"{temp}°C, {desc}, humidity {humidity}%"
        else:
            weather_str = "Weather data unavailable"
    except Exception:
        weather_str = "Weather data unavailable"
    return {**state, "weather": weather_str}

def generate_briefing(state: BriefingState) -> BriefingState:
    tourist = state["tourist"]
    weather = state["weather"]
    model = get_model()

    prompt = f"""You are a tourist safety assistant for Northeast India.
Generate a friendly WhatsApp safety briefing message.

Tourist: {tourist['name']}
Destination: {tourist['place']}
Check-in: {tourist['checkIn']}
Check-out: {tourist['checkOut']}
Purpose: {tourist['purpose']}
Current weather: {weather}

Instructions:
- Keep it under 200 words
- Warm friendly tone
- Include weather info
- Give 2-3 safety tips for their destination
- End with: Reply SOS anytime if you need emergency help
- No markdown, plain text only
- Start with their name"""

    response = model.invoke(prompt)
    return {**state, "briefing": response.content}

def build_briefing_graph():
    graph = StateGraph(BriefingState)
    graph.add_node("fetch_weather", fetch_weather)
    graph.add_node("generate_briefing", generate_briefing)
    graph.set_entry_point("fetch_weather")
    graph.add_edge("fetch_weather", "generate_briefing")
    graph.add_edge("generate_briefing", END)
    return graph.compile()

async def run_briefing_agent(tourist: dict) -> str:
    graph = build_briefing_graph()
    result = graph.invoke({
        "tourist": tourist,
        "weather": None,
        "briefing": None
    })
    return result["briefing"]