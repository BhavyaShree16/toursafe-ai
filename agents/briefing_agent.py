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
    tourist = state["tourist"]

    district = tourist.get("district", "")
    place = tourist.get("place", "")

    # Use district for weather search
    search_location = district if district else place.split(",")[0].strip()

    api_key = os.getenv("OPENWEATHER_API_KEY")

    weather_str = None

    try:
        res = requests.get(
            "http://api.openweathermap.org/data/2.5/weather",
            params={
                "q": f"{search_location},IN",
                "appid": api_key,
                "units": "metric"
            },
            timeout=5
        )

        data = res.json()

        if res.status_code == 200:
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            humidity = data["main"]["humidity"]

            weather_str = f"{temp}°C, {desc}, humidity {humidity}%"

    except Exception:
        weather_str = None

    return {
        **state,
        "weather": weather_str
    }

def generate_briefing(state: BriefingState) -> BriefingState:
    tourist = state["tourist"]
    weather = state["weather"]

    district = tourist.get("district", "your destination")
    place = tourist.get("place", district)

    model = get_model()

    weather_line = (
        f"Current weather in {district}: {weather}"
        if weather else
        ""
    )

    prompt = f"""
You are a tourist safety assistant for Tamil Nadu, India.

Generate a friendly WhatsApp travel safety briefing.

Tourist name: {tourist['name']}
District: {district}
Places visiting: {place}
Check-in: {tourist['checkIn']}
Check-out: {tourist['checkOut']}
Purpose: {tourist.get('purpose', 'travel')}

{weather_line}

Instructions:
- Start with the tourist's name
- Mention the district naturally
- Include weather only if available
- Give 2-3 short useful safety/travel tips
- Keep it conversational and realistic
- Under 150 words
- Plain text only
- No markdown
- End exactly with:
Reply SOS anytime if you need emergency help
"""

    response = model.invoke(prompt)

    return {
        **state,
        "briefing": response.content
    }

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