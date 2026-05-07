import os
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from typing import TypedDict, Optional
from datetime import datetime

class EFIRState(TypedDict):
    tourist: dict
    location_text: Optional[str]
    efir_text: Optional[str]

def get_model():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

def format_location(state: EFIRState) -> EFIRState:
    place = state["tourist"].get("place", "Unknown location")
    return {**state, "location_text": f"Last known destination: {place}"}

def generate_efir(state: EFIRState) -> EFIRState:
    model = get_model()
    tourist = state["tourist"]
    now = datetime.now().strftime("%d %B %Y, %I:%M %p")

    prompt = f"""Generate a formal missing person First Information Report (FIR)
in standard Indian police format.

FIR Date/Time: {now}
Tourist Name: {tourist['name']}
Nationality: {tourist['nationality']}
ID Document: {tourist['idNumber']}
Phone: {tourist['phone']}
Hotel: {tourist['hotelName']}
Destination: {tourist['place']}
Check-in: {tourist['checkIn']}
Check-out: {tourist['checkOut']}
Emergency Contact: {tourist['emergencyName']} ({tourist['emergencyPhone']})
Tourist ID: {tourist['touristId']}
Location: {state['location_text']}
SOS raised at: {now}

Generate a complete formal FIR document including:
1. FIR Header (date, time, station: Tourism Safety Cell)
2. Complainant details (TourSafe System)
3. Missing person details
4. Last known location and circumstances
5. Action requested
6. Declaration

Format professionally as an official document."""

    response = model.invoke(prompt)
    return {**state, "efir_text": response.content}

def build_efir_graph():
    graph = StateGraph(EFIRState)
    graph.add_node("format_location", format_location)
    graph.add_node("generate_efir", generate_efir)
    graph.set_entry_point("format_location")
    graph.add_edge("format_location", "generate_efir")
    graph.add_edge("generate_efir", END)
    return graph.compile()

async def run_efir_agent(tourist: dict) -> str:
    graph = build_efir_graph()
    result = graph.invoke({
        "tourist": tourist,
        "location_text": None,
        "efir_text": None
    })
    return result["efir_text"]