import os
from pymongo import MongoClient
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from typing import TypedDict, Optional

class RiskState(TypedDict):
    incidents: Optional[dict]
    tourist_counts: Optional[dict]
    report: Optional[str]

def get_model():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

def get_db():
    client = MongoClient(os.getenv("MONGO_URI"))
    return client["toursafe"]

def fetch_data(state: RiskState) -> RiskState:
    db = get_db()
    tourists = list(db.tourists.find({}))

    alert_tourists = [t for t in tourists if t.get("status") == "alert"]
    active_tourists = [t for t in tourists if t.get("status") == "active"]
    departed_tourists = [t for t in tourists if t.get("status") == "departed"]

    destination_counts = {}
    destination_alerts = {}

    for t in tourists:
        place = t.get("place", "Unknown").split(",")[0].strip()
        destination_counts[place] = destination_counts.get(place, 0) + 1

    for t in alert_tourists:
        place = t.get("place", "Unknown").split(",")[0].strip()
        destination_alerts[place] = destination_alerts.get(place, 0) + 1

    return {**state, "incidents": {
        "total_tourists": len(tourists),
        "active": len(active_tourists),
        "alerts": len(alert_tourists),
        "departed": len(departed_tourists),
        "by_destination": destination_counts,
        "alerts_by_destination": destination_alerts,
    }}

def generate_report(state: RiskState) -> RiskState:
    model = get_model()
    data = state["incidents"]

    prompt = f"""You are a safety analyst for tourism authorities in Northeast India.
Generate a weekly risk report based on this data:

Total tourists: {data['total_tourists']}
Active: {data['active']}
SOS alerts: {data['alerts']}
Departed: {data['departed']}
By destination: {data['by_destination']}
Alerts by destination: {data['alerts_by_destination']}

Write a concise professional report with:
1. Overall risk level (Low/Medium/High)
2. Key findings (3 points)
3. Destinations needing attention
4. Recommendations for police (2 actionable items)"""

    response = model.invoke(prompt)
    return {**state, "report": response.content}

def build_risk_graph():
    graph = StateGraph(RiskState)
    graph.add_node("fetch_data", fetch_data)
    graph.add_node("generate_report", generate_report)
    graph.set_entry_point("fetch_data")
    graph.add_edge("fetch_data", "generate_report")
    graph.add_edge("generate_report", END)
    return graph.compile()

async def run_risk_agent() -> str:
    graph = build_risk_graph()
    result = graph.invoke({
        "incidents": None,
        "tourist_counts": None,
        "report": None
    })
    return result["report"]