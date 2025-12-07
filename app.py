from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Traveller Agent", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class TravelInput(BaseModel):
    destination: str = Field(..., description="City or location to visit")
    area: Optional[str] = Field(None, description="Preferred area within destination")
    duration_days: int = Field(..., gt=0, description="Number of days")
    budget: int = Field(..., gt=0, description="Total budget")
    travel_type: str = Field(..., description="Type of travel (budget, luxury, etc.)")
    interests: List[str] = Field(default_factory=list, description="List of interests")


class TravelPlan(BaseModel):
    travel_summary: str
    hotel_options: str
    places_to_visit: str
    cost_estimate: str
    full_plan: str


def get_llm():
    """Initialize LLM with Google Gemini API"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )
    except ImportError:
        return None


def extract_content(response):
    """Extract text content from LLM response"""
    if isinstance(response, str):
        return response
    if hasattr(response, 'content'):
        content = response.content
        if isinstance(content, list):
            return "".join(str(chunk.get("text", "")) if isinstance(chunk, dict) else str(chunk) for chunk in content)
        return str(content)
    return str(response)


def plan_trip(travel_input: TravelInput) -> TravelPlan:
    """Generate travel plan using LangGraph workflow"""
    llm = get_llm()
    if not llm:
        return TravelPlan(
            travel_summary="Please set GOOGLE_API_KEY in .env file to use the travel planner.",
            hotel_options="",
            places_to_visit="",
            cost_estimate="",
            full_plan="Please configure your API key to generate travel plans."
        )
    
    state = {"travel_input": travel_input.dict()}
    
    # Step 1: Analyze travel profile
    summary_prompt = f"""You are a smart travel agent. Analyze the following user preferences:
- Destination: {travel_input.destination} ({travel_input.area or 'general area'})
- Duration: {travel_input.duration_days} days
- Budget: ₹{travel_input.budget} total
- Travel type: {travel_input.travel_type}
- Interests: {', '.join(travel_input.interests) if travel_input.interests else 'General exploration'}

Give a concise summary and note any constraints or considerations."""
    state["travel_summary"] = extract_content(llm.invoke(summary_prompt))
    
    # Step 2: Suggest hotels
    area_text = f"{travel_input.area}, {travel_input.destination}" if travel_input.area else travel_input.destination
    hotel_prompt = f"""Recommend 2 {travel_input.travel_type}-friendly accommodation options in or near {area_text}
for {travel_input.duration_days} days stay under ₹{travel_input.budget} total.

For each option include:
- Name
- Approximate price per night
- Total cost
- Why it's a good choice"""
    state["hotel_options"] = extract_content(llm.invoke(hotel_prompt))
    
    # Step 3: Suggest places
    places_prompt = f"""Suggest 5 must-visit places in or near {travel_input.destination} focusing on:
- Interests: {', '.join(travel_input.interests) if travel_input.interests else 'General exploration'}
- Budget constraints (travel type: {travel_input.travel_type})
- Located within 10-15 km of {travel_input.area or travel_input.destination}

Include place name and reason to visit."""
    state["places_to_visit"] = extract_content(llm.invoke(places_prompt))
    
    # Step 4: Cost estimation
    cost_prompt = f"""Estimate the total travel cost for this {travel_input.duration_days}-day trip to {travel_input.destination} with the following:
- Hotel: {state['hotel_options']}
- Sightseeing: {state['places_to_visit']}

Break down cost for:
- Hotel stay
- Transport/local travel
- Food
- Entry tickets (if any)
- Misc

Give total cost and whether it's within ₹{travel_input.budget}."""
    state["cost_estimate"] = extract_content(llm.invoke(cost_prompt))
    
    # Step 5: Final summary
    full_plan = f"""Travel Summary:
{state['travel_summary']}

Hotel Options:
{state['hotel_options']}

Places to Visit:
{state['places_to_visit']}

Cost Estimate:
{state['cost_estimate']}"""
    
    return TravelPlan(
        travel_summary=state["travel_summary"],
        hotel_options=state["hotel_options"],
        places_to_visit=state["places_to_visit"],
        cost_estimate=state["cost_estimate"],
        full_plan=full_plan
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/plan", response_model=TravelPlan)
async def create_plan(travel_input: TravelInput):
    try:
        return plan_trip(travel_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

