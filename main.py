from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from anthropic import Anthropic
from dotenv import load_dotenv
import httpx

app = FastAPI()

# CORS – MVP simples e estável
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://onesynth-frontend.vercel.app",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

with open("prompt_v1.txt", "r", encoding="utf-8") as f:
    PROMPT_BASE = f.read()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

class ApifyAnalyzeRequest(BaseModel):
    hotel_name: str


async def fetch_reviews_google_places(hotel_name: str) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        search_params = {
            "query": hotel_name,
            "type": "lodging",
            "key": GOOGLE_PLACES_API_KEY,
        }

        r = await client.get(search_url, params=search_params)
        r.raise_for_status()
        data = r.json()

        if not data.get("results"):
            raise HTTPException(status_code=404, detail="Hotel não encontrado")

        place_id = data["results"][0]["place_id"]

        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        details_params = {
            "place_id": place_id,
            "fields": "reviews,rating,user_ratings_total",
            "key": GOOGLE_PLACES_API_KEY,
        }

        r = await client.get(details_url, params=details_params)
        r.raise_for_status()
        details = r.json()

        result = details.get("result", {})
        reviews = result.get("reviews", [])

        if not reviews:
            raise HTTPException(status_code=404, detail="Sem reviews")

        return {
            "reviews": reviews,
            "reviews_with_text_count": len([r for r in reviews if r.get("text")]),
            "rating": result.get("rating"),
            "reviews_count": result.get("user_ratings_total"),
        }


def analyze_with_claude(reviews_text: str) -> str:
    content = PROMPT_BASE + "\n\nREVIEWS:\n" + reviews_text

    msg = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=2000,
        messages=[{"role": "user", "content": content}],
    )
    return msg.content[0].text


@app.post("/analyze-apify")
async def analyze_apify(request: ApifyAnalyzeRequest):
    if not request.hotel_name.strip():
        raise HTTPException(status_code=400, detail="Nome do hotel vazio")

    google_data = await fetch_reviews_google_places(request.hotel_name)

    reviews_text = ""
    for i, r in enumerate(google_data["reviews"], 1):
        reviews_text += (
            f"Review {i} ({r.get('rating', 0)}/5 - {r.get('author_name','Anónimo')}):\n"
            f"{r.get('text','')}\n\n"
        )

    result_markdown = analyze_with_claude(reviews_text)

    return {
        "result_markdown": result_markdown,
        "meta": {
            "received_reviews": len(google_data["reviews"]),
            "reviews_with_text": google_data["reviews_with_text_count"],
            "google_rating": google_data["rating"],
            "total_ratings": google_data["reviews_count"],
        },
    }


@app.get("/")
async def root():
    return {"message": "Resumo Honesto de Reviews API - MVP (até 20 reviews)"}
