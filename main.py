import os
from typing import Dict, Any, List

import httpx
from dotenv import load_dotenv
from anthropic import Anthropic

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

app = FastAPI()

# --- CORS (permitir Vercel do teu frontend) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    # qualquer domínio que comece por onesynth-frontend e termine em .vercel.app
    allow_origin_regex=r"^https://onesynth-frontend.*\.vercel\.app$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Prompt (não alterar o ficheiro, apenas ler) ---
with open("prompt_v1.txt", "r", encoding="utf-8") as f:
    PROMPT_BASE = f.read()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)


class ApifyAnalyzeRequest(BaseModel):
    hotel_name: str


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def root():
    return {"message": "Resumo Honesto de Reviews API - MVP (até 20 reviews)"}


async def fetch_reviews_google_places(hotel_name: str) -> Dict[str, Any]:
    if not GOOGLE_PLACES_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_PLACES_API_KEY em falta no servidor")

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

        results: List[Dict[str, Any]] = data.get("results") or []
        if not results:
            raise HTTPException(status_code=404, detail="Hotel não encontrado (Google Places)")

        place_id = results[0].get("place_id")
        if not place_id:
            raise HTTPException(status_code=404, detail="Hotel sem place_id (Google Places)")

        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        details_params = {
            "place_id": place_id,
            "fields": "reviews,rating,user_ratings_total",
            "key": GOOGLE_PLACES_API_KEY,
        }

        r = await client.get(details_url, params=details_params)
        r.raise_for_status()
        details = r.json()

        result = details.get("result") or {}
        reviews = result.get("reviews") or []

        if not reviews:
            raise HTTPException(status_code=404, detail="Sem reviews (Google Places)")

        return {
            "reviews": reviews,
            "reviews_with_text_count": len([rv for rv in reviews if (rv.get("text") or "").strip()]),
            "rating": result.get("rating"),
            "reviews_count": result.get("user_ratings_total"),
        }


def analyze_with_claude(reviews_text: str) -> str:
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY em falta no servidor")

    content = PROMPT_BASE + "\n\nREVIEWS:\n" + reviews_text

    msg = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=2000,
        messages=[{"role": "user", "content": content}],
    )
    return msg.content[0].text


@app.post("/analyze-apify")
async def analyze_apify(request: ApifyAnalyzeRequest):
    hotel = (request.hotel_name or "").strip()
    if not hotel:
        raise HTTPException(status_code=400, detail="Nome do hotel vazio")

    google_data = await fetch_reviews_google_places(hotel)

    reviews_text = ""
    for i, rv in enumerate(google_data["reviews"], 1):
        reviews_text += (
            f"Review {i} ({rv.get('rating', 0)}/5 - {rv.get('author_name','Anónimo')}):\n"
            f"{rv.get('text','')}\n\n"
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
