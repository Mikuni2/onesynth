from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import os
from anthropic import Anthropic
from dotenv import load_dotenv
import httpx

app = FastAPI()

# ✅ CORS corrigido para:
# - domínio "production" (onesynth-frontend.vercel.app)
# - previews do Vercel (onesynth-frontend-...vercel.app)
# - opcional: localhost (se testares local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://onesynth-frontend.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"^https://onesynth-frontend-.*\.vercel\.app$",
    allow_credentials=False,   # ✅ não usas cookies/sessão no fetch
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
print("ANTHROPIC_API_KEY loaded:", bool(os.getenv("ANTHROPIC_API_KEY")))

# carregar prompt do ficheiro
with open("prompt_v1.txt", "r", encoding="utf-8") as f:
    PROMPT_BASE = f.read()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# ⚠️ SYSTEM_PROMPT - NÃO ALTERAR EM NENHUMA CIRCUNSTÂNCIA
SYSTEM_PROMPT = """És um analista imparcial de opiniões de utilizadores.

Objetivo:
Transformar múltiplas reviews (fornecidas manualmente) num resumo curto, claro e honesto, que ajude uma pessoa a decidir rapidamente.

Contexto:
- As reviews foram coladas manualmente pelo utilizador.
- Podem estar incompletas ou misturadas.
- Trabalha apenas com a informação fornecida.

Princípios obrigatórios:
- NÃO inventes factos.
- NÃO assumas coisas que não aparecem nas reviews.
- Dá prioridade a padrões recorrentes, não a casos isolados.
- Se houver pouca informação ou contradições fortes, diz isso claramente.
- Usa linguagem simples, direta e humana (sem marketing, sem tom "IA").
- Evita palavras vagas ("bom", "agradável", "ok").
  Prefere observações concretas suportadas pelas reviews
  (ex.: "silencioso à noite", "pequeno-almoço elogiado pela variedade", "check-in lento em várias menções").

FORMATO DE SAÍDA (OBRIGATÓRIO) — Markdown:

### Conclusão
Produz uma conclusão mais assumida, sem ser recomendação explícita.
Usa a seguinte lógica:
"Para quem valoriza X e tolera Y, este hotel tende a ser uma escolha segura."
X e Y só podem ser padrões recorrentes claramente suportados pelas reviews.

### Pontos fortes recorrentes
- Apenas padrões recorrentes e concretos.
- Não incluir elogios vagos.

### Pontos fracos recorrentes
- Apenas padrões recorrentes e concretos.
- Não dramatizar.

### Divergências / depende de
Indica claramente onde as opiniões divergem e de que fatores isso depende
(ex.: tipo de quarto, expectativas, localização, ruído).
Só incluir se houver base clara nas reviews.

### Ideal para
Torna esta secção concreta e específica.
Evita generalidades como "ideal para casais".
Prefere descrições do tipo:
"Ideal para quem valoriza silêncio e estadias curtas",
"Ideal para quem quer proximidade a X e aceita Y".

### Evitar se
Torna esta secção concreta e específica.
Ex.: "Evitar se és sensível a ruído",
"Evitar se esperas luxo de resort".
Tudo deve estar claramente suportado pelas reviews.

### Red flags
Só incluir red flags quando:
- aparecem de forma recorrente, ou
- são graves mesmo que poucas (mas nesse caso indica que são menções isoladas).
Nunca exagerar.

### Confiança do resumo
Alta / Média / Baixa — com justificação em 1 linha, baseada em:
- quantidade de informação percebida
- consistência dos padrões
- presença de contradições

Exemplos:
"Alta — padrões consistentes em múltiplos temas."
"Média — pontos fortes claros, mas com divergências relevantes."
"Baixa — poucas reviews e opiniões contraditórias."

IMPORTANTE: Quando o número de reviews for muito reduzido (menos de 10), a confiança deve ser no máximo "Média" ou "Baixa", independentemente da consistência.

### Nota de limitações (linha fixa obrigatória)
"Este resumo baseia-se apenas nas reviews fornecidas, pode não refletir experiências raras."

Regra final:
Se algo não estiver claramente suportado pelas reviews fornecidas,
não incluas no resumo."""


class ApifyAnalyzeRequest(BaseModel):
    hotel_name: str


async def fetch_reviews_google_places(hotel_name: str) -> dict:
    """Busca reviews do Google Places API (versão antiga - mais estável)"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        search_params = {
            "query": hotel_name,
            "type": "lodging",
            "key": GOOGLE_PLACES_API_KEY
        }

        try:
            search_response = await client.get(search_url, params=search_params)
            search_response.raise_for_status()
            search_data = search_response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao pesquisar hotel: {str(e)}")

        if not search_data.get("results") or len(search_data["results"]) == 0:
            raise HTTPException(status_code=404, detail="Hotel não encontrado")

        place_id = search_data["results"][0]["place_id"]

        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        details_params = {
            "place_id": place_id,
            "fields": "name,formatted_address,reviews,rating,user_ratings_total",
            "key": GOOGLE_PLACES_API_KEY
        }

        try:
            details_response = await client.get(details_url, params=details_params)
            details_response.raise_for_status()
            details_data = details_response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao obter detalhes: {str(e)}")

        if details_data.get("status") != "OK":
            raise HTTPException(status_code=500, detail=f"Erro da API: {details_data.get('status')}")

        result = details_data.get("result", {})
        reviews_list = result.get("reviews", [])

        if not reviews_list:
            raise HTTPException(status_code=404, detail="Nenhuma review encontrada para este hotel")

        reviews_with_text_count = 0
        for r in reviews_list:
            text = r.get("text", "")
            if text and text.strip():
                reviews_with_text_count += 1

        return {
            "reviews": reviews_list,
            "reviews_with_text_count": reviews_with_text_count,
            "rating": result.get("rating"),
            "reviews_count": result.get("user_ratings_total")
        }


def analyze_with_claude(reviews_text: str) -> str:
    content = PROMPT_BASE + "\n\nREVIEWS:\n" + reviews_text

    message = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=2000,
        messages=[
            {"role": "user", "content": content}
        ],
    )
    return message.content[0].text


@app.post("/analyze-apify")
async def analyze_apify(request: ApifyAnalyzeRequest):
    ...
    """Análise com Apify - até 20 reviews"""
    if not request.hotel_name.strip():
        raise HTTPException(status_code=400, detail="Nome do hotel não pode estar vazio")

    google_data = await fetch_reviews_google_places(request.hotel_name)

    reviews_list = google_data["reviews"]
    reviews_with_text_count = google_data["reviews_with_text_count"]
    rating = google_data["rating"]
    total_reviews = google_data["reviews_count"]

    reviews_text = ""
    for i, review in enumerate(reviews_list, 1):
        author = review.get("author_name", "Anónimo")
        text = review.get("text", "")
        review_rating = review.get("rating", 0)
        reviews_text += f"Review {i} (Rating: {review_rating}/5 - {author}):\n{text}\n\n"

    result_markdown = analyze_with_claude(reviews_text)

    return {
        "result_markdown": result_markdown,
        "meta": {
            "requested_reviews": 5,
            "received_reviews": len(reviews_list),
            "reviews_with_text": reviews_with_text_count,
            "estimated_reviews": reviews_with_text_count,
            "data_source": "Google Places (Maps)",
            "google_rating": rating,
            "total_ratings": total_reviews
        }
    }


@app.get("/")
async def root():
    return {"message": "Resumo Honesto de Reviews API - MVP (até 20 reviews)"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
