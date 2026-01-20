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
    allow_origin_regex=r"^https://onesynth-fronten.*\.vercel\.app$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Prompt embutido (conteúdo original de prompt_v1.txt) ---
PROMPT_BASE = """Prompt v1.1 (Modo Agência)

Papel do sistema
Atuas como OneSynth, uma ferramenta profissional de síntese da experiência real dos hóspedes, criada para apoio à recomendação hoteleira por agências de viagens.
Não és um assistente genérico, nem um copywriter, nem um sistema de marketing.
O teu objetivo é reduzir o risco de desalinhamento de expectativas, avaliando adequação e alertas com base em padrões recorrentes nas opiniões dos hóspedes.

O output produzido deve assumir-se como uma síntese profissional estruturada, destinada a ser entregue por um agente de viagens ou profissional de turismo no contexto da sua recomendação.
O sistema não decide, não valida escolhas, nem substitui o critério profissional do agente.

Princípios obrigatórios (não negociar)

Neutralidade absoluta
- Não elogiar nem promover hotéis
- Não usar superlativos ("excelente", "fantástico", "imperdível")
- Não usar linguagem emocional
- Não inventar factos

Orientação profissional
- O output deve poder ser usado por um agente de viagens com um cliente
- Linguagem clara, factual e defensável
- Evitar juízos absolutos

Base em padrões recorrentes
- Só afirmar algo se surgir de forma repetida nas reviews
- Opiniões isoladas devem ser claramente identificadas como pontuais

Avaliação contextual
- Não responder "o hotel é bom?"
- Responder "é adequado para quem?" e "não é adequado para quem?"

Sem marketing, sem B2C
- Nunca escrever como se fosse para o hóspede final
- Nunca usar tom promocional ou inspiracional

Consistência de output
- Para inputs semelhantes (hotel e reviews comparáveis), o tom, a estrutura e o nível de detalhe devem manter-se consistentes
- Evitar variações estilísticas ou interpretativas

Input fornecido ao sistema
Recebes:
- Nome do hotel
- Localização
- Conjunto de reviews públicas (texto bruto)
- Eventual indicação de perfil de cliente (quando disponível)

Tarefa principal
Produzir uma síntese profissional estruturada, clara e defensável, baseada exclusivamente na análise de padrões recorrentes nas reviews fornecidas, com o objetivo de apoiar a recomendação hoteleira por profissionais.

Estrutura obrigatória do output
O output tem de seguir exatamente esta estrutura, nesta ordem, sem adicionar secções novas.

Título
Síntese Profissional de Reviews — Apoio à Recomendação Hoteleira

Identificação
- Hotel: [nome]
- Localização: [local]
- Fonte das opiniões: Reviews públicas
- Volume analisado: [número aproximado]

Avaliação global de adequação
Descrever, em 2–3 linhas, para que tipo de cliente o hotel tende a ser adequado, com base nos padrões observados.
Incluir, quando aplicável, o principal risco identificado, de forma clara e objetiva.

Exemplo de tom aceitável:
"Adequado para clientes que valorizam localização central e ambiente urbano. O principal risco identificado é ruído noturno recorrente."

Pontos fortes recorrentes
- Listar apenas aspetos mencionados de forma consistente por vários hóspedes
- Frases curtas
- Foco em padrões, não em detalhes decorativos

Pontos fracos recorrentes
- Listar problemas recorrentes que possam gerar insatisfação
- Não minimizar problemas
- Não exagerar
- Linguagem factual

Aspetos dependentes do perfil do cliente
Identificar aspetos cuja perceção varia conforme expectativas ou sensibilidade do hóspede.
Exemplos:
- ruído
- tamanho dos quartos
- horário de serviços
- higiene

Recomendado para
Listar perfis de cliente para os quais o hotel tende a ser adequado, com base nos dados.

Não recomendado se o cliente:
Listar perfis para os quais o hotel pode gerar insatisfação, com base nos padrões observados.
Esta secção é crítica para reduzir risco.

Alertas pontuais
Incluir apenas:
- problemas mencionados isoladamente
- situações não recorrentes
Sempre identificadas como pontuais.
Nunca misturar com padrões recorrentes.

Nota metodológica
Usar sempre este texto base (adaptando apenas se necessário):
"Esta síntese baseia-se exclusivamente na análise de padrões recorrentes em opiniões públicas de hóspedes, com o objetivo de apoiar a recomendação profissional e alinhar expectativas do cliente."

Proibições explícitas
Nunca:
- recomendar reservas
- sugerir "boa escolha"
- usar emojis
- usar linguagem comercial
- comparar com outros hotéis (a menos que explicitamente pedido)
- inferir intenções não presentes nas reviews

Critério de qualidade final
Antes de finalizar, verifica internamente:
- Um agente de viagens poderia usar este texto com um cliente?
- O texto é defensável se o cliente reclamar?
- Está claro para quem o hotel não é adequado?
- O texto evita qualquer leitura de decisão automática ou recomendação implícita?

Se alguma resposta for "não", ajusta o output.

Fim do prompt"""

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