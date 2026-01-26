import os
import base64
from typing import Dict, Any, List

import httpx
from dotenv import load_dotenv
from anthropic import Anthropic

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

app = FastAPI()

# --- CORS (permitir TODOS os domínios - TEMPORÁRIO PARA TESTE) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TEMPORÁRIO: permite tudo para testar
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Prompt embutido (v1.4 Modo Agência) ---
PROMPT_BASE = """OneSynth – Prompt Modo Agência (v1.4)
Elabora uma síntese profissional a partir das opiniões de hóspedes de um hotel, seguindo rigorosamente a estrutura e as diretrizes abaixo. O objetivo é fornecer um relatório objetivo que auxilie uma agência de viagens a avaliar o hotel e a alinhar as expectativas do cliente.

Estrutura do Relatório
Identificação
Indicar o nome do hotel e informações contextuais em linhas separadas:
Hotel: [nome do hotel]
Localização: [cidade / região]
Fonte das opiniões: [origem das reviews]

Avaliação global de adequação
Apresentar um resumo curto (1–2 frases) sobre a adequação geral do hotel, com base nas reviews. Deve incluir uma avaliação geral imparcial.

Pontos fortes recorrentes
Listar os principais aspetos positivos mencionados com frequência pelos hóspedes, de forma concisa (bullet points telegráficos). Devem refletir pontos fortes consistentes nas opiniões (ex.: qualidade do pequeno-almoço, localização central, limpeza dos quartos, atendimento, etc.).

Pontos fracos recorrentes
Listar os principais aspetos negativos ou críticas recorrentes mencionadas pelos hóspedes (bullet points concisos), evidenciando pontos fracos comuns (ex.: quartos pequenos, ruído noturno, Wi-Fi instável, pouca higiene, pouca manutenção, hotel anticuado, etc.).

Aspetos dependentes do perfil do cliente
Listar características do hotel cuja apreciação varia consoante o perfil, expectativas ou preferências do cliente, diferenciando claramente perceções distintas conforme o tipo de hóspede.
Exemplos:
barulho e perturbação
tamanho dos quartos
horário de serviço
higiene
manutenção
hotel anticuado ou recente
atendimento
preço

Recomendado para
Indicar, em bullet points, os perfis de clientes ou situações para os quais o hotel é particularmente adequado, de acordo com os pontos fortes e características identificadas (ex.: viajantes em negócios, casais que procuram tranquilidade, estadias curtas de passagem, famílias com crianças, etc.).

Não recomendado se o cliente
Indicar, em bullet points, os tipos de clientes, preferências ou prioridades para os quais o hotel poderá não ser a melhor escolha, com base nos pontos fracos ou características do hotel (ex.: não é ideal para famílias com crianças pequenas se o hotel for orientado para adultos; não recomendado para quem procura vida noturna agitada, se o ambiente for calmo).

Alertas pontuais
Mencionar quaisquer problemas específicos ou temporários referidos nas opiniões que sejam relevantes destacar (ex.: obras em curso no hotel, incidência pontual de insetos, alteração recente de gestão).
Cada alerta deve ser apresentado como um bullet point separado. Caso não existam alertas relevantes, esta secção deve permanecer vazia.

Nota metodológica
Incluir uma nota final padronizada esclarecendo a base da síntese. Utilizar sempre o seguinte texto, sem alterações:
"Esta síntese baseia-se exclusivamente na análise de padrões recorrentes em opiniões públicas de hóspedes, com o objetivo de apoiar a recomendação profissional e alinhar expectativas do cliente."

Obrigatoriedade de apresentação no output final
Independentemente do volume, diversidade ou natureza das reviews analisadas, o output final apresentado deve conter sempre, de forma clara e explicitamente identificável, as seguintes secções:
Avaliação global de adequação
Pontos fortes recorrentes
Pontos fracos recorrentes
Recomendado para
Não recomendado se o cliente
Estas secções são consideradas nucleares para apoio à decisão profissional e não devem ser omitidas em nenhuma circunstância.

Diretrizes de Redação e Estilo
Neutralidade e objetividade
Manter um tom isento e descritivo. Apresentar os factos reportados nas opiniões de forma imparcial, sem julgamentos de valor ou opiniões próprias do analista.
Baseado em padrões
Fundamentar a síntese nos padrões recorrentes identificados nas reviews, focando o consenso ou tendências gerais. Evitar dar destaque desproporcional a comentários isolados, exceto quando representem alertas relevantes.
Tom profissional
Redigir em linguagem formal, mas acessível, semelhante a um relatório de consultoria. O texto deve ser claro, direto e adequado a um contexto profissional de recomendação ao cliente.
Consistência
Assegurar coerência na terminologia e no estilo ao longo de todo o relatório. Utilizar sempre o termo "hóspedes" para referir os autores das avaliações e manter um formato uniforme em todas as secções.
Estrutura fixa
Seguir rigorosamente os títulos e a organização das secções conforme definido. Todas as secções devem estar presentes na síntese final, mesmo que alguma tenha conteúdo limitado (nesse caso, indicar claramente a ausência ou insuficiência de dados).

Proibições explícitas
Não usar linguagem promocional ou adjetivos superlativos não sustentados pelas opiniões.
Não fazer generalizações absolutas sem suporte; relativizar sempre que necessário.
Não introduzir informações que não estejam presentes nas opiniões dos hóspedes.
Não dirigir o texto diretamente ao cliente nem utilizar linguagem subjetiva ou informal.
Não mencionar o processo de análise nem a existência de qualquer modelo de IA.

Mantendo estes princípios e esta estrutura, produzir uma síntese final clara, coerente e útil para a recomendação hoteleira profissional, sem viés ou informação indevida. O resultado deve ser um relatório sucinto e informativo, refletindo fielmente as opiniões dos hóspedes analisadas e destacando os aspetos mais relevantes para diferentes perfis de clientes.

Fim do prompt"""

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
OUTSCRAPER_API_KEY = os.getenv("OUTSCRAPER_API_KEY", "")

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

        first_result = results[0]
        place_id = first_result.get("place_id")
        if not place_id:
            raise HTTPException(status_code=404, detail="Hotel sem place_id (Google Places)")

        # Guardar nome e endereço do hotel encontrado
        found_hotel_name = first_result.get("name", "")
        formatted_address = first_result.get("formatted_address", "")

        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        details_params = {
            "place_id": place_id,
            "fields": "reviews,rating,user_ratings_total,photos,name,formatted_address",
            "key": GOOGLE_PLACES_API_KEY,
        }

        r = await client.get(details_url, params=details_params)
        r.raise_for_status()
        details = r.json()

        result = details.get("result") or {}
        reviews = result.get("reviews") or []

        if not reviews:
            raise HTTPException(status_code=404, detail="Sem reviews (Google Places)")

        # Obter foto do hotel e converter para base64
        photo_url = None
        photos = result.get("photos") or []
        if photos:
            photo_reference = photos[0].get("photo_reference")
            if photo_reference:
                try:
                    photo_api_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photo_reference={photo_reference}&key={GOOGLE_PLACES_API_KEY}"
                    photo_response = await client.get(photo_api_url, follow_redirects=True)
                    if photo_response.status_code == 200:
                        content_type = photo_response.headers.get("content-type", "image/jpeg")
                        photo_base64 = base64.b64encode(photo_response.content).decode("utf-8")
                        photo_url = f"data:{content_type};base64,{photo_base64}"
                except Exception:
                    pass

        # Usar nome/endereço dos detalhes se disponível
        if result.get("name"):
            found_hotel_name = result.get("name")
        if result.get("formatted_address"):
            formatted_address = result.get("formatted_address")

        return {
            "reviews": reviews,
            "reviews_with_text_count": len([rv for rv in reviews if (rv.get("text") or "").strip()]),
            "rating": result.get("rating"),
            "reviews_count": result.get("user_ratings_total"),
            "found_hotel_name": found_hotel_name,
            "formatted_address": formatted_address,
            "photo_url": photo_url,
        }


async def fetch_reviews_outscraper(hotel_name: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Busca reviews do Outscraper - mais recentes e com texto detalhado.
    Usa polling para aguardar resultados assíncronos.
    """
    if not OUTSCRAPER_API_KEY:
        return []

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            url = "https://api.app.outscraper.com/maps/reviews-v3"
            params = {
                "query": hotel_name,
                "reviewsLimit": 2,
                "sort": "newest",
                "ignoreEmpty": True,
            }
            headers = {
                "X-API-KEY": OUTSCRAPER_API_KEY,
            }

            # Fazer request inicial
            response = await client.get(url, params=params, headers=headers)

            # Se retornar 202, precisamos fazer polling
            if response.status_code == 202:
                data = response.json()
                results_url = data.get("results_location")
                if not results_url:
                    return []

                # Polling - aguardar até 60 segundos
                import asyncio
                for _ in range(12):  # 12 tentativas x 5 segundos = 60 seg
                    await asyncio.sleep(5)
                    poll_response = await client.get(results_url, headers=headers)
                    if poll_response.status_code == 200:
                        poll_data = poll_response.json()
                        if poll_data.get("status") == "Success" and poll_data.get("data"):
                            data = poll_data
                            break
                else:
                    return []  # Timeout
            elif response.status_code == 200:
                data = response.json()
            else:
                print(f"Outscraper erro: {response.status_code} - {response.text[:200]}")
                return []

            # Processar resultados
            if not data.get("data") or len(data["data"]) == 0:
                return []

            place_data = data["data"][0]
            if not isinstance(place_data, dict):
                return []

            reviews_list = place_data.get("reviews_data") or []

            # Filtrar reviews com texto substancial
            reviews_with_text = [
                rv for rv in reviews_list
                if rv.get("review_text") and len(rv.get("review_text", "").strip()) > 150
            ]

            # Ordenar por tamanho do texto (mais detalhadas primeiro)
            reviews_with_text.sort(key=lambda x: len(x.get("review_text", "")), reverse=True)

            # Converter para formato padronizado
            formatted_reviews = []
            for rv in reviews_with_text[:limit]:
                formatted_reviews.append({
                    "rating": rv.get("review_rating", 0),
                    "text": rv.get("review_text", ""),
                    "author_name": rv.get("author_title", "Anónimo"),
                    "time": rv.get("review_datetime_utc", ""),
                    "source": "Outscraper"
                })

            return formatted_reviews

    except Exception as e:
        print(f"Erro Outscraper: {e}")
        return []


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

    # Buscar reviews de ambas as fontes em paralelo
    google_data = await fetch_reviews_google_places(hotel)
    outscraper_reviews = await fetch_reviews_outscraper(hotel, limit=2)

    # Combinar reviews: 5 do Google Places + 2 do Outscraper
    reviews_text = ""
    review_count = 0

    # Google Places reviews (até 5)
    google_reviews = google_data["reviews"][:5]
    for rv in google_reviews:
        text = (rv.get("text") or "").strip()
        if text:
            review_count += 1
            reviews_text += (
                f"Review {review_count} ({rv.get('rating', 0)}/5 - {rv.get('author_name','Anónimo')} - Google):\n"
                f"{text}\n\n"
            )

    # Outscraper reviews (até 2)
    for rv in outscraper_reviews:
        text = (rv.get("text") or "").strip()
        if text:
            review_count += 1
            reviews_text += (
                f"Review {review_count} ({rv.get('rating', 0)}/5 - {rv.get('author_name','Anónimo')} - Outscraper):\n"
                f"{text}\n\n"
            )

    result_markdown = analyze_with_claude(reviews_text)

    return {
        "result_markdown": result_markdown,
        "meta": {
            "received_reviews": review_count,
            "google_reviews": len(google_reviews),
            "outscraper_reviews": len(outscraper_reviews),
            "reviews_with_text": google_data["reviews_with_text_count"],
            "google_rating": google_data["rating"],
            "total_ratings": google_data["reviews_count"],
            "found_hotel_name": google_data.get("found_hotel_name", ""),
            "formatted_address": google_data.get("formatted_address", ""),
            "photo_url": google_data.get("photo_url"),
        },
    }