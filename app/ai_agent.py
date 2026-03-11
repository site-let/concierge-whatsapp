"""
ai_agent.py — Integração com Google Gemini API (gratuito)
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai

logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT_TEMPLATE = """Você é o assistente virtual do {hotel_name}, um concierge digital disponível 24 horas.

SEU COMPORTAMENTO:
- Tom: {personality}
- LÍNGUA: Responda SEMPRE na mesma língua em que o hóspede falar com você (Português, Inglês, Espanhol, etc.).
- Seja direto e objetivo — hóspedes no celular não querem textos longos.
- Use emojis com moderação (1-2 por mensagem, apenas quando natural).
- NUNCA invente informações. Se não souber, diga que vai verificar com a equipe.
- Se a pergunta for urgente ou fora do escopo, direcione para a recepção.

INFORMAÇÕES DO HOTEL (extraídas do manual):
─────────────────────────────────────────
{hotel_context}
─────────────────────────────────────────

REGRAS IMPORTANTES:
1. Responda APENAS com base nas informações acima
2. Se a informação não estiver no manual, diga: "Não tenho essa informação no momento, mas nossa recepção pode te ajudar! 😊"
3. Para emergências médicas ou de segurança, sempre indique ligar para 192 (SAMU) ou 193 (Bombeiros)
4. Mantenha as respostas curtas: máximo 3-4 linhas"""


async def get_concierge_response(
    guest_message: str,
    hotel_context: str,
    hotel_name: str,
    hotel_personality: str = "amigável e profissional",
    conversation_history: list = []
) -> str:
    try:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            hotel_name=hotel_name,
            personality=hotel_personality,
            hotel_context=hotel_context[:8000]
        )

        model = genai.GenerativeModel(
            model_name="gemma-3-4b-it"
        )

        history = []
        # Gemma pode não suportar system_instruction no construtor. 
        # Inserimos como primeira mensagem para garantir o contexto.
        history.append({"role": "user", "parts": [f"INSTRUÇÕES DO SISTEMA:\n{system_prompt}\n\nEntendido? Responda apenas com 'OK'."]})
        history.append({"role": "model", "parts": ["OK"]})

        for turn in conversation_history[-10:]:
            role = "user" if turn["role"] == "user" else "model"
            history.append({"role": role, "parts": [turn["content"]]})

        chat = model.start_chat(history=history)
        response = chat.send_message(guest_message)

        if not response.text:
             logger.warning("Gemini retornou resposta vazia ou bloqueada.")
             return "Desculpe, não consegui processar sua pergunta agora. Pode repetir?"

        answer = response.text.strip()
        logger.info(f"Gemini respondeu: {answer[:80]}...")
        return answer

    except Exception as e:
        logger.error(f"Erro no agente Gemini: {str(e)}", exc_info=True)
        return "Desculpe, estou com uma instabilidade no momento. Por favor, fale com nossa recepção. 🙏"
