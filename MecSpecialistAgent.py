"""
Agente Especialista em MEC
===========================

Responsável por responder perguntas específicas sobre MEC, Regimento,
Resoluções, TCC, ACC, etc. com confiança e rastreamento.

Este módulo pode usar a API remota ou integrar-se com o AgenteSabia local.
"""

import logging
import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
load_dotenv(override=True)

logger = logging.getLogger("agente_rag")

AGENTE2_API_URL: str = os.getenv("AGENTE2_API_URL", "").strip()
AGENTE2_API_TOKEN: str = os.getenv("AGENTE2_API_TOKEN", "").strip()
AGENTE2_API_TIMEOUT_SECONDS: float = float(os.getenv("AGENTE2_API_TIMEOUT_SECONDS", "30"))


# ---------------------------------------------------------------------------
# Resultado do especialista
# ---------------------------------------------------------------------------
@dataclass
class SpecialistResult:
    """Resultado da resposta do especialista MEC."""
    answer: str
    confidence: float


# ---------------------------------------------------------------------------
# Agente Especialista MEC
# ---------------------------------------------------------------------------
class MecSpecialistAgent:
    """Agente especialista em questões de MEC/FASI."""

    def __init__(self, rag=None) -> None:
        """
        Inicializa o agente especialista.

        Args:
            rag: Instância de AgenteSabia (opcional). Se None, usa API remota.
        """
        self.rag = rag
        self.api_url = AGENTE2_API_URL
        self.api_token = AGENTE2_API_TOKEN
        self.api_timeout = AGENTE2_API_TIMEOUT_SECONDS

    def _answer_remote(self, question: str) -> SpecialistResult:
        """Obtém resposta via API remota."""
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_token:
            headers["x-token"] = self.api_token

        payload = {
            "message": question,
            "chat_history": [],
        }

        with httpx.Client(timeout=self.api_timeout) as client:
            response = client.post(
                self.api_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json() if response.content else {}

        answer = (
            data.get("answer")
            or data.get("response")
            or data.get("message")
            or "Não foi possível gerar uma resposta."
        )
        raw_confidence = data.get("confidence", 0.8 if answer else 0.0)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.8 if answer else 0.0
        return SpecialistResult(answer=answer, confidence=confidence)

    def answer(self, question: str, session_id: str, channel_type: str = "chat") -> SpecialistResult:
        """
        Processa uma pergunta sobre MEC e retorna resposta com confiança.

        Args:
            question:     Pergunta do usuário
            session_id:   ID da sessão/conversa
            channel_type: 'email' ou 'chat' – define tom e formato das respostas

        Returns:
            SpecialistResult com a resposta e nível de confiança
        """
        if self.api_url:
            return self._answer_remote(question)

        if not self.rag:
            raise RuntimeError(
                "AgenteSabia não inicializado e AGENTE2_API_URL não configurado."
            )

        response = self.rag.ask(question, session_id, channel_type)

        # Heurística simples de confiança
        confidence = 0.8 if response and len(response) > 20 else 0.5

        return SpecialistResult(
            answer=response,
            confidence=confidence
        )
