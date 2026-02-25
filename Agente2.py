"""
Agente 2: Especialista em MEC
==============================

Responsável por responder perguntas específicas sobre MEC, Regimento,
Resoluções, TCC, ACC, etc. com confiança e rastreamento.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Agente import RagSystem


@dataclass
class SpecialistResult:
    """Resultado da resposta do especialista MEC."""
    answer: str
    confidence: float


class MecSpecialistAgent:
    """Agente especialista em questões de MEC/FASI."""

    def __init__(self, rag: "RagSystem") -> None:
        self.rag = rag

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
        response = self.rag.ask(question, session_id, channel_type)
        
        # Heurística simples de confiança
        confidence = 0.8 if response and len(response) > 20 else 0.5
        
        return SpecialistResult(
            answer=response,
            confidence=confidence
        )
