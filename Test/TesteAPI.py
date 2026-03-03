"""
TesteAPI.py
===========

Testa o orquestrador diretamente, sem Chatwoot nem servidor HTTP.

O ChatwootClient é substituído por um mock que captura todas as
chamadas (send_message, set_labels, assign_team, etc.) em memória
e as exibe ao final de cada teste.

Uso:
    cd /home/nees/Documents/VSCodigo/AgenteFastAPI
    source myenv/bin/activate
    python Test/TesteAPI.py

Variáveis de ambiente necessárias (no .env):
    MARITALK_API_KEY=...
"""

import asyncio
import logging
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Garante que o diretório raiz do projeto está no path
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Logging: INFO visível, libs de ML silenciadas
# ---------------------------------------------------------------------------
for _lib in ("sentence_transformers", "transformers", "mlx_lm", "huggingface_hub",
             "agno", "httpx", "lancedb"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("teste_api")


# ---------------------------------------------------------------------------
# Mock do ChatwootClient
# ---------------------------------------------------------------------------
@dataclass
class MockCall:
    method: str
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)


class MockChatwootClient:
    """
    Substitui o ChatwootClient real.
    Registra cada chamada em `self.calls` e exibe no console,
    sem fazer nenhuma requisição HTTP.
    """

    def __init__(self):
        self.calls: list[MockCall] = []
        self._team_cache: dict[str, int] = {"suporte": 1, "support": 1}
        # Simula times disponíveis
        self._mock_teams = [{"id": 1, "name": "Suporte"}, {"id": 2, "name": "Financeiro"}]

    def _record(self, method: str, *args, **kwargs):
        self.calls.append(MockCall(method, args, kwargs))

    async def send_message(self, conversation_id, account_id, content, message_type="outgoing"):
        self._record("send_message", conversation_id, account_id, content)
        print(f"\n  📤 RESPOSTA AO USUÁRIO:\n{textwrap.indent(content, '     ')}")
        return {"id": 999, "content": content}

    async def set_labels(self, conversation_id, account_id, labels):
        self._record("set_labels", conversation_id, account_id, labels)
        print(f"  🏷  Labels definidas: {labels}")
        return {"payload": labels}

    async def assign_team(self, conversation_id, account_id, team_id):
        self._record("assign_team", conversation_id, account_id, team_id)
        print(f"  👥  Time atribuído: id={team_id}")
        return {"id": team_id}

    async def update_conversation_meta(self, conversation_id, account_id,
                                        custom_attributes=None, team_id=None,
                                        clear_assignment=False):
        self._record("update_conversation_meta", conversation_id, account_id,
                     custom_attributes=custom_attributes, team_id=team_id)
        if custom_attributes:
            route   = custom_attributes.get("orchestrator_route", "?")
            reason  = custom_attributes.get("orchestrator_reason", "?")
            handled = custom_attributes.get("handled_by", "?")
            conf    = custom_attributes.get("orchestrator_confidence", "?")
            print(f"  📊  Meta → route={route!r}  reason={reason!r}  "
                  f"handled_by={handled!r}  confidence={conf}")
        return {}

    async def set_conversation_open(self, conversation_id, account_id):
        self._record("set_conversation_open", conversation_id, account_id)
        return {}

    async def resolve_team_id(self, account_id, team_name_or_id) -> Optional[int]:
        if not team_name_or_id:
            return None
        if str(team_name_or_id).isdigit():
            return int(team_name_or_id)
        return self._team_cache.get(str(team_name_or_id).casefold(), 1)

    async def _list_teams(self, account_id):
        return self._mock_teams

    async def close(self):
        pass

    def reset(self):
        self.calls.clear()


# ---------------------------------------------------------------------------
# Casos de teste
# ---------------------------------------------------------------------------
TEST_CASES = [
    # (descrição, mensagem, canal, labels_iniciais)
    ("Saudação simples",
     "Bom dia!",
     "chat", []),

    ("Pergunta FAQ – MEC",
     "Como faço a adesão da minha secretaria na Plataforma MEC Gestão Presente?",
     "chat", []),

    ("Pergunta técnica – MEC",
     "Os CPFs precisam ter exatamente 11 dígitos mesmo começando com zero?",
     "chat", []),

    ("Pedido de humano – explícito",
     "Quero falar com um atendente humano",
     "chat", []),

    ("Frustração / escalada implícita",
     "já tentei de tudo que falaram no manual e o erro continua dando na linha 40",
     "chat", []),

    ("Pergunta via e-mail (canal email)",
     "Qual é a relação entre o SGP e o GPE?",
     "email", []),

    ("Agradecimento",
     "Obrigado pela ajuda!",
     "chat", []),

    ("Pergunta sobre API/integração",
     "Qual é o fluxo de solicitação da chave da API?",
     "chat", []),
]


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------
async def run_tests():
    logger.info("=== Inicializando sistema RAG (sem Chatwoot) ===")

    # Importações tardias para não carregar antes do logging estar configurado
    from AgenteSabia import MecSpecialistAgent, RagSystem

    rag = RagSystem()

    # Carrega documentos (skip se já existirem no LanceDB)
    logger.info("Verificando/carregando documentos da base de conhecimento…")
    rag.load_documents(recreate=False)

    mock_chatwoot = MockChatwootClient()

    specialist = MecSpecialistAgent(rag)

    # Importa o orquestrador com o mock no lugar do ChatwootClient real
    from OrquestradorAPI import MessageOrchestratorAgent, fold_text

    orchestrator = MessageOrchestratorAgent(specialist, mock_chatwoot)

    # Pré-aquece o classificador HF
    logger.info("Pré-aquecendo classificador semântico…")
    orchestrator._hf_classifier.warmup()
    logger.info("Sistema pronto!\n")

    sep = "─" * 70

    for idx, (desc, message, channel, labels) in enumerate(TEST_CASES, 1):
        print(f"\n{sep}")
        print(f"  [{idx}/{len(TEST_CASES)}] {desc}")
        print(f"  Canal : {channel!r}")
        print(f"  Labels: {labels or '(nenhuma)'}")
        print(f"  Msg   : {message!r}")
        print(sep)

        mock_chatwoot.reset()

        try:
            await orchestrator.handle_incoming(
                conversation_id=1000 + idx,
                account_id=1,
                content=message,
                current_labels=labels,
                force_ia_label=False,
                channel_type=channel,
            )
        except Exception as exc:
            print(f"  ❌  ERRO: {exc}")
            logger.exception("Erro no teste %d", idx)

    print(f"\n{sep}")
    print("  ✅  Todos os testes concluídos.")
    print(sep)


# ---------------------------------------------------------------------------
# Menu interativo (opcional)
# ---------------------------------------------------------------------------
async def run_interactive():
    """Modo interativo: digita mensagens e vê o roteamento em tempo real."""
    from AgenteSabia import MecSpecialistAgent, RagSystem
    from OrquestradorAPI import MessageOrchestratorAgent

    logger.info("=== Modo interativo ===")
    rag = RagSystem()
    rag.load_documents(recreate=False)

    mock_chatwoot = MockChatwootClient()
    specialist = MecSpecialistAgent(rag)
    orchestrator = MessageOrchestratorAgent(specialist, mock_chatwoot)
    orchestrator._hf_classifier.warmup()
    logger.info("Pronto! Digite 'sair' para encerrar.\n")

    conv_id = 9000
    labels: list[str] = []

    while True:
        try:
            raw = input("Você > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw or raw.lower() in {"sair", "exit", "quit"}:
            break

        channel = "chat"
        if raw.lower().startswith("email:"):
            channel = "email"
            raw = raw[6:].strip()

        mock_chatwoot.reset()
        conv_id += 1
        print()
        try:
            await orchestrator.handle_incoming(
                conversation_id=conv_id,
                account_id=1,
                content=raw,
                current_labels=labels,
                force_ia_label=False,
                channel_type=channel,
            )
        except Exception as exc:
            print(f"  ❌  ERRO: {exc}")
        print()

    print("Encerrando modo interativo.")


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "batch"

    if mode == "interactive":
        asyncio.run(run_interactive())
    else:
        asyncio.run(run_tests())
