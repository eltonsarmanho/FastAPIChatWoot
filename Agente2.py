"""
Agente 2: Especialista em MEC
==============================

Responsável por responder perguntas específicas sobre MEC, Regimento,
Resoluções, TCC, ACC, etc. com confiança e rastreamento.

Contém toda a lógica do sistema RAG:
  - Configuração do modelo de linguagem (Maritaca Sabiá)
  - Base de conhecimento vetorial (LanceDB + SentenceTransformer)
  - Carregamento de documentos Markdown
  - Gerenciamento de agentes por sessão
  - Cache de respostas
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from agno.agent import Agent
from agno.db.sqlite.sqlite import SqliteDb
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.openai import OpenAILike
from agno.vectordb.lancedb import LanceDb
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
load_dotenv(override=True)

logger = logging.getLogger("agente_rag")

MARITALK_API_KEY: str = os.getenv("MARITALK_API_KEY", "")
DOCS_FOLDER: str = os.getenv("DOCS_FOLDER", "Docs")
DB_FILE: str = os.getenv("DB_FILE", "data.db")
LANCEDB_URI: str = os.getenv("LANCEDB_URI", "lancedb")
RAG_MAX_DOCS: int = int(os.getenv("RAG_MAX_DOCS", "5"))
RESPONSE_CACHE_TTL_SECONDS: int = int(os.getenv("RESPONSE_CACHE_TTL_SECONDS", "300"))
RESPONSE_CACHE_MAX_ITEMS: int = int(os.getenv("RESPONSE_CACHE_MAX_ITEMS", "256"))


# ---------------------------------------------------------------------------
# Instruções dos agentes por tipo de canal
# ---------------------------------------------------------------------------
_INSTRUCTIONS_CHAT: str = (
    "Você é um assistente inteligente especializado nos documentos internos da organização.\n"
    "Responda de forma clara, objetiva e precisa utilizando o conhecimento disponível nos documentos.\n"
    "Caso a informação não esteja disponível nos documentos, informe ao usuário de forma educada.\n"
    "Responda sempre no mesmo idioma da pergunta."
)

_INSTRUCTIONS_EMAIL: str = (
    "Você é um assistente institucional que responde e-mails formais em nome da organização.\n"
    "Utilize sempre linguagem formal e institucional em suas respostas.\n"
    "Estrutura obrigatória da resposta:\n"
    "  - Inicie com saudação formal: 'Prezado(a),'\n"
    "  - Desenvolva a resposta de forma completa, clara e embasada nos documentos internos.\n"
    "  - Finalize com: 'Atenciosamente,\\nEquipe de Suporte'\n"
    "Caso a informação não esteja disponível nos documentos, informe com educação e formalidade.\n"
    "Responda sempre no mesmo idioma da pergunta."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def looks_like_no_answer(answer: str) -> bool:
    """Heurística simples para identificar quando a IA não encontrou resposta suficiente."""
    normalized = answer.strip().lower()
    fallback_markers = [
        "não encontrei",
        "nao encontrei",
        "não está disponível",
        "nao esta disponivel",
        "não tenho essa informação",
        "nao tenho essa informacao",
        "não consta nos documentos",
        "nao consta nos documentos",
    ]
    return any(marker in normalized for marker in fallback_markers)


# ---------------------------------------------------------------------------
# Sistema RAG
# ---------------------------------------------------------------------------
class RagSystem:
    """Gerencia a base de conhecimento e os agentes por sessão."""

    def __init__(self) -> None:
        if not MARITALK_API_KEY:
            raise ValueError("MARITALK_API_KEY é obrigatória no arquivo .env")

        self._agents: Dict[str, Agent] = {}
        self._response_cache: dict[tuple[str, str], tuple[str, float]] = {}
        self._setup_model()
        self._setup_knowledge()

    # ------------------------------------------------------------------
    # Inicialização dos componentes
    # ------------------------------------------------------------------
    def _setup_model(self) -> None:
        """Configura o modelo de linguagem (Maritaca Sabiá-3)."""
        self.model = OpenAILike(
            id="sabiazinho-4",
            name="Maritaca Sabia 4",
            api_key=MARITALK_API_KEY,
            base_url="https://chat.maritaca.ai/api",
            temperature=0,
        )

    def _setup_knowledge(self) -> None:
        """Configura a base de conhecimento vetorial com LanceDB."""
        embedder = SentenceTransformerEmbedder()
        vector_db = LanceDb(
            table_name="docs_knowledge",
            uri=LANCEDB_URI,
            embedder=embedder,
        )
        self.knowledge = Knowledge(
            name="Documentos Internos",
            vector_db=vector_db,
            max_results=RAG_MAX_DOCS,
        )

    # ------------------------------------------------------------------
    # Carregamento de documentos
    # ------------------------------------------------------------------
    def _has_existing_data(self) -> bool:
        """
        Verifica rapidamente se a tabela LanceDB já contém registros,
        evitando re-processar embeddings desnecessariamente no restart.
        """
        try:
            import lancedb
            db = lancedb.connect(LANCEDB_URI)
            if "docs_knowledge" in db.table_names():
                return db.open_table("docs_knowledge").count_rows() > 0
        except Exception as exc:
            logger.debug(f"[load_documents] Não foi possível verificar dados existentes: {exc}")
        return False

    def load_documents(self, recreate: bool = False) -> None:
        """
        Carrega todos os arquivos .md da pasta Docs na base de conhecimento.

        Se ``recreate=False`` e a tabela já tiver dados, o carregamento é
        ignorado para evitar re-computar embeddings a cada restart.

        Args:
            recreate: Se True, limpa e recarrega todos os vetores.
        """
        docs_path = Path(DOCS_FOLDER)
        if not docs_path.exists():
            logger.warning(f"Pasta de documentos não encontrada: {DOCS_FOLDER}")
            return

        md_files = list(docs_path.glob("**/*.md"))
        if not md_files:
            logger.warning(f"Nenhum arquivo .md encontrado em {DOCS_FOLDER}")
            return

        # Atalho: se não é recriação e o DB já tem dados, pula o carregamento.
        # Isso evita que o agno releia, rechunke e re-embedde todos os arquivos
        # apenas para depois concluir que eles já existem (skip_if_exists).
        if not recreate and self._has_existing_data():
            logger.info(
                f"[load_documents] Base já contém dados e recreate=False — "
                f"carregamento ignorado. Use /reload-docs?recreate=true para forçar."
            )
            return

        logger.info(f"Carregando {len(md_files)} documento(s) de '{DOCS_FOLDER}'...")

        if recreate:
            try:
                self.knowledge.remove_all_content()
                logger.info("Base de conhecimento limpa para recriação.")
            except Exception as exc:
                logger.warning(f"Não foi possível limpar a base: {exc}")

        for md_file in md_files:
            try:
                self.knowledge.insert(
                    name=md_file.stem,
                    path=str(md_file),
                    skip_if_exists=not recreate,
                )
                logger.info(f"  ✓ {md_file.name}")
            except Exception as exc:
                logger.error(f"  ✗ Erro ao carregar '{md_file.name}': {exc}")

        logger.info("Documentos carregados com sucesso!")

    # ------------------------------------------------------------------
    # Gerenciamento de agentes por sessão
    # ------------------------------------------------------------------
    def get_agent(self, session_id: str, channel_type: str = "chat") -> Agent:
        """
        Retorna o agente associado a uma sessão (conversa do Chatwoot).
        Cria um novo agente se ainda não existir para esta sessão.

        Args:
            session_id:   ID único da conversa (ex.: 'chatwoot_123').
            channel_type: 'email' ou 'chat' – define tom e formato das respostas.
        """
        if session_id not in self._agents:
            instructions = _INSTRUCTIONS_EMAIL if channel_type == "email" else _INSTRUCTIONS_CHAT
            db = SqliteDb(db_file=DB_FILE)
            self._agents[session_id] = Agent(
                model=self.model,
                name="Assistente RAG",
                knowledge=self.knowledge,
                db=db,
                session_id=session_id,
                search_knowledge=True,   # busca semântica: só os chunks relevantes
                add_knowledge_to_context=False,  # evita injetar TODO o conhecimento
                telemetry=False,
                instructions=instructions,
            )
            logger.info(f"Novo agente criado para sessão: {session_id} (canal={channel_type})")
        return self._agents[session_id]

    @staticmethod
    def _normalize_question(question: str) -> str:
        return re.sub(r"\s+", " ", question.strip().lower())

    @staticmethod
    def _is_quick_smalltalk(question: str) -> bool:
        q = RagSystem._normalize_question(question)
        if len(q) > 40:
            return False
        return q in {
            "oi",
            "ola",
            "olá",
            "bom dia",
            "boa tarde",
            "boa noite",
            "tudo bem",
            "ok",
            "obrigado",
            "obrigada",
            "valeu",
        }

    def _get_cached_answer(self, session_id: str, question: str) -> str | None:
        key = (session_id, self._normalize_question(question))
        cached = self._response_cache.get(key)
        if not cached:
            return None
        answer, ts = cached
        if (time.time() - ts) > RESPONSE_CACHE_TTL_SECONDS:
            self._response_cache.pop(key, None)
            return None
        return answer

    def _cache_answer(self, session_id: str, question: str, answer: str) -> None:
        if not answer:
            return
        if len(self._response_cache) >= RESPONSE_CACHE_MAX_ITEMS:
            # Remove item mais antigo para manter o cache pequeno e rápido.
            oldest_key = min(self._response_cache, key=lambda k: self._response_cache[k][1])
            self._response_cache.pop(oldest_key, None)
        key = (session_id, self._normalize_question(question))
        self._response_cache[key] = (answer, time.time())

    # ------------------------------------------------------------------
    # Processamento de perguntas
    # ------------------------------------------------------------------
    def ask(self, question: str, session_id: str, channel_type: str = "chat") -> str:
        """
        Envia uma pergunta ao agente da sessão e retorna a resposta.

        Args:
            question:     Texto da pergunta/mensagem do usuário.
            session_id:   ID único da conversa (ex.: 'chatwoot_123').
            channel_type: 'email' ou 'chat' – define tom e formato das respostas.

        Returns:
            Texto da resposta gerada pelo agente.
        """
        # Para e-mail sempre gera resposta completa (ignora atalho de smalltalk),
        # pois e-mails formais merecem resposta elaborada mesmo para saudações.
        if channel_type != "email" and self._is_quick_smalltalk(question):
            return (
                "Olá! Posso te ajudar com dúvidas sobre os documentos internos. "
                "Me diga sua pergunta."
            )

        cached = self._get_cached_answer(session_id, question)
        if cached:
            logger.debug(f"[cache] hit sessão={session_id}")
            return cached

        agent = self.get_agent(session_id, channel_type)
        response = agent.run(question)
        answer = response.content or "Não foi possível gerar uma resposta."
        self._cache_answer(session_id, question, answer)
        return answer


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

    def __init__(self, rag: RagSystem) -> None:
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
