"""
Agente RAG com FastAPI + Chatwoot Webhook
==========================================

Sistema RAG que integra com Chatwoot via webhook.
Carrega documentos Markdown da pasta Docs e responde
perguntas automaticamente nas conversas do Chatwoot.

Fluxo:
  1. Chatwoot envia POST /webhook ao receber mensagem
  2. O agente processa via RAG (por conversa/sessão)
  3. A resposta é enviada de volta via API do Chatwoot
"""

import asyncio
import logging
import os
import re
import time
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Literal

import httpx
from agno.agent import Agent
from agno.db.sqlite.sqlite import SqliteDb
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.openai import OpenAILike
from agno.vectordb.lancedb import LanceDb
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from Agente2 import MecSpecialistAgent
from chatwoot_client import ChatwootClient

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
load_dotenv(override=True)

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agente_rag")

MARITALK_API_KEY: str = os.getenv("MARITALK_API_KEY", "")
CHATWOOT_API_URL: str = os.getenv("CHATWOOT_API_URL", "http://localhost:3000")
CHATWOOT_API_TOKEN: str = os.getenv("CHATWOOT_API_TOKEN", "")
ROBO_TOKEN: str = os.getenv("ROBO_TOKEN", "")
CHATWOOT_ACCOUNT_ID: str = os.getenv("CHATWOOT_ACCOUNT_ID", "1")
WEBHOOK_TOKEN: str = os.getenv("WEBHOOK_TOKEN", "")  # ?token= na URL do webhook
DOCS_FOLDER: str = os.getenv("DOCS_FOLDER", "Docs")
DB_FILE: str = os.getenv("DB_FILE", "data.db")
LANCEDB_URI: str = os.getenv("LANCEDB_URI", "lancedb")
RAG_MAX_DOCS: int = int(os.getenv("RAG_MAX_DOCS", "5"))
RESPONSE_CACHE_TTL_SECONDS: int = int(os.getenv("RESPONSE_CACHE_TTL_SECONDS", "300"))
RESPONSE_CACHE_MAX_ITEMS: int = int(os.getenv("RESPONSE_CACHE_MAX_ITEMS", "256"))
CHATWOOT_LABEL_HUMANO: str = os.getenv("CHATWOOT_LABEL_HUMANO", "humano")
CHATWOOT_LABEL_IA_ORQUESTRADOR: str = os.getenv("CHATWOOT_LABEL_IA_ORQUESTRADOR", "ia_orquestrador")
CHATWOOT_LABEL_IA_MEC: str = os.getenv("CHATWOOT_LABEL_IA_MEC", "ia_mec")
CHATWOOT_LABEL_IA_FALHA: str = os.getenv("CHATWOOT_LABEL_IA_FALHA", "ia_falha")
CHATWOOT_HUMAN_TEAM_ID: str = os.getenv("CHATWOOT_HUMAN_TEAM_ID", "")
TEAM: str = os.getenv("TEAM", "")
TEAM_DEFAULT_HUMAN: str = os.getenv("TEAM_DEFAULT_HUMAN", "Suporte")
ORCHESTRATOR_CONFIDENCE_THRESHOLD: float = float(os.getenv("ORCHESTRATOR_CONFIDENCE_THRESHOLD", "0.7"))
ORCHESTRATOR_USE_LLM_CLASSIFIER: bool = os.getenv(
    "ORCHESTRATOR_USE_LLM_CLASSIFIER", "false"
).lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _HTMLStripper(HTMLParser):
    """Remove todas as tags HTML e devolve texto puro."""
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        # join all parts and collapse whitespace
        return " ".join("".join(self._parts).split())


def strip_html(text: str) -> str:
    """Remove tags HTML do texto (ex.: '<p>Bom dia</p>' → 'Bom dia')."""
    if "<" not in text:
        return text.strip()
    stripper = _HTMLStripper()
    stripper.feed(text)
    return stripper.get_text()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def fold_text(text: str) -> str:
    """Lowercase + remove acentos para comparação semântica."""
    lowered = normalize_text(text)
    return "".join(
        c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn"
    )


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


@dataclass
class IntentDecision:
    route: Literal["mec", "direct", "human"]
    reason: str
    requested_human: bool = False
    requested_ai: bool = False
    requested_team: str | None = None  # time extraído pelo LLM ou padrão


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
    def load_documents(self, recreate: bool = False) -> None:
        """
        Carrega todos os arquivos .md da pasta Docs na base de conhecimento.

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
    def get_agent(self, session_id: str) -> Agent:
        """
        Retorna o agente associado a uma sessão (conversa do Chatwoot).
        Cria um novo agente se ainda não existir para esta sessão.
        """
        if session_id not in self._agents:
            db = SqliteDb(db_file=DB_FILE)
            self._agents[session_id] = Agent(
                model=self.model,
                name="Assistente RAG",
                knowledge=self.knowledge,
                db=db,
                session_id=session_id,
                search_knowledge=False,
                add_knowledge_to_context=True,
                telemetry=False,
                instructions="""Você é um assistente inteligente especializado nos documentos internos da organização.
Responda de forma clara, objetiva e precisa utilizando o conhecimento disponível nos documentos.
Caso a informação não esteja disponível nos documentos, informe ao usuário de forma educada.
Responda sempre no mesmo idioma da pergunta.""",
            )
            logger.info(f"Novo agente criado para sessão: {session_id}")
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
    def ask(self, question: str, session_id: str) -> str:
        """
        Envia uma pergunta ao agente da sessão e retorna a resposta.

        Args:
            question:   Texto da pergunta/mensagem do usuário.
            session_id: ID único da conversa (ex.: 'chatwoot_123').

        Returns:
            Texto da resposta gerada pelo agente.
        """
        if self._is_quick_smalltalk(question):
            return (
                "Olá! Posso te ajudar com dúvidas sobre os documentos internos. "
                "Me diga sua pergunta."
            )

        cached = self._get_cached_answer(session_id, question)
        if cached:
            logger.debug(f"[cache] hit sessão={session_id}")
            return cached

        agent = self.get_agent(session_id)
        response = agent.run(question)
        answer = response.content or "Não foi possível gerar uma resposta."
        self._cache_answer(session_id, question, answer)
        return answer


class MessageOrchestratorAgent:
    """
    Agente 1: orquestrador de mensagens.
    Responsável por classificar intenção, rotear, etiquetar, atualizar
    atributos e decidir IA vs humano.
    """

    def __init__(self, specialist: MecSpecialistAgent, chatwoot: ChatwootClient) -> None:
        self.specialist = specialist
        self.chatwoot = chatwoot
        self._active_teams = parse_csv(TEAM)
        self._active_teams_folded = {team: fold_text(team) for team in self._active_teams}
        self._managed_labels = {
            CHATWOOT_LABEL_IA_ORQUESTRADOR,
            CHATWOOT_LABEL_IA_MEC,
            CHATWOOT_LABEL_HUMANO,
            CHATWOOT_LABEL_IA_FALHA,
        }
        self._human_patterns = [
            r"\bhumano\b",
            r"\batendente\b",
            r"\bespecialista\b",
            r"\bfinanceir\w*\b",  # financeiro, financeira, financeiros …
            r"\bsuporte\b",
            r"\bsupport\b",
            r"falar com (uma )?pessoa",
            r"quero falar com (o|a|um|uma)?\s*(suporte|financeiro|atendente|especialista|equipe|time|humano)",
            r"falar com (o|a|um|uma)?\s*(suporte|financeiro|atendente|especialista|equipe|time)",
            r"suporte humano",
            r"quero falar com .*humano",
            r"encaminhar para .*humano",
            r"encaminh(a|e|ar).*(suporte|financeiro|time|equipe|atendente|especialista|humano)",
            r"me encaminh(a|e).*(suporte|financeiro|time|equipe|humano)",
            r"passar para (o|a)?\s*(suporte|financeiro|time|equipe|humano)",
        ]
        self._human_action_keywords = {
            "falar", "encaminhar", "passar", "transferir", "atender",
            "talk", "speak", "transfer", "escalate",
            "hablar", "transferir", "escalar",
        }
        self._human_target_keywords = {
            "humano", "pessoa", "atendente", "especialista", "equipe", "time", "suporte", "financeir",
            "human", "person", "agent", "team", "support",
            "persona", "agente", "equipo", "soporte",
        }
        self._ai_patterns = [
            r"\bia\b",
            r"intelig[eê]ncia artificial",
            r"quero ajuda da ia",
            r"voltar para ia",
            r"pode ser pela ia",
        ]
        self._mec_keywords = {
            "mec",
            "regimento",
            "resolução",
            "resolucao",
            "tcc",
            "acc",
            "ufpa",
            "fasi",
            "documento",
            "norma",
            "regra",
            "artigo",
            "credito",
            "crédito",
            "carga horaria",
            "carga horária",
        }
        self._smalltalk = {
            "oi",
            "ola",
            "olá",
            "bom dia",
            "boa tarde",
            "boa noite",
            "tudo bem",
            "obrigado",
            "obrigada",
            "valeu",
            "ok",
        }

        self._classifier_agent: Agent | None = None
        if ORCHESTRATOR_USE_LLM_CLASSIFIER:
            team_list = ", ".join(self._active_teams) if self._active_teams else "suporte"
            self._classifier_agent = Agent(
                model=self.specialist.rag.model,
                name="Classificador de intenção",
                search_knowledge=False,
                telemetry=False,
                instructions=(
                    "Você é um classificador de roteamento para atendimento. "
                    "Classifique a mensagem em uma rota: MEC, HUMAN ou DIRECT. "
                    "Use HUMAN quando o usuário pedir pessoa/time/suporte (qualquer idioma). "
                    f"Times disponíveis: {team_list}. "
                    "Se identificar um time específico na mensagem, responda: HUMAN:<nome_do_time> "
                    f"usando EXATAMENTE um dos nomes disponíveis ({team_list}). "
                    "Se não identificar time específico, responda apenas: HUMAN. "
                    "Use DIRECT para smalltalk/saudações/agradecimentos. "
                    "Use MEC para dúvidas acadêmicas, regulatórias e de documentos. "
                    "Exemplos: 'HUMAN:financeiro', 'HUMAN:suporte', 'HUMAN', 'MEC', 'DIRECT'."
                ),
            )

    def _requested_human(self, text: str) -> bool:
        if any(re.search(pattern, text) for pattern in self._human_patterns):
            return True

        folded = fold_text(text)
        has_action = any(keyword in folded for keyword in self._human_action_keywords)
        has_target = any(keyword in folded for keyword in self._human_target_keywords)
        return has_action and has_target

    def _requested_ai(self, text: str) -> bool:
        return any(re.search(pattern, text) for pattern in self._ai_patterns)

    def _is_mec_topic(self, text: str) -> bool:
        return any(keyword in text for keyword in self._mec_keywords)

    def _is_smalltalk(self, text: str) -> bool:
        return text in self._smalltalk

    def _classify_with_llm(self, text: str) -> IntentDecision | None:
        if not self._classifier_agent:
            return None
        try:
            result = self._classifier_agent.run(text)
            value_raw = (result.content or "").strip()
            value = normalize_text(value_raw)

            # Aceita respostas com pontuação/explicação curta, ex: "HUMAN." ou "HUMAN:financeiro"
            if "human" in value:
                # Tenta extrair nome do time: "human:financeiro" ou "human: suporte"
                team_match = re.search(r"human[:\s]+([a-z0-9_-]+)", value)
                extracted_team = team_match.group(1).strip() if team_match else None
                # Valida que o time extraído é um dos times ativos
                if extracted_team and not any(
                    extracted_team in fold_text(t) or fold_text(t) in extracted_team
                    for t in self._active_teams
                ):
                    logger.debug("[llm] time extraído %r não reconhecido, ignorando.", extracted_team)
                    extracted_team = None
                logger.info("[llm_classifier] HUMAN detectado, time=%r", extracted_team)
                return IntentDecision(
                    route="human",
                    reason="llm_classifier",
                    requested_human=True,
                    requested_team=extracted_team,
                )
            if "direct" in value:
                return IntentDecision(route="direct", reason="llm_classifier", requested_ai=False)
            if "mec" in value:
                return IntentDecision(route="mec", reason="llm_classifier", requested_ai=False)

            logger.warning("Classificador LLM retornou valor inesperado: %r", value_raw)
        except Exception as exc:
            logger.warning(f"Falha no classificador LLM do orquestrador: {exc}")
        return None

    def classify_intent(self, message: str, current_labels: set[str]) -> IntentDecision:
        text = normalize_text(message)
        requested_human = self._requested_human(text)
        requested_ai = self._requested_ai(text)

        if requested_human:
            return IntentDecision(
                route="human",
                reason="explicit_human_request",
                requested_human=True,
                requested_ai=requested_ai,
            )

        # Trava em humano apenas quando houve escalonamento real por baixa confiança.
        # Label "humano" isolada não deve bloquear resposta da IA.
        if (
            CHATWOOT_LABEL_HUMANO in current_labels
            and CHATWOOT_LABEL_IA_FALHA in current_labels
            and not requested_ai
        ):
            return IntentDecision(
                route="human",
                reason="conversation_already_human",
                requested_human=False,
                requested_ai=False,
            )

        if requested_ai:
            return IntentDecision(
                route="mec",
                reason="explicit_ai_request",
                requested_human=False,
                requested_ai=True,
            )

        # Classificação dinâmica por LLM (quando habilitada).
        # Mantemos prioridades explícitas acima (pedido humano/IA e lock humano) por segurança.
        llm_decision = self._classify_with_llm(text)
        if llm_decision:
            return llm_decision

        if self._is_smalltalk(text):
            return IntentDecision(
                route="direct",
                reason="smalltalk",
                requested_human=False,
                requested_ai=False,
            )

        if self._is_mec_topic(text):
            return IntentDecision(
                route="mec",
                reason="mec_domain_keyword",
                requested_human=False,
                requested_ai=False,
            )

        return IntentDecision(
            route="mec",
            reason="default_mec_route",
            requested_human=False,
            requested_ai=False,
        )

    def _compose_state_labels(
        self,
        current_labels: set[str],
        target_labels: set[str],
    ) -> list[str]:
        labels = set(current_labels)
        labels.difference_update(self._managed_labels)
        labels.update(target_labels)
        return sorted(labels)

    def _pick_human_team(self, content: str) -> str | None:
        normalized = fold_text(content)

        def _support_team_name() -> str | None:
            for original_name, folded_name in self._active_teams_folded.items():
                if "suporte" in folded_name or "support" in folded_name:
                    return original_name
            return None

        def _name_matches_text(folded_name: str, text: str) -> bool:
            """Aceita nome exato ou formas flexionadas (ex.: 'financeira' → 'financeiro')."""
            if folded_name in text:
                return True
            # Radical sem o último caractere cobre gênero/número (financeiro→financeir)
            if len(folded_name) > 4:
                return bool(re.search(r"\b" + re.escape(folded_name[:-1]), text))
            return False

        # Se o usuário mencionou explicitamente um time ativo, prioriza ele.
        for original_name, folded_name in self._active_teams_folded.items():
            if folded_name and _name_matches_text(folded_name, normalized):
                return original_name

        # Regras contextuais simples (fallback).
        if re.search(r"\bfinanceir", normalized):
            for original_name, folded_name in self._active_teams_folded.items():
                if "financeiro" in folded_name:
                    return original_name
        if re.search(r"\bsuport", normalized):
            for original_name, folded_name in self._active_teams_folded.items():
                if "suporte" in folded_name:
                    return original_name
        if "support" in normalized or "soporte" in normalized:
            for original_name, folded_name in self._active_teams_folded.items():
                if "support" in folded_name or "suporte" in folded_name or "soporte" in folded_name:
                    return original_name

        if "equipe" in normalized or "time" in normalized or "team" in normalized or "equipo" in normalized:
            support_team = _support_team_name()
            if support_team:
                return support_team
            if TEAM_DEFAULT_HUMAN in self._active_teams:
                return TEAM_DEFAULT_HUMAN

        if "mec" in normalized and "suporte" in {fold_text(t) for t in self._active_teams}:
            for original_name, folded_name in self._active_teams_folded.items():
                if folded_name == "suporte":
                    return original_name

        if self._active_teams:
            # Pedido humano sem equipe explícita => padrão Suporte.
            support_team = _support_team_name()
            if support_team:
                return support_team
            if TEAM_DEFAULT_HUMAN in self._active_teams:
                return TEAM_DEFAULT_HUMAN
            return self._active_teams[0]
        if CHATWOOT_HUMAN_TEAM_ID:
            return CHATWOOT_HUMAN_TEAM_ID
        return TEAM_DEFAULT_HUMAN if TEAM_DEFAULT_HUMAN else None

    @staticmethod
    def _direct_answer(message: str) -> str:
        text = normalize_text(message)
        if text in {"oi", "ola", "olá", "bom dia", "boa tarde", "boa noite"}:
            return "Olá! Posso ajudar com dúvidas acadêmicas e regulatórias. Qual sua pergunta?"
        return "Entendi. Posso te ajudar com dúvidas sobre os documentos e regras acadêmicas."

    async def handle_incoming(
        self,
        conversation_id: int,
        account_id: int | str,
        content: str,
        current_labels: list[str],
        force_ia_label: bool = False,
    ) -> None:
        label_set = set(current_labels)
        decision = self.classify_intent(content, label_set)
        session_id = f"chatwoot_{conversation_id}"

        custom_attrs = {
            "orchestrator_route": decision.route,
            "orchestrator_reason": decision.reason,
            "orchestrator_ts": int(time.time()),
            "first_interaction": force_ia_label,
        }
        # Prioriza time extraído pelo LLM; fallback para regex.
        selected_human_team = decision.requested_team or self._pick_human_team(content)
        resolved_human_team_id = await self.chatwoot.resolve_team_id(account_id, selected_human_team)
        if resolved_human_team_id is None and CHATWOOT_HUMAN_TEAM_ID:
            resolved_human_team_id = CHATWOOT_HUMAN_TEAM_ID
        logger.info(
            "[orchestrator] human_route team_selected=%r team_id=%r source=%s",
            selected_human_team,
            resolved_human_team_id,
            "llm" if decision.requested_team else "regex",
        )

        # Rota: atendimento humano.
        if decision.route == "human":
            if decision.reason == "explicit_human_request":
                try:
                    await self.chatwoot.send_message(
                        conversation_id,
                        account_id,
                        "Entendido. Vou encaminhar seu atendimento para o time humano.",
                    )
                except Exception as exc:
                    logger.warning("[human_route] Falha ao enviar mensagem de confirmação: %s", exc)

            try:
                labels = self._compose_state_labels(
                    label_set,
                    target_labels={CHATWOOT_LABEL_HUMANO},
                )
                await self.chatwoot.set_labels(conversation_id, account_id, labels)
            except Exception as exc:
                logger.warning("[human_route] Falha ao atualizar labels: %s", exc)

            try:
                await self.chatwoot.update_conversation_meta(
                    conversation_id,
                    account_id,
                    custom_attributes={
                        **custom_attrs,
                        "handled_by": "human_team",
                        "orchestrator_confidence": 0.0,
                    },
                )
            except Exception as exc:
                logger.warning("[human_route] Falha ao atualizar custom_attributes: %s", exc)

            if resolved_human_team_id:
                try:
                    await self.chatwoot.assign_team(
                        conversation_id, account_id, resolved_human_team_id
                    )
                except Exception as exc:
                    logger.warning("[human_route] Falha ao atribuir time %s: %s", resolved_human_team_id, exc)

            try:
                await self.chatwoot.set_conversation_open(conversation_id, account_id)
            except Exception as exc:
                logger.warning("[human_route] Falha ao abrir conversa: %s", exc)
            return

        # Rota: resposta direta do próprio orquestrador.
        if decision.route == "direct":
            answer = self._direct_answer(content)
            await self.chatwoot.send_message(conversation_id, account_id, answer)
            labels = self._compose_state_labels(
                label_set,
                target_labels={CHATWOOT_LABEL_IA_ORQUESTRADOR},
            )
            await self.chatwoot.set_labels(conversation_id, account_id, labels)
            await self.chatwoot.update_conversation_meta(
                conversation_id,
                account_id,
                custom_attributes={
                    **custom_attrs,
                    "handled_by": "agent_1_orchestrator",
                    "orchestrator_confidence": 0.95,
                },
                clear_assignment=True,
            )
            await self.chatwoot.set_conversation_open(conversation_id, account_id)
            return

        # Rota: especialista MEC (Agente 2)
        specialist_result = await asyncio.to_thread(self.specialist.answer, content, session_id)
        high_confidence = specialist_result.confidence >= ORCHESTRATOR_CONFIDENCE_THRESHOLD

        if high_confidence:
            await self.chatwoot.send_message(conversation_id, account_id, specialist_result.answer)
            labels = self._compose_state_labels(
                label_set,
                target_labels={CHATWOOT_LABEL_IA_MEC},
            )
            await self.chatwoot.set_labels(conversation_id, account_id, labels)
            await self.chatwoot.update_conversation_meta(
                conversation_id,
                account_id,
                custom_attributes={
                    **custom_attrs,
                    "handled_by": "agent_2_mec",
                    "orchestrator_confidence": specialist_result.confidence,
                },
                clear_assignment=True,
            )
            await self.chatwoot.set_conversation_open(conversation_id, account_id)
            return

        # Baixa confiança: escalona para humano.
        await self.chatwoot.send_message(
            conversation_id,
            account_id,
            "Não encontrei segurança suficiente para responder com precisão. "
            "Vou encaminhar para um especialista humano.",
        )
        labels = self._compose_state_labels(
            label_set,
            target_labels={CHATWOOT_LABEL_HUMANO, CHATWOOT_LABEL_IA_FALHA},
        )
        await self.chatwoot.set_labels(conversation_id, account_id, labels)
        await self.chatwoot.update_conversation_meta(
            conversation_id,
            account_id,
            custom_attributes={
                **custom_attrs,
                "handled_by": "human_team_after_low_confidence",
                "orchestrator_confidence": specialist_result.confidence,
            },
            team_id=resolved_human_team_id,
        )
        await self.chatwoot.set_conversation_open(conversation_id, account_id)


# ---------------------------------------------------------------------------
# Estado global (inicializado no startup)
# ---------------------------------------------------------------------------
rag_system: RagSystem
mec_specialist_agent: MecSpecialistAgent
orchestrator_agent: MessageOrchestratorAgent
chatwoot_client: ChatwootClient
_docs_loaded: bool = False
_loading_error: str = ""
_processed_message_ids: dict[int, float] = {}


# ---------------------------------------------------------------------------
# Background: carregamento de documentos
# ---------------------------------------------------------------------------
async def _load_docs_background() -> None:
    """Carrega documentos em background sem bloquear o servidor."""
    global _docs_loaded, _loading_error
    try:
        logger.info("[background] Iniciando carregamento de documentos...")
        await asyncio.to_thread(rag_system.load_documents)
        _docs_loaded = True
        logger.info("[background] ✓ Documentos carregados com sucesso!")
    except Exception as exc:
        _loading_error = str(exc)
        logger.error(f"[background] ✗ Erro ao carregar documentos: {exc}")


# ---------------------------------------------------------------------------
# Lifespan – startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401
    """Inicia o sistema RAG e agenda carregamento de documentos em background."""
    global rag_system, mec_specialist_agent, orchestrator_agent, chatwoot_client
    logger.info("Iniciando o Agente RAG…")
    rag_system = RagSystem()
    chatwoot_client = ChatwootClient(
        base_url=CHATWOOT_API_URL,
        api_token=CHATWOOT_API_TOKEN,
    )
    mec_specialist_agent = MecSpecialistAgent(rag_system)
    orchestrator_agent = MessageOrchestratorAgent(mec_specialist_agent, chatwoot_client)
    # Pré-carrega cache de times para resolução correta de team_id.
    try:
        teams = await chatwoot_client._list_teams(CHATWOOT_ACCOUNT_ID)
        for t in teams:
            name = str(t.get("name") or "").strip()
            tid = t.get("id")
            if isinstance(tid, str) and tid.isdigit():
                tid = int(tid)
            if name and isinstance(tid, int):
                from chatwoot_client import _fold_text
                chatwoot_client._team_cache[name.casefold()] = tid
                chatwoot_client._team_cache[_fold_text(name)] = tid
        logger.info("[startup] Times carregados: %s", {k: v for k, v in chatwoot_client._team_cache.items()})

        # Se TEAM não foi configurado no .env, usa automaticamente os times do Chatwoot.
        if not TEAM and teams:
            api_team_names = [str(t.get("name") or "").strip() for t in teams if t.get("name")]
            orchestrator_agent._active_teams = api_team_names
            orchestrator_agent._active_teams_folded = {
                n: fold_text(n) for n in api_team_names
            }
            logger.info("[startup] TEAM não configurado — times carregados da API: %s", api_team_names)
        else:
            logger.info("[startup] TEAM configurado via .env: %s", orchestrator_agent._active_teams)

        # Reconstrói o prompt do classificador LLM com a lista final de times.
        if orchestrator_agent._classifier_agent and orchestrator_agent._active_teams:
            team_list = ", ".join(orchestrator_agent._active_teams)
            orchestrator_agent._classifier_agent.instructions = (
                "Você é um classificador de roteamento para atendimento. "
                "Classifique a mensagem em uma rota: MEC, HUMAN ou DIRECT. "
                "Use HUMAN quando o usuário pedir pessoa/time/suporte (qualquer idioma). "
                f"Times disponíveis: {team_list}. "
                "Se identificar um time específico na mensagem, responda: HUMAN:<nome_do_time> "
                f"usando EXATAMENTE um dos nomes disponíveis ({team_list}). "
                "Se não identificar time específico, responda apenas: HUMAN. "
                "Use DIRECT para smalltalk/saudações/agradecimentos. "
                "Use MEC para dúvidas acadêmicas, regulatórias e de documentos. "
                "Exemplos: 'HUMAN:financeiro', 'HUMAN:suporte', 'HUMAN', 'MEC', 'DIRECT'."
            )
            logger.info("[startup] Prompt do classificador LLM atualizado com times: %s", team_list)
    except Exception as exc:
        logger.warning("[startup] Não foi possível pré-carregar times: %s", exc)
    # Agenda carregamento em background: servidor fica disponível IMEDIATAMENTE
    asyncio.create_task(_load_docs_background())
    logger.info("Servidor pronto! Documentos sendo carregados em background...")
    yield
    logger.info("Encerrando o Agente RAG.")


# ---------------------------------------------------------------------------
# Aplicação FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Agente RAG – Chatwoot",
    description="Agente RAG que responde mensagens recebidas pelo Chatwoot via webhook.",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Tarefa de fundo – processa mensagem e responde no Chatwoot
# ---------------------------------------------------------------------------
async def process_and_reply(
    conversation_id: int,
    content: str,
    account_id: int | str,
    current_labels: list[str],
    force_ia_label: bool = False,
) -> None:
    """Executa o fluxo hierárquico do orquestrador."""
    try:
        logger.info(f"[conv #{conversation_id}] Mensagem recebida para orquestração.")
        await orchestrator_agent.handle_incoming(
            conversation_id=conversation_id,
            account_id=account_id,
            content=content,
            current_labels=current_labels,
            force_ia_label=force_ia_label,
        )
    except httpx.HTTPStatusError as exc:
        logger.error(f"Erro HTTP na orquestração: {exc.response.status_code} – {exc.response.text}")
    except Exception as exc:
        logger.exception(f"Erro inesperado na conversa #{conversation_id}: {exc}")
        try:
            await chatwoot_client.send_message(
                conversation_id,
                account_id,
                "⚠️ Desculpe, ocorreu um erro ao processar sua mensagem. Por favor, tente novamente.",
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/webhook", summary="Webhook do Chatwoot")
async def chatwoot_webhook(request: Request, background_tasks: BackgroundTasks, token: str = ""):
    """
    Recebe eventos do Chatwoot e processa mensagens recebidas.

    Configurar no Chatwoot em:
    Settings → Integrations → Webhooks → URL:
        https://<ngrok-host>/api/webhook?token=<WEBHOOK_TOKEN>
    Habilitar o evento: **Message Created**
    """
    # ── Validação do token ─────────────────────────────────────────────────
    if WEBHOOK_TOKEN and token != WEBHOOK_TOKEN:
        logger.warning(f"Webhook recusado – token inválido: {token!r}")
        raise HTTPException(status_code=403, detail="Token inválido.")

    # ── Leitura do payload ─────────────────────────────────────────────────
    try:
        payload: dict = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Payload JSON inválido.")

    # Log completo do payload (nível DEBUG – visível com LOG_LEVEL=DEBUG)
    logger.debug(f"Payload recebido:\n{payload}")

    event: str = payload.get("event", "")
    message_type: str = payload.get("message_type", "")
    is_private: bool = payload.get("private", False)
    message_id: int | None = payload.get("id")

    logger.info(
        f"Webhook recebido – event={event!r}  "
        f"message_type={message_type!r}  private={is_private}"
    )

    # Loga payload completo para qualquer event=message_created (facilita debug)
    if event == "message_created":
        import json as _json
        logger.info(f"[message_created] payload completo:\n{_json.dumps(payload, indent=2, default=str)}")

    # ── Filtragem ──────────────────────────────────────────────────────────
    # Processa apenas mensagens recebidas de contatos (evita loop com o bot).
    # Nota: message_type="incoming" já garante que é mensagem do contato.
    # O campo sender.type nem sempre existe no payload de topo do Chatwoot.
    if (
        event == "message_created"
        and message_type == "incoming"
        and not is_private
    ):
        # Evita reprocessamento quando Chatwoot reenvia o mesmo evento.
        if message_id is not None:
            now = time.time()
            for old_id, ts in list(_processed_message_ids.items()):
                if (now - ts) > RESPONSE_CACHE_TTL_SECONDS:
                    _processed_message_ids.pop(old_id, None)
            if message_id in _processed_message_ids:
                logger.info(f"Mensagem duplicada ignorada (id={message_id})")
                return JSONResponse({"status": "ok", "dedup": True})
            _processed_message_ids[message_id] = now

        raw_content: str = payload.get("content") or ""
        # Remove HTML que o Chatwoot às vezes envia (ex.: "<p>Bom dia</p>")
        content: str = strip_html(raw_content)

        conversation: dict = payload.get("conversation") or {}
        # Tenta pegar o id da conversa de diferentes chaves
        conversation_id: int | None = (
            conversation.get("id")
            or payload.get("conversation_id")
        )
        current_labels: list[str] = conversation.get("labels") or []
        # Primeira interação: primeira mensagem do contato na conversa.
        force_ia_label: bool = (
            conversation.get("first_reply_created_at") in (None, "")
            and message_type == "incoming"
        )
        account: dict = payload.get("account") or {}
        account_id: int | str = account.get("id") or CHATWOOT_ACCOUNT_ID

        # Informações do sender apenas para log
        sender: dict = payload.get("sender") or {}
        sender_name: str = sender.get("name", "?")

        if content and conversation_id and account_id:
            logger.info(
                f"Mensagem enfileirada – conv #{conversation_id} "
                f"| sender={sender_name!r} "
                f"| conteúdo={content[:80]!r}"
            )
            background_tasks.add_task(
                process_and_reply,
                conversation_id,
                content,
                account_id,
                current_labels,
                force_ia_label,
            )
        else:
            logger.debug(
                f"Mensagem ignorada – conv={conversation_id} "
                f"content_raw={raw_content[:60]!r}"
            )

    return JSONResponse({"status": "ok"})


@app.get("/health", summary="Health check")
async def health_check():
    """Verifica se o serviço está no ar e o status do carregamento dos documentos."""
    return {
        "status": "healthy",
        "service": "Agente RAG Chatwoot",
        "docs_loaded": _docs_loaded,
        "loading_error": _loading_error or None,
        "docs_folder": DOCS_FOLDER,
        "chatwoot_url": CHATWOOT_API_URL,
    }


@app.get("/teams", summary="Listar times do Chatwoot")
async def list_teams():
    """Lista os times disponíveis no Chatwoot com seus IDs reais e o cache atual."""
    try:
        teams = await chatwoot_client._list_teams(CHATWOOT_ACCOUNT_ID)
        return {
            "teams": [
                {"id": t.get("id"), "name": t.get("name"), "description": t.get("description")}
                for t in teams
            ],
            "team_cache": dict(chatwoot_client._team_cache),
            "env_TEAM": TEAM,
            "env_TEAM_DEFAULT_HUMAN": TEAM_DEFAULT_HUMAN,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/reload-docs", summary="Recarregar documentos")
async def reload_documents(recreate: bool = False):
    """
    Recarrega os arquivos .md da pasta Docs na base de conhecimento.

    - `recreate=false` (padrão): insere apenas documentos novos.
    - `recreate=true`: limpa toda a base e recarrega tudo.
    """
    try:
        await asyncio.to_thread(rag_system.load_documents, recreate)
        return {"status": "success", "message": "Documentos recarregados com sucesso."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("Agente:app", host="0.0.0.0", port=8000, reload=True)
