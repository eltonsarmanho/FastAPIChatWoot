"""
Cliente Chatwoot API
====================

Módulo para interação com a API REST do Chatwoot.
Responsável por enviar mensagens, atualizar rótulos e
gerenciar atributos de conversas.
"""

import logging
import unicodedata
from typing import Any, Optional

import httpx

logger = logging.getLogger("chatwoot_client")


def _fold_text(value: str) -> str:
    lowered = " ".join((value or "").strip().lower().split())
    return "".join(
        c for c in unicodedata.normalize("NFD", lowered) if unicodedata.category(c) != "Mn"
    )


class ChatwootClient:
    """Cliente para interagir com a API do Chatwoot."""

    def __init__(self, base_url: str, api_token: str):
        """
        Inicializa o cliente Chatwoot.

        Args:
            base_url: URL base do Chatwoot (ex: http://localhost:3000)
            api_token: Token de autenticação da API
        """
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"api_access_token": api_token},
            timeout=30.0,
        )
        self._team_cache: dict[str, int] = {}

    async def _list_teams(self, account_id: int | str) -> list[dict[str, Any]]:
        """Lista os times disponíveis na conta."""
        url = f"/api/v1/accounts/{account_id}/teams"
        response = await self.client.get(url)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            # Compatibilidade com diferentes formatos de resposta.
            if isinstance(data.get("payload"), list):
                return data["payload"]
            if isinstance(data.get("data"), list):
                return data["data"]
        if isinstance(data, list):
            return data
        return []

    async def resolve_team_id(
        self,
        account_id: int | str,
        team_name_or_id: Optional[str],
    ) -> Optional[int | str]:
        """
        Resolve nome de time para ID.

        - Se já vier numérico, retorna como inteiro.
        - Se vier nome, busca em cache e depois na API.
        """
        if not team_name_or_id:
            return None

        value = str(team_name_or_id).strip()
        if not value:
            return None

        if value.isdigit():
            return int(value)

        folded = value.casefold()
        cached = self._team_cache.get(folded)
        if cached is not None:
            return cached

        try:
            teams = await self._list_teams(account_id)
            query_folded = _fold_text(value)
            best_match: int | None = None
            for team in teams:
                name = str(team.get("name") or "").strip()
                team_id = team.get("id")

                # Chatwoot pode retornar id como int ou str.
                resolved_id: int | None = None
                if isinstance(team_id, int):
                    resolved_id = team_id
                elif isinstance(team_id, str) and team_id.isdigit():
                    resolved_id = int(team_id)

                if not name or resolved_id is None:
                    continue

                team_name_folded = _fold_text(name)
                self._team_cache[name.casefold()] = resolved_id
                self._team_cache[team_name_folded] = resolved_id

                if team_name_folded == query_folded:
                    return resolved_id

                # Match parcial para frases como "equipe de financeiro".
                if query_folded in team_name_folded or team_name_folded in query_folded:
                    best_match = best_match or resolved_id

            if best_match is not None:
                return best_match

            return self._team_cache.get(folded) or self._team_cache.get(query_folded)
        except Exception as exc:
            logger.warning("Não foi possível resolver team_id para '%s': %s", value, exc)
            return None

    async def send_message(
        self,
        conversation_id: int,
        account_id: int | str,
        content: str,
        message_type: str = "outgoing",
    ) -> dict[str, Any]:
        """
        Envia uma mensagem para uma conversa no Chatwoot.

        Args:
            conversation_id: ID da conversa
            account_id: ID da conta
            content: Conteúdo da mensagem
            message_type: Tipo de mensagem (outgoing, incoming, etc)

        Returns:
            Resposta da API
        """
        url = f"/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"
        payload = {
            "content": content,
            "message_type": message_type,
        }
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    async def set_labels(
        self,
        conversation_id: int,
        account_id: int | str,
        labels: list[str],
    ) -> dict[str, Any]:
        """
        Define os rótulos de uma conversa.

        Args:
            conversation_id: ID da conversa
            account_id: ID da conta
            labels: Lista de rótulos

        Returns:
            Resposta da API
        """
        payload = {"labels": labels}

        # Endpoint oficial de labels do Chatwoot.
        labels_url = f"/api/v1/accounts/{account_id}/conversations/{conversation_id}/labels"
        response = await self.client.post(labels_url, json=payload)

        # Fallback para versões/instâncias que aceitam labels via PATCH na conversa.
        if response.status_code >= 400:
            logger.warning(
                "Falha ao atualizar labels via /labels (status=%s). Tentando fallback PATCH.",
                response.status_code,
            )
            conversation_url = f"/api/v1/accounts/{account_id}/conversations/{conversation_id}"
            response = await self.client.patch(conversation_url, json=payload)

        if response.status_code >= 400:
            logger.error(
                "Falha ao atualizar labels (status=%s): %s",
                response.status_code,
                response.text[:200],
            )
            return {"error": response.status_code}

        return response.json()

    async def assign_team(
        self,
        conversation_id: int,
        account_id: int | str,
        team_id: int | str,
    ) -> dict[str, Any]:
        """
        Atribui uma conversa a um time específico via endpoint dedicado.

        Args:
            conversation_id: ID da conversa
            account_id: ID da conta
            team_id: ID do time

        Returns:
            Resposta da API
        """
        url = f"/api/v1/accounts/{account_id}/conversations/{conversation_id}/assignments"
        payload = {"team_id": team_id}
        response = await self.client.post(url, json=payload)

        if response.status_code >= 400:
            # Fallback: PATCH direto na conversa
            logger.warning(
                "[assign_team] /assignments falhou (status=%s), tentando PATCH. body=%s",
                response.status_code, response.text[:200],
            )
            conv_url = f"/api/v1/accounts/{account_id}/conversations/{conversation_id}"
            response = await self.client.patch(conv_url, json={"team_id": team_id})

        if response.status_code >= 400:
            logger.error(
                "[assign_team] Falha ao atribuir time %s à conversa %s (status=%s): %s",
                team_id, conversation_id, response.status_code, response.text[:200],
            )
            return {"error": response.status_code}

        logger.info("[assign_team] time_id=%s atribuído à conversa %s", team_id, conversation_id)
        return response.json()

    async def update_conversation_meta(
        self,
        conversation_id: int,
        account_id: int | str,
        custom_attributes: Optional[dict[str, Any]] = None,
        team_id: Optional[int | str] = None,
        clear_assignment: bool = False,
    ) -> dict[str, Any]:
        """
        Atualiza metadados da conversa (atributos customizados, time, etc).

        Args:
            conversation_id: ID da conversa
            account_id: ID da conta
            custom_attributes: Atributos customizados
            team_id: ID do time para atribuir
            clear_assignment: Se True, remove atribuição atual

        Returns:
            Resposta da API
        """
        url = f"/api/v1/accounts/{account_id}/conversations/{conversation_id}"
        payload = {}

        if custom_attributes:
            payload["custom_attributes"] = custom_attributes

        if team_id:
            payload["team_id"] = team_id

        if clear_assignment:
            payload["assignee_id"] = None

        response = await self.client.patch(url, json=payload)
        response.raise_for_status()
        return response.json()

    async def set_conversation_open(
        self,
        conversation_id: int,
        account_id: int | str,
    ) -> dict[str, Any]:
        """
        Define uma conversa como aberta.

        Args:
            conversation_id: ID da conversa
            account_id: ID da conta

        Returns:
            Resposta da API
        """
        url = f"/api/v1/accounts/{account_id}/conversations/{conversation_id}"
        payload = {"status": "open"}
        response = await self.client.patch(url, json=payload)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """Fecha a conexão do cliente."""
        await self.client.aclose()
