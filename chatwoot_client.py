"""
Cliente Chatwoot API
====================

Módulo para interação com a API REST do Chatwoot.
Responsável por enviar mensagens, atualizar rótulos e
gerenciar atributos de conversas.
"""

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger("chatwoot_client")


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
        url = f"/api/v1/accounts/{account_id}/conversations/{conversation_id}"
        payload = {"labels": labels}
        response = await self.client.patch(url, json=payload)
        response.raise_for_status()
        return response.json()

    async def update_conversation_meta(
        self,
        conversation_id: int,
        account_id: int | str,
        custom_attributes: Optional[dict[str, Any]] = None,
        team_id: Optional[str] = None,
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
