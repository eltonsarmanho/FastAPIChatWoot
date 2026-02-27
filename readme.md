# Agente RAG FastAPI + Chatwoot

Sistema RAG (Retrieval-Augmented Generation) integrado com Chatwoot para atendimento automático via inteligência artificial.

## Características

- 🤖 **RAG com LanceDB**: Base de conhecimento vetorial para busca semântica
- 🔀 **Orquestrador de Mensagens**: Roteamento inteligente entre IA e atendimento humano
- 📚 **Suporte a Documentos MD**: Carrega automaticamente documentos Markdown
- 🔗 **Integração Chatwoot**: Webhook para recebimento de mensagens
- 🏷️ **Sistema de Labels**: Rastreamento automático de conversas
- ⚡ **Cache Inteligente**: Respostas cacheadas para melhor performance
- 📊 **Atributos Customizados**: Tracking de confiança e roteamento

## Arquitetura

### Agentes

1. **Agente 1 - Orquestrador** (`MessageOrchestratorAgent`)
   - Classifica intenção das mensagens
   - Roteia para IA ou atendimento humano
   - Gerencia labels e atributos customizados

2. **Agente 2 - Especialista MEC** (`MecSpecialistAgent`)
   - Responde perguntas sobre regulamentos
   - Retorna confiança da resposta
   - Escala para humano em caso de baixa confiança

### Sistema RAG

- Utiliza **LanceDB** para vetorização
- **SentenceTransformer** como embedder
- **Maritaca Sabiá** como modelo LLM
- Cache de respostas com TTL configurável

## Instalação

```bash
pip install -r requirement.txt
```

## Configuração

Configure as variáveis de ambiente no arquivo `.env`:

```env
MARITALK_API_KEY=seu_token_aqui
CHATWOOT_API_URL=http://localhost:3000
CHATWOOT_API_TOKEN=seu_token_aqui
CHATWOOT_ACCOUNT_ID=1
WEBHOOK_TOKEN=seu_token_webhook
DOCS_FOLDER=Docs
LOG_LEVEL=INFO
```

## Execução

```bash
uvicorn OrquestradorAPI:app --host 0.0.0.0 --port 8000
```

O servidor estará disponível em `http://localhost:8000`

### Health Check

```bash
curl http://localhost:8000/health
```

### Recarregar Documentos

```bash
curl -X POST http://localhost:8000/reload-docs
```

## Estrutura de Diretórios

```
.
├── OrquestradorAPI.py       # Orquestrador principal (FastAPI + webhook)
├── Agente2.py               # Agente especialista MEC + Sistema RAG
├── ClassificadorIntencao.py  # Classificador NLU de intenções (HuggingFace)
├── chatwoot_client.py       # Cliente API Chatwoot
├── Docs/                    # Documentos Markdown
│   ├── Regimento_Interno_Docling.md
│   └── Resolução ACC FASI 2024_Docling.md
├── Diagrama/
│   └── Arquitetura.wsd      # Diagrama da arquitetura
├── lancedb/                 # Base de conhecimento vetorial
├── .env                     # Variáveis de ambiente
└── requirement.txt          # Dependências
```

## Fluxo de Processamento

1. **Webhook**: Chatwoot envia evento de nova mensagem
2. **Validação**: Token verificado
3. **Classificação**: Orquestrador classifica intenção
4. **Roteamento**:
   - Small talk → resposta direta
   - Domínio MEC → especialista
   - Pedido explícito → escalada
5. **Resposta**: Mensagem enviada de volta para Chatwoot

## Roteamento

### Rotas Disponíveis

- **DIRECT**: Respostas rápidas (saudações, etc)
- **MEC**: Perguntas sobre regulamentos (especialista)
- **HUMAN**: Escalada para atendimento humano

### Critérios de Escalada

- Confiança < 70%
- Pedido explícito do usuário
- Padrões de "falar com humano"

## Labels Automáticos

- `ia_orquestrador`: Resposta do orquestrador
- `ia_mec`: Resposta do especialista
- `humano`: Escalada para humano
- `ia_falha`: Baixa confiança

## Desenvolvimento

### Dependências Principais

- `fastapi`: Framework web
- `agno`: Framework de agentes IA
- `lancedb`: Vector database
- `httpx`: Client HTTP assíncrono
- `python-dotenv`: Gerenciamento de variáveis de ambiente

### Logging

Controle o nível de logging com `LOG_LEVEL`:

```env
LOG_LEVEL=DEBUG  # Mais verboso
LOG_LEVEL=INFO   # Padrão
LOG_LEVEL=ERROR  # Apenas erros
```

## API Endpoints

### POST /api/webhook

Recebe eventos do Chatwoot (configurar em Settings → Integrations → Webhooks)

**Query Parameters:**
- `token`: Token de autenticação do webhook

**Evento esperado:** `message_created` com `message_type=incoming`

### GET /health

Status da aplicação e carregamento de documentos

**Resposta:**
```json
{
  "status": "healthy",
  "docs_loaded": true,
  "loading_error": null
}
```

### POST /reload-docs

Recarrega documentos da pasta Docs

**Query Parameters:**
- `recreate` (bool, default=false): Limpar e recriar base

## Monitoramento

Logs são salvos em `agente.log` (se configurado) e exibidos no console.

Atributos customizados rastreiam:
- `orchestrator_route`: Rota tomada
- `orchestrator_reason`: Motivo da rota
- `orchestrator_confidence`: Confiança da resposta
- `handled_by`: Qual agente processou
