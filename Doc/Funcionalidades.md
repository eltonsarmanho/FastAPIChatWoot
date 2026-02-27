# Funcionalidades — Agente RAG Chatwoot

## 🎯 Visão Geral

Sistema inteligente de orquestração de mensagens que integra um agente RAG (Retrieval-Augmented Generation) com o Chatwoot. Classifica automaticamente intenções de usuários e roteia para o atendimento apropriado: IA (especialista MEC), resposta direta (orquestrador) ou atendimento humano.

---

## 🔄 Arquitetura de Agentes

### **Agente 1: Orquestrador** (MessageOrchestratorAgent)
- Recebe mensagens do Chatwoot via webhook
- Classifica a intenção:
  - **HUMAN:** usuário solicita atendimento com pessoa/time humano
  - **MEC:** dúvida sobre documentos internos
  - **DIRECT:** smalltalk/saudações respondidas diretamente pelo orquestrador
- Roteia para o agente apropriado ou atribuição de time
- Gerencia rótulos e atributos customizados da conversa

### **Agente 2: Especialista MEC** (MecSpecialistAgent)
- Responde perguntas sobre documentos internos via RAG
- Retorna resposta + nível de confiança (0.0-1.0)
- Se confiança ≥ limiar: envia resposta diretamente
- Se confiança < limiar: escalona para atendimento humano

### **Sistema RAG** (RagSystem)
- Gerencia base de conhecimento vetorial (LanceDb)
- Carrega documentos `.md` da pasta `Docs/`
- Mantém agentes e cache de respostas por sessão

---

## 📋 Fluxo de Processamento

```
1. Webhook Chatwoot
   ↓
2. Validação (token, formato, deduplicação)
   ↓
3. Classificação de Intenção
   ├─ Padrões explícitos (regex + keywords)
   ├─ Classificador LLM (se habilitado)
   └─ Fallback: smalltalk / domínio MEC / padrão
   ↓
4. Decisão de Rota
   ├─ HUMAN → Atribui time + etiqueta "humano"
   ├─ DIRECT → Resposta do orquestrador
   └─ MEC → Envia ao especialista
   ↓
5. Atualização Chatwoot
   ├─ Envio de mensagem
   ├─ Atualização de etiquetas
   ├─ Atribuição de time (se HUMAN)
   └─ Atributos customizados
```

---

## 🧠 Classificação de Intenção

### **1. Detecção Explícita de Humano**

Detecta automaticamente pedidos como:
- "Quero falar com humano"
- "Quero falar com a equipe de financeiro"
- "Me encaminhe para suporte"


**Funciona com:**
- Formas flexionadas: "financeira", "financeiras", "financeiros" → mapeiam para time "financeiro"
- Nomes de times: "suporte", "financeiro" (extratos do Chatwoot)
- Múltiplos idiomas: português

### **2. Classificador LLM Dinâmico** (opcional)

Se `ORCHESTRATOR_USE_LLM_CLASSIFIER=true`:
- LLM recebe a mensagem e lista de times disponíveis
- Retorna: `HUMAN:teamname` ou `HUMAN` ou `MEC` ou `DIRECT`
- Extrai nome do time automaticamente (ex: "HUMAN:financeiro")
- **Prioridade:** LLM sobre regex (mais preciso em linguagem natural)

### **3. Fallback a Regex + Keywords**

Se nenhum dos acima, testa:
- Palavras-chave de ação: "falar", "encaminhar", "transferir"
- Palavras-chave de alvo: "humano", "equipe", "time", "suporte"
- Domínio MEC: "regimento", "resolução", "tcc", "documento", "crédito"
- Smalltalk: "oi", "obrigado", "tudo bem"

---

## 👥 Roteamento de Times

### **Descoberta Automática**

1. **Startup:** Sistema carrega times do Chatwoot automaticamente
2. **Cache:** Armazena mapeamento nome → ID para visualização rápida
3. **Fallback:** Se `TEAM` vazio no `.env`, usa todos os times do Chatwoot

### **Seleção de Time**

**Prioridade:**
1. Time extraído pelo LLM (se `ORCHESTRATOR_USE_LLM_CLASSIFIER=true`)
2. Time mencionado explicitamente na mensagem (regex matching)
3. Regras contextuais: "financeiro" → time financeiro, "suporte" → time suporte
4. Padrão: `TEAM_DEFAULT_HUMAN` (suporte)

### **Configuração `.env`** (opcional)

```env
TEAM=suporte,financeiro
TEAM_DEFAULT_HUMAN=suporte
```

- Se deixar `TEAM=` vazio → carrega automaticamente da API
- Se configurar → filtra apenas esses times
- Prompt do LLM é reconstruído dinamicamente no startup

---

## 🏷️ Etiquetagem

Gerencia automaticamente:

| Etiqueta | Significado |
|---|---|
| `humano` | Conversa atribuída a atendente humano |
| `ia_orquestrador` | Respondida pelo agente orquestrador (smalltalk) |
| `ia_mec` | Respondida pelo especialista (alta confiança) |
| `ia_falha` | Escalada para humano (baixa confiança) |

**Comportamento:**
- Remove etiquetas gerenciadas antes de atualizar
- Preserva etiquetas customizadas do usuário
- Atualiza via endpoint dedicado `POST /labels`

---

## 🔗 Integração Chatwoot

### **Webhook**

```
POST /api/webhook?token=<WEBHOOK_TOKEN>
```

- Validação de token obrigatória
- Filtra apenas mensagens recebidas (`message_type=incoming`)
- Ignora mensagens privadas
- Deduplicação por ID de mensagem

### **Operações na Conversa**

| Operação | Endpoint | Uso |
|---|---|---|
| Enviar mensagem | `POST /messages` | Resposta e confirmação |
| Atualizar etiquetas | `POST /labels` | Classificação |
| Atribuir time | `POST /assignments` | Roteamento humano |
| Atualizar meta | `PATCH /conversations/{id}` | Atributos customizados |
| Abrir conversa | `PATCH /conversations/{id}` | Status open |

**Resiliência:**
- Cada operação em `try/except` independente
- Falha em uma não bloqueia as outras
- Fallbacks para endpoints alternativos (ex: `/labels` → `/conversations`)

---

## 📊 Confiança e Escalação

### **Limiar de Confiança**

```env
ORCHESTRATOR_CONFIDENCE_THRESHOLD=0.7
```

- MEC responde se confiança ≥ 0.7
- Se < 0.7 → escala para humano com etiqueta `ia_falha`

### **Respostas**

- **Confiança alta (MEC):** Resposta técnica baseada em documentos
- **Confiança baixa:** Mensagem de escalonamento + atribuição humana
- **Erro do sistema:** Aviso ao usuário + log técnico

---

## 🎯 Cache e Performance

### **Cache de Respostas**

```env
RESPONSE_CACHE_TTL_SECONDS=300
RESPONSE_CACHE_MAX_ITEMS=256
```

- Evita reprocessamento de perguntas idênticas
- TTL (Time-To-Live): válido por 5 minutos
- Limite de tamanho: 256 respostas em memória

### **Cache de Times**

- Carregado no startup
- Atualizado quando necessário resolver novo time
- Lookup O(1) para mapeamento nome → ID

---

## ⚙️ Configuração

### **Variáveis Essenciais**

```env
# Maritaca AI (LLM)
MARITALK_API_KEY=...

# Chatwoot
CHATWOOT_API_URL=http://localhost:3000
CHATWOOT_API_TOKEN=...
CHATWOOT_ACCOUNT_ID=3

# Webhook
WEBHOOK_TOKEN=abc123

# Times (opcional - carrega da API se vazio)
TEAM=
TEAM_DEFAULT_HUMAN=suporte

# Classificador LLM
ORCHESTRATOR_USE_LLM_CLASSIFIER=true
ORCHESTRATOR_CONFIDENCE_THRESHOLD=0.7

# RAG
DOCS_FOLDER=Docs
LANCEDB_URI=lancedb
RAG_MAX_DOCS=5

# Logging
LOG_LEVEL=INFO
```

---

## 🚀 Endpoints

### **Health Check**
```
GET /health
```
Status do serviço e carregamento de documentos.

### **Listar Times**
```
GET /teams
```
Times do Chatwoot + cache + configuração `.env`.

### **Recarregar Documentos**
```
POST /reload-docs?recreate=false
```
- `recreate=false`: insere apenas novos
- `recreate=true`: limpa e recarrega tudo

### **Webhook**
```
POST /api/webhook?token=<WEBHOOK_TOKEN>
```
Recebe mensagens do Chatwoot.

---

## 📝 Logging

Todos os eventos são registrados com contexto:

```
[orchestrator] human_route team_selected='financeiro' team_id=2 source=llm
[llm_classifier] HUMAN detectado, time='financeiro'
[assign_team] time_id=2 atribuído à conversa 27
[background] ✓ Documentos carregados com sucesso!
```

Nível configurável via `LOG_LEVEL` (DEBUG, INFO, WARNING, ERROR).

---

## 🔄 Fluxo Exemplo

**Mensagem:** "Quero falar com a equipe financeira"

```
1. Webhook recebe a mensagem
2. Classificação explícita detecta "falar com" + "equipe" + "financeira"
3. Padrão regex identifica como pedido HUMAN
4. _pick_human_team() testa formas flexionadas → encontra "financeiro"
5. resolve_team_id('financeiro') → team_id=2 (do cache)
6. set_labels() → adiciona "humano"
7. assign_team(team_id=2) → atribui à equipe financeira
8. Mensagem: "Entendido. Vou encaminhar seu atendimento para o time humano."
9. Conversa abre com team=financeiro e etiqueta=humano
```

---

## 📚 Documentos Suportados

Sistema RAG carrega automaticamente arquivos `.md` de `Docs/`:

- `Regimento_Interno_Docling.md` — Regras e normas acadêmicas
- `Resolução ACC FASI 2024_Docling.md` — Resoluções oficiais

Novos documentos podem ser adicionados e recarregados via `POST /reload-docs`.
