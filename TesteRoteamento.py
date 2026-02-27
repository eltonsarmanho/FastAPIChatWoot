
"""
Classificador de Intenção baseado em Embeddings Semânticos
Modelo gratuito: all-MiniLM-L6-v2 (multilíngue, rápido, ~22MB)
"""

import os
import logging
from dotenv import load_dotenv

# Suprime logs verbosos das bibliotecas (LOAD REPORT, UNEXPECTED keys, etc.)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("mlx_lm").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
# Silencia qualquer logger nao identificado que ainda emita o LOAD REPORT
logging.basicConfig(level=logging.WARNING)

from sentence_transformers import SentenceTransformer, util
import numpy as np

# Carrega variáveis do .env (incluindo HF_TOKEN)
load_dotenv(override=True)

# Configura HF_TOKEN automaticamente
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# Modelo gratuito multilíngue compacto
# O LOAD REPORT é emitido por código C interno do MLX direto no fd do SO.
# os.dup2 é a única forma de suprimi-lo no nível do sistema operacional.
import sys
_devnull = open(os.devnull, "w")
_fd1_bkp = os.dup(1)   # backup stdout fd
_fd2_bkp = os.dup(2)   # backup stderr fd
os.dup2(_devnull.fileno(), 1)
os.dup2(_devnull.fileno(), 2)
model = SentenceTransformer('all-MiniLM-L6-v2')
os.dup2(_fd1_bkp, 1)    # restaura stdout
os.dup2(_fd2_bkp, 2)    # restaura stderr
os.close(_fd1_bkp)
os.close(_fd2_bkp)
_devnull.close()

# Exemplos de cada intenção (português)
intent_examples = {
    "HUMAN": [
        "Quero falar com suporte",
        "Quero falar com um humano",
        "Me encaminhe para a equipe de suporte técnico",
        "Preciso falar com um atendente",
        "Quero conversar com o time de especialistas",
        "Gostaria de ser atendido por uma pessoa",
        "Quero contato com o suporte humano do MEC",
        "Preciso de atendimento pessoal com o consultor",
        "Me transfira para um agente humano",
        "Quero falar com alguém da equipe",
        "meu caso é muito especifico e a documentação não ta ajudando",
        "nossa sistema horrivel, não consigo mandar essa planilha de jeito nenhum",
        "ja tentei de tudo q falaram no manual e o erro de entidade continua dando na linha 40",
        "isso q vc explicou nao resolve minha pendencia, continuo travado no simec",
        "a situacao do meu municipio é diferente, pq a gnt dividiu as turmas de um jeito que a planilha nao aceita",
        "to irritado c essa plataforma dando timeout direto, quem pode resolver essa falha?",
        "olha, vc ja me respondeu q tem q enviar no sgp mas a api la de integracao responde com erro 500 faz dias",
        "nao entendi nd q ta escrito nesses guias ou manuais",
        "preciso q avaliem uma situacao atipica da minha rede q não ta nesses modelos padroes",
        "a matricula sumiu e nenhuma orientaçao sua se encaixa no q aconteceu aqui na prefeitura",
        "minha integracao ta quebrando por causa daquela autenticaçao de token q nao disseram como faz",
        "estou totalmente insatisfeito com essa explicaçao generica sobre os professores",
        "tem um bug grave na aba de profissionais qndo tento editar os vinculos de lotacao",
        "vcs apagaram o historico de lote enviado, o prob e no servidor central de vcs!",
        "cara eu acho q o meu municipio foi bloqueado indevidamente e a faq nao diz oq fzer pra me desbloquear",
        "o erro na minha api ta muito complexo, me diz q nao foi vcs q mudaram as rotas hj!!",
        "nenhuma faq fala de qndo o inep ta com divergencia entre as tabelas de cpf, oq eu faço agora?",
        "o sistema simplesmente recusa as minhas escolas mesmo tudo estando formatado perfeitamente",
        "sua resposta de robo nao serve pq meu prob e de permissao de usuario avançado no portal",
        "ja abri chamado e niguem resolve o defeito no gerador de oficios da nossa adessao de hj a tarde"
    ],
    "MEC": [
        "Quem pode aderir à Plataforma MEC Gestão Presente?",
        "A plataforma MEC Gestão Presente é gratuita?",
        "Qual é a relação entre o SGP e o GPE?",
        "Como incluir novos operadores no sistema?",
        "Qual é o fluxo de solicitação da chave da API?",
        "A inserção de dados no GPE será via planilhas como no Pé de Meia?",
        "Como cadastrar um profissional com mais de um contrato/vínculo?",
        "Quais as prioridades de envios de dados para municípios no piloto?",
        "O que é o Conjunto Mínimo de Dados da Educação Básica (CMDEB)?",
        "O que fazer quando não temos o nome da mãe na planilha?",
        "Como preencher a forma de organização da turma no SGP?",
        "A chave da API é atrelada ao operador ou a Secretaria?",
        "Posso adaptar algumas colunas da planilha para o modelo que estou acostumado?",
        "Os CPFS precisam ter exatamente onze dígitos, mesmo que comecem com zero?",
        "Como eu faço a adesão da minha secretaria na Plataforma MEC Gestão Presente?",
        "qual a difrença do SGP pro módulo GPE na escola?",
        "preciso entender o que é esse CMDEB q pediram",
        "tem cm enviar os dados do ensino medio do Pé-de-Meia agora ou espero?",
        "como cadastro um profissional q tem 2 vinculos efetivos na rede municipal?",
        "o gestao presente é de graça msm ou tem algum custo escondido pra prefeitura q nao vi?",
        "onde vejo o modelo de oficio pra solicitar a chave de integracao da api e ql o email?",
        "posso mudar as colunas da planilha do SGP pra ficar igual ao meus sistema proprietario?",
        "oq eu coloco rpa mandar qnd o aluno não tem cpf e nem nis q perdi no censo?",
        "os dados dos alunos q mandamos na hr do pé de meia servem pro sgp tb tb automaticamente?",
        "cara, como faço dpeois pra exclui um operador que saiu da nossa secretaria aqui?",
        "como que exatamente as leis protegem os dados sensiveis das criancas no sgp? lgpd...",
        "modulo gpe vai ser liberado pra todos os municipios de uma vez no piloto de vcs?",
        "quais sao as prioridades e a ordem q preciso pra os envios de dados pros estados agora na janela?",
        "eu tenho q fazer adesao dnv msm ja estando no pé de meia dpsq saiu a portaria nova 234?",
        "O SGP qnto tempo vai ter integracao direta com o portal do Educacenso do Inep esse ano?",
        "nao sei como nem oq eu deveria preencher na tal da forma de organizacao da turma do aluno no form",
        "olha, o cpf precisa ter 11 digitos mesmo qndo comeca com zero? a pranilnha do excek ta recusando...",
        "achei q errei os dados do estudante de terca, mando arquivo da planilha de td mundo dnovo ou so desse um?",
        "onde q vai ver e como eu arrumo se a minha unica escola da zona rural nao aparece no sgp de jeito nenhum listada?"
    ],
    "DIRECT": [
        "Oi, tudo bem?",
        "Bom dia!",
        "Obrigado",
        "Valeu",
        "Tudo certo",
        "Olá!",
        "Boa tarde!",
        "Como vai?",
        "Até mais",
        "Tchau",
        "Boa noite!",
        "Agradeço pela ajuda"  
    ],
}

# Codifica exemplos de cada intenção
intent_embeddings = {}
intent_centroids = {}

for intent, examples in intent_examples.items():
    embeddings = model.encode(examples)
    intent_embeddings[intent] = embeddings
    # Calcula centróide (média dos embeddings) para cada intenção
    intent_centroids[intent] = np.mean(embeddings, axis=0)

print("✓ Modelos de intenção carregados\n")


def classify_intent(message: str, threshold: float = 0.5) -> tuple[str, float]:
    """
    Classifica mensagem em uma das 3 intenções usando similaridade semântica.
    
    Args:
        message: Texto da mensagem
        threshold: Score mínimo para classificação (0-1)
    
    Returns:
        (intenção, confiança)
    """
    query_embedding = model.encode(message)
    
    scores = {}
    for intent, centroid in intent_centroids.items():
        similarity = util.cos_sim(query_embedding, centroid)[0][0].item()
        scores[intent] = similarity
    
    best_intent = max(scores, key=scores.get)
    confidence = scores[best_intent]
    
    # Se confiança < threshold, retorna DIRECT (fallback seguro)
    if confidence < threshold:
        return "DIRECT", confidence
    
    return best_intent, confidence


# ============================================================================
# TESTES
# ============================================================================

test_messages = [
    # HUMAN
    "Isso aqui ta com um bug mt cabuloso no simec q a faq nao explica!",
    "já fiz tudo igual ao guia e continua negando acesso a minha api",
    "sou gestor avancado e tem uma divergencia critica nas planilhas sumindo",
    "essa sua resposta automatica ta mt fraca pro problema real daki",
    
    # MEC
    "como solicitar a exclusão de operador por oficio cgge-seb pro mec?",
    "como funciona o processo se eu botar 1 na coluna EDITA de enturmação no GPE?",
    "A api do simec manda cpf ou nis do aluno pra escola primeiro?",
    
     # DIRECT
    "Oi!",
    "Obrigado",
    "Tudo bem?",
    
    # Ambíguo
    "Dúvida sobre os dados",
    "Preciso de ajuda com a escola",
]

print("=" * 70)
print("TESTES DE CLASSIFICAÇÃO")
print("=" * 70)

for msg in test_messages:
    intent, confidence = classify_intent(msg)
    bar = "█" * int(confidence * 20)
    print(f"\n📝 {msg}")
    print(f"   → {intent:6} | Confiança: {confidence:.2%} {bar}")

print("\n" + "=" * 70)