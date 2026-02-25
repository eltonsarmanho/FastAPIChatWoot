
"""
Classificador de Intenção baseado em Embeddings Semânticos
Modelo gratuito: all-MiniLM-L6-v2 (multilíngue, rápido, ~22MB)
"""

from sentence_transformers import SentenceTransformer, util
import numpy as np

# Modelo gratuito multilíngue compacto
model = SentenceTransformer('all-MiniLM-L6-v2')

# Exemplos de cada intenção (português)
intent_examples = {
    "HUMAN": [
        "Quero falar com suporte",
        "Quero falar com um humano",
        "Me encaminhe para a equipe financeira",
        "Preciso falar com um atendente",
        "Quero conversar com o time de suporte",
        "Gostaria de ser atendido por uma pessoa",
        "Quero contato com o suporte humano",
        "Preciso de atendimento pessoal",
        "Me transfira para um agente humano",
        "Quero falar com alguém da equipe",
    ],
    "MEC": [
        "Como faço um processo de crédito?",
        "Qual é o regimento interno?",
        "O que diz a resolução sobre TCC?",
        "Como solicitar credenciamento?",
        "Qual a carga horária mínima?",
        "Como funciona o processo de matrícula?",
        "Quais são os documentos necessários para inscrição?",
        "Como consultar o histórico escolar?",
        "Qual o procedimento para trancamento de matrícula?",
        "Como solicitar declaração de matrícula?",
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
    "Quero falar com a equipe financeira",
    "Me encaminhe para um humano",
    "Preciso de suporte imediato",
    "Falar com suporte",
    
    # MEC
    "Como faço para pedir crédito?",
    "Qual é o regimento interno da FASI?",
    "Qual a resolução sobre TCC?",
    
    # DIRECT
    "Oi!",
    "Obrigado",
    "Tudo bem?",
    
    # Ambíguo
    "Informações",
    "Preciso de ajuda",
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