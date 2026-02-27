import logging
import os
import threading
from dotenv import load_dotenv

# Suprime logs verbosos de bibliotecas de ML
for _lib in ("sentence_transformers", "transformers", "mlx_lm", "huggingface_hub"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

load_dotenv(override=True)
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

_INTENT_EXAMPLES: dict[str, list[str]] = {
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

logger = logging.getLogger("classificador_intencao")


class OrquestradorHF:
    """
    Classificador semântico de intenção (HUMAN | MEC | DIRECT).

    O modelo e os embeddings são inicializados na primeira chamada a
    ``classify()`` (lazy init), mantendo o import instantâneo.
    A classificação usa média dos top-k vizinhos mais próximos em vez
    de centróide simples, reduzindo erros em classes com alta variância.
    """

    _MODEL_NAME = "all-MiniLM-L6-v2"
    _TOP_K = 5  # vizinhos considerados por classe

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._lock = threading.Lock()
        self._model = None
        self._embeddings: dict[str, object] = {}  # intent -> np.ndarray (N, D)

    # ------------------------------------------------------------------
    # Inicialização lazy
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Carrega modelo e embeddings de forma thread-safe na 1ª chamada."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:  # double-checked
                return
            logger.info("[classificador] Carregando modelo %s…", self._MODEL_NAME)
            from sentence_transformers import SentenceTransformer
            import numpy as np

            model = SentenceTransformer(self._MODEL_NAME)
            embeddings: dict[str, object] = {}
            for intent, examples in _INTENT_EXAMPLES.items():
                embeddings[intent] = model.encode(examples, show_progress_bar=False)
            self._np = np
            self._embeddings = embeddings
            self._model = model
            logger.info("[classificador] Modelo carregado com sucesso.")

    # ------------------------------------------------------------------
    # Pré-aquecimento opcional (chamar no startup em background)
    # ------------------------------------------------------------------
    def warmup(self) -> None:
        """Força o carregamento do modelo antecipadamente."""
        self._ensure_loaded()

    # ------------------------------------------------------------------
    # Classificação
    # ------------------------------------------------------------------
    def classify(self, message: str) -> tuple[str, float]:
        """
        Classifica a intenção da mensagem (HUMAN, MEC, DIRECT).

        Estratégia: para cada intenção, calcula a similaridade cosseno
        contra todos os exemplos e retorna a média dos top-k scores.
        Isso captura melhor classes com alta variância interna do que
        o centróide simples.

        Returns:
            (intenção, confiança) – confiança em [0, 1].
        """
        from sentence_transformers import util

        self._ensure_loaded()
        np = self._np

        query_emb = self._model.encode(message, show_progress_bar=False)

        scores: dict[str, float] = {}
        for intent, intent_embs in self._embeddings.items():
            sims = util.cos_sim(query_emb, intent_embs)[0].numpy()  # shape (N,)
            top_k = min(self._TOP_K, len(sims))
            scores[intent] = float(np.sort(sims)[-top_k:].mean())

        best_intent = max(scores, key=lambda k: scores[k])
        confidence = scores[best_intent]

        if confidence < self.threshold:
            return "DIRECT", confidence

        return best_intent, confidence
