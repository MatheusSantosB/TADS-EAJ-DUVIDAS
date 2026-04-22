from functools import lru_cache
import re
import unicodedata

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from src.config import CHROMA_DIR, settings


STOPWORDS = {
    "a",
    "as",
    "apenas",
    "cadastrar",
    "com",
    "da",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "eu",
    "me",
    "no",
    "o",
    "os",
    "para",
    "posso",
    "por",
    "quais",
    "quantas",
    "quero",
    "que",
    "sao",
    "saber",
    "tem",
    "tenho",
    "uma",
}

SYNONYMS = {
    "disciplina": ["componente", "curricular"],
    "disciplinas": ["componentes", "curriculares"],
    "materia": ["componente", "curricular"],
    "materias": ["componentes", "curriculares"],
    "obrigatoria": ["obrigatorios"],
    "obrigatorias": ["obrigatorios"],
    "obrigatorio": ["obrigatorios"],
    "estagio": ["profissional"],
    "estagios": ["profissional"],
    "hora": ["carga", "horaria"],
    "horas": ["carga", "horaria"],
    "reprovar": ["pre", "requisito", "pre requisito", "correquisito"],
    "reprovado": ["pre", "requisito", "pre requisito", "correquisito"],
    "pagar": ["pre", "requisito", "pre requisito", "correquisito"],
    "prende": ["pre", "requisito", "pre requisito", "correquisito"],
    "trava": ["pre", "requisito", "pre requisito", "correquisito"],
    "formar": ["tcc", "trabalho", "conclusao", "relatorio"],
    "formatura": ["tcc", "trabalho", "conclusao", "relatorio"],
}


@lru_cache(maxsize=1)
def get_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(
        model_name=settings.embedding_model,
    )


def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
        collection_name="tads_docs",
        client_settings=ChromaSettings(
            is_persistent=True,
            persist_directory=str(CHROMA_DIR),
            anonymized_telemetry=False,
        ),
    )


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("º", "o").replace("°", "o")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _query_terms(question: str) -> list[str]:
    normalized = _normalize(question)
    terms = [
        term
        for term in normalized.split()
        if (len(term) >= 3 or term.isdigit()) and term not in STOPWORDS
    ]

    expanded_terms = list(terms)
    for term in terms:
        expanded_terms.extend(SYNONYMS.get(term, []))

    # Normaliza buscas comuns por periodo: "1 periodo", "1o periodo", "1º periodo".
    period_match = re.search(r"\b(\d+)\s*o?\s*periodo\b", normalized)
    if period_match:
        number = period_match.group(1)
        expanded_terms.extend([f"{number} periodo", f"{number}o periodo"])

    expanded_terms.extend(re.findall(r"\btad\d{4}\b", normalized))
    return list(dict.fromkeys(expanded_terms))


def _has_structured_intent(question: str) -> bool:
    normalized = _normalize(question)
    return any(
        term in normalized
        for term in [
            "periodo",
            "materia",
            "materias",
            "disciplina",
            "disciplinas",
            "carga horaria",
            "horas",
            "divid",
            "distribu",
            "2295",
            "estagio",
            "formar",
            "tcc",
        ]
    )


def _lexical_score(question: str, content: str) -> float:
    terms = _query_terms(question)
    if not terms:
        return 0.0

    normalized_question = _normalize(question)
    normalized_content = _normalize(content)
    score = 0.0
    matched_single_terms = 0
    single_terms = [term for term in terms if " " not in term]

    for term in terms:
        if " " in term:
            if term in normalized_content:
                score += 6.0
        elif term in normalized_content:
            matched_single_terms += 1
            score += 2.0 if term.startswith("tad") else 1.0

    if normalized_question and normalized_question in normalized_content:
        score += 8.0

    if single_terms and matched_single_terms == len(single_terms):
        score += 5.0

    # Sinais estruturais do proprio documento, uteis para tabelas do PPC.
    if "periodo" in normalized_question and "codigos nomes dos componentes" in normalized_content:
        score += 18.0
    if "carga horaria" in normalized_question or "horas" in normalized_question:
        if "carga horaria total" in normalized_content or "subtotais das cargas horarias" in normalized_content:
            score += 10.0
        if (
            "curso" in normalized_question
            and "atividades complementares" not in normalized_question
            and ("2295" in normalized_content or "2 295" in normalized_content)
        ):
            score += 35.0
        if "curso superior de tecnologia" in normalized_content and "carga horaria total" in normalized_content:
            score += 12.0
    if any(term in normalized_question for term in ["divid", "distribu", "composicao"]):
        if "componentes obrigatorios e optativos" in normalized_content:
            score += 18.0
        if "total geral" in normalized_content:
            score += 8.0
    if "estagio" in normalized_question and "estagio nao obrigatorio" in normalized_content:
        score += 18.0
    if "estagio" in normalized_question and any(term in normalized_question for term in ["formar", "tcc", "monografia"]):
        if "relatorio de estagio" in normalized_content:
            score += 35.0
        if "trabalho de conclusao" in normalized_content or "tcc" in normalized_content:
            score += 12.0
    if any(term in normalized_question for term in ["reprovar", "reprovado", "pagar", "prende", "trava"]):
        if "codigos nomes dos componentes curriculares cargas horarias pre requisitos" in normalized_content:
            score += 25.0
        if "pre requisitos" in normalized_content or "pre requisito" in normalized_content:
            score += 12.0
        if "correquisitos" in normalized_content:
            score += 6.0
    if "atividades complementares" in normalized_question:
        if "atividades complementares" in normalized_content:
            score += 20.0
        if "chi" in normalized_content and "chtp" in normalized_content:
            score += 18.0
        if "art 2" in normalized_content or "art 2o" in normalized_content:
            score += 8.0
        if "art 3" in normalized_content or "art 3o" in normalized_content:
            score += 8.0
        if "iniciacao a docencia" in normalized_content:
            score += 5.0
        if "iniciacao a pesquisa" in normalized_content:
            score += 5.0
        if "iniciacao profissional" in normalized_content:
            score += 5.0
        if "participacao em eventos" in normalized_content:
            score += 5.0
        if "componentes curriculares optativos" in normalized_content:
            score -= 12.0

    return score


def _lexical_search(vectorstore: Chroma, question: str, k: int) -> list[Document]:
    data = vectorstore._collection.get(include=["documents", "metadatas"])
    scored: list[tuple[float, Document]] = []

    for content, metadata in zip(data["documents"], data["metadatas"]):
        if not content:
            continue

        score = _lexical_score(question, content)

        if score >= 2.0:
            scored.append((score, Document(page_content=content, metadata=metadata or {})))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [document for _, document in scored[:k]]


def _expanded_context_documents(
    vectorstore: Chroma,
    question: str,
    seed_documents: list[Document],
    max_documents: int = 1,
) -> list[Document]:
    if not seed_documents or not _has_structured_intent(question):
        return []

    data = vectorstore._collection.get(include=["documents", "metadatas"])
    documents = data["documents"]
    metadatas = data["metadatas"]
    expanded_documents: list[Document] = []
    seen: set[tuple[str, int]] = set()

    for seed in seed_documents:
        source = seed.metadata.get("source")
        page = seed.metadata.get("page")
        if not source or not isinstance(page, int):
            continue

        key = (str(source), page)
        if key in seen:
            continue
        seen.add(key)

        ordered_pages = [page, page + 1, page + 2, page - 1]
        parts: list[str] = []
        pages: list[int] = []

        for current_page in ordered_pages:
            for content, metadata in zip(documents, metadatas):
                if not content or not metadata:
                    continue
                if metadata.get("source") != source:
                    continue
                candidate_page = metadata.get("page")
                if candidate_page != current_page:
                    continue

                parts.append(content)
                if isinstance(candidate_page, int):
                    pages.append(candidate_page)

        if not parts:
            continue

        combined_content = _focus_content(question, "\n\n".join(parts))[:2800]
        expanded_documents.append(
            Document(
                page_content=combined_content,
                metadata={
                    "source": source,
                    "page": page,
                    "expanded_start_page": min(pages) if pages else page,
                    "expanded_from_page": page,
                },
            )
        )

        if len(expanded_documents) >= max_documents:
            break

    expanded_documents.sort(
        key=lambda document: _lexical_score(question, document.page_content),
        reverse=True,
    )
    return expanded_documents


def _focus_content(question: str, content: str, window_chars: int = 3200) -> str:
    normalized_question = _normalize(question)
    start = 0

    period_match = re.search(r"\b(\d+)\s*o?\s*periodo\b", normalized_question)
    if period_match:
        period_number = period_match.group(1)
        raw_period_match = re.search(
            rf"{period_number}\s*[º°o]?\s*PER[IÍ]ODO",
            content,
            flags=re.IGNORECASE,
        )
        if raw_period_match:
            start = max(0, raw_period_match.start() - 80)

    if start == 0 and any(
        term in normalized_question for term in ["2295", "divid", "distribu", "carga horaria total"]
    ):
        for marker in ["Componentes Obrigatórios", "COMPONENTE CURRICULAR", "Total Geral"]:
            marker_index = content.find(marker)
            if marker_index != -1:
                start = max(0, marker_index - 120)
                break
        if start == 0:
            for marker in ["carga horária total", "carga horaria total", "2.295", "2295"]:
                marker_index = content.lower().find(marker)
                if marker_index != -1:
                    start = max(0, marker_index - 180)
                    break

    if start == 0 and "estagio" in normalized_question:
        for marker in ["Estágio não obrigatório", "Estágio não obrigatório"]:
            marker_index = content.find(marker)
            if marker_index != -1:
                start = max(0, marker_index - 180)
                break
    if start == 0 and "atividades complementares" in normalized_question:
        for marker in ["Art. 2", "Atividades de", "CHI CHTP", "atividades complementares"]:
            marker_index = content.find(marker)
            if marker_index != -1:
                start = max(0, marker_index - 180)
                break

    return content[start : start + window_chars]


def _neighbor_documents(
    vectorstore: Chroma,
    seed_documents: list[Document],
    max_neighbors: int = 8,
) -> list[Document]:
    """Inclui chunks vizinhos quando uma tabela continua no chunk/pagina seguinte."""
    if not seed_documents:
        return []

    data = vectorstore._collection.get(include=["documents", "metadatas"])
    neighbors: list[Document] = []

    for seed in seed_documents:
        source = seed.metadata.get("source")
        page = seed.metadata.get("page")
        if source is None or not isinstance(page, int):
            continue

        candidate_pages = {page, page + 1}
        for content, metadata in zip(data["documents"], data["metadatas"]):
            if len(neighbors) >= max_neighbors:
                return neighbors
            if not content or not metadata:
                continue
            if metadata.get("source") != source:
                continue
            if metadata.get("page") not in candidate_pages:
                continue

            normalized_content = _normalize(content)
            if (
                "periodo" in normalized_content
                or "carga horaria total" in normalized_content
                or re.search(r"\btad\d{4}\b", normalized_content)
            ):
                neighbors.append(Document(page_content=content, metadata=metadata))

    return neighbors


def _deduplicate_documents(documents: list[Document]) -> list[Document]:
    unique_documents: list[Document] = []
    seen: set[tuple[str, int | None, str]] = set()

    for document in documents:
        key = (
            str(document.metadata.get("source", "")),
            document.metadata.get("page"),
            document.page_content[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        unique_documents.append(document)

    return unique_documents


def retrieve_documents(question: str, k: int | None = None) -> list[Document]:
    final_k = k or settings.retriever_k
    vectorstore = get_vectorstore()
    lexical_documents = _lexical_search(vectorstore, question, k=max(final_k, 12))
    if lexical_documents:
        expanded_documents = _expanded_context_documents(
            vectorstore=vectorstore,
            question=question,
            seed_documents=lexical_documents[:4],
        )
        return _deduplicate_documents(
            expanded_documents + lexical_documents
        )[:final_k]

    vector_documents = vectorstore.similarity_search(question, k=final_k)
    return _deduplicate_documents(vector_documents)[:final_k]
