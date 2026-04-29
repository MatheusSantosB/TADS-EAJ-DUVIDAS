import re
import unicodedata
from typing import Any, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from src.llm import get_llm
from src.prompts import ANSWER_PROMPT
from src.retriever import retrieve_documents
from src.utils import documents_to_context, extract_sources


class ChatState(TypedDict, total=False):
    question: str
    resolved_question: str
    retrieval_question: str
    history: list[dict[str, str]]
    category: str
    documents: list[Document]
    case: dict[str, Any] | None
    answer: str
    sources: list[str]


def prepare_question(state: ChatState) -> ChatState:
    question = state["question"]
    if _is_small_talk(question):
        return {
            "answer": "Certo. Pode mandar a proxima pergunta sobre o curso de TADS.",
            "sources": [],
        }

    history = state.get("history", [])
    normalized = question.lower()
    refers_to_previous = any(
        term in normalized
        for term in ["essas", "esses", "isso", "isto", "elas", "eles", "delas", "deles"]
    )

    if refers_to_previous and history:
        last_turn = history[-1]
        previous_question = last_turn.get("question", "")
        retrieval_question = (
            f"Pergunta anterior: {previous_question}\n"
            f"Pergunta atual: {question}"
        )
        resolved_question = retrieval_question
    else:
        retrieval_question = question
        resolved_question = question

    return {"retrieval_question": retrieval_question, "resolved_question": resolved_question}


def route_after_prepare(state: ChatState) -> Literal["final_response", "classify_question"]:
    if state.get("answer"):
        return "final_response"
    return "classify_question"


def classify_question(state: ChatState) -> ChatState:
    question = state.get("retrieval_question", state["question"]).lower()

    if any(term in question for term in ["tcc", "monografia", "artigo"]):
        category = "TCC"
    elif any(term in question for term in ["estagio", "estágio"]):
        category = "Estagio"
    elif any(term in question for term in ["hora", "complementar", "atividade"]):
        category = "Atividades complementares"
    elif any(term in question for term in ["disciplina", "grade", "curriculo", "currículo"]):
        category = "Estrutura curricular"
    else:
        category = "Geral"

    return {"category": category}


def retrieve_docs(state: ChatState) -> ChatState:
    documents = retrieve_documents(state.get("retrieval_question", state["question"]))
    return {
        "documents": documents,
        "sources": extract_sources(documents),
    }


def retrieve_cases(state: ChatState) -> ChatState:
    return {"case": None}


def generate_answer(state: ChatState) -> ChatState:
    documents = state.get("documents", [])

    if not documents:
        return {
            "answer": (
                "Nao encontrei trechos relevantes nos documentos indexados para "
                "responder com seguranca. Verifique se os arquivos foram colocados "
                "em data/ e se a ingestao foi executada."
            )
        }

    case = state.get("case")
    if case:
        case_context = (
            f"Pergunta parecida: {case.get('pergunta')}\n"
            f"Categoria: {case.get('categoria')}\n"
            f"Fontes anteriores: {case.get('fontes')}\n"
            f"Similaridade: {case.get('similaridade')}\n"
            "Use o caso apenas para entender a intencao da pergunta. "
            "A resposta deve ser extraida dos documentos recuperados agora."
        )
    else:
        case_context = "Nenhum caso parecido encontrado."

    chain = ANSWER_PROMPT | get_llm()
    response = chain.invoke(
        {
            "question": state.get("resolved_question", state["question"]),
            "category": state.get("category", "Geral"),
            "context": documents_to_context(documents),
            "case_context": case_context,
            "conversation_context": _format_history(
                state.get("history", []),
                include_history=_uses_previous_reference(state["question"]),
            ),
        }
    )

    return {"answer": response.content}


def final_response(state: ChatState) -> ChatState:
    return state


def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("prepare_question", prepare_question)
    graph.add_node("classify_question", classify_question)
    graph.add_node("retrieve_docs", retrieve_docs)
    graph.add_node("retrieve_cases", retrieve_cases)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("final_response", final_response)

    graph.set_entry_point("prepare_question")
    graph.add_conditional_edges(
        "prepare_question",
        route_after_prepare,
        {
            "final_response": "final_response",
            "classify_question": "classify_question",
        },
    )
    graph.add_edge("classify_question", "retrieve_docs")
    graph.add_edge("retrieve_docs", "retrieve_cases")
    graph.add_edge("retrieve_cases", "generate_answer")
    graph.add_edge("generate_answer", "final_response")
    graph.add_edge("final_response", END)

    return graph.compile()


def _format_history(history: list[dict[str, str]], include_history: bool = False) -> str:
    if not include_history or not history:
        return "Sem historico recente."

    turns = []
    for item in history[-1:]:
        turns.append(f"Usuario: {item.get('question', '')}")
    return "\n".join(turns)


def _normalize_short_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_small_talk(question: str) -> bool:
    normalized = _normalize_short_text(question)
    if not normalized:
        return True

    acknowledgements = {
        "ta",
        "tá",
        "ok",
        "okay",
        "certo",
        "beleza",
        "blz",
        "entendi",
        "valeu",
        "obrigado",
        "obrigada",
        "show",
        "sim",
        "nao",
        "não",
    }
    if normalized in {_normalize_short_text(item) for item in acknowledgements}:
        return True
    if normalized.startswith("se eu falar so "):
        trailing_text = normalized.removeprefix("se eu falar so ").strip()
        if trailing_text in {_normalize_short_text(item) for item in acknowledgements}:
            return True
    if normalized.startswith("so "):
        trailing_text = normalized.removeprefix("so ").strip()
        if trailing_text in {_normalize_short_text(item) for item in acknowledgements}:
            return True

    academic_markers = [
        "tads",
        "curso",
        "hora",
        "semestre",
        "periodo",
        "materia",
        "disciplina",
        "tcc",
        "estagio",
        "complementar",
        "matriz",
        "ppc",
        "formar",
        "monografia",
    ]
    has_academic_marker = any(marker in normalized for marker in academic_markers)
    has_question_signal = "?" in question or any(
        normalized.startswith(prefix)
        for prefix in ["qual", "quais", "quanto", "quantos", "como", "onde", "quando", "posso"]
    )

    return len(normalized.split()) <= 2 and not (has_academic_marker or has_question_signal)


def _uses_previous_reference(question: str) -> bool:
    normalized = _normalize_short_text(question)
    return any(
        term in normalized.split()
        for term in ["essas", "esses", "isso", "isto", "elas", "eles", "delas", "deles"]
    )
