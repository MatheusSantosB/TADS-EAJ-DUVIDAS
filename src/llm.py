from langchain_groq import ChatGroq

from src.config import settings


def get_llm() -> ChatGroq:
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY nao encontrada. Crie um arquivo .env com sua chave da Groq."
        )

    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.temperature,
    )
