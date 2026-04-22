from langchain_core.documents import Document


def print_header(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def documents_to_context(documents: list[Document]) -> str:
    parts = []
    total_chars = 0
    max_total_chars = 4200
    max_doc_chars = 2200

    for index, doc in enumerate(documents, start=1):
        if total_chars >= max_total_chars:
            break

        source = doc.metadata.get("source", "fonte_desconhecida")
        page = doc.metadata.get("page")
        page_text = f", pagina {page + 1}" if isinstance(page, int) else ""
        content = doc.page_content[:max_doc_chars]
        parts.append(
            f"[Trecho {index} | Fonte: {source}{page_text}]\n{content}"
        )
        total_chars += len(content)

    return "\n\n".join(parts)


def extract_sources(documents: list[Document]) -> list[str]:
    sources = []

    for doc in documents:
        source = doc.metadata.get("source", "fonte_desconhecida")
        page = doc.metadata.get("page")
        if isinstance(page, int):
            label = f"{source} - pagina {page + 1}"
        else:
            label = source

        if label not in sources:
            sources.append(label)

    return sources


def format_sources(sources: list[str]) -> str:
    if not sources:
        return "Fontes usadas: nenhuma fonte encontrada."

    lines = ["Fontes usadas:"]
    for source in sources:
        lines.append(f"- {source}")
    return "\n".join(lines)
