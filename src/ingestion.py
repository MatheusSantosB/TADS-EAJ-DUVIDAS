import shutil
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings

from src.config import CHROMA_DIR, DATA_DIR, settings
from src.retriever import get_embeddings


def _is_supported_file(path: Path) -> bool:
    return path.suffix.lower() in {".pdf", ".txt"}


def _load_pdf(path: Path) -> list[Document]:
    loader = PyPDFLoader(str(path))
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = path.name
    return documents


def _load_txt(path: Path) -> list[Document]:
    loader = TextLoader(str(path), encoding="utf-8")
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = path.name
    return documents


def _load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".txt":
        return _load_txt(path)

    return []


def _remove_empty_documents(documents: list[Document]) -> tuple[list[Document], int]:
    valid_documents = [doc for doc in documents if doc.page_content.strip()]
    empty_documents = len(documents) - len(valid_documents)
    return valid_documents, empty_documents


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    documents: list[Document] = []

    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        return documents

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue

        if _is_supported_file(path):
            loaded_documents, _ = _remove_empty_documents(_load_file(path))
            documents.extend(loaded_documents)

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)


def _create_vectorstore() -> Chroma:
    return Chroma(
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_name="tads_docs",
        client_settings=ChromaSettings(
            is_persistent=True,
            persist_directory=str(CHROMA_DIR),
            anonymized_telemetry=False,
        ),
    )


def _add_documents_in_batches(
    vectorstore: Chroma,
    chunks: list[Document],
    batch_size: int,
) -> int:
    inserted = 0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectorstore.add_documents(batch)
        inserted += len(batch)
        print(f"  Chunks inseridos: {inserted}/{len(chunks)}")

    return inserted


def ingest_documents(reset: bool = True) -> dict[str, object]:
    files = [
        path
        for path in sorted(DATA_DIR.rglob("*"))
        if path.is_file() and _is_supported_file(path)
    ] if DATA_DIR.exists() else []

    if not files:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return {
            "documents": 0,
            "chunks": 0,
            "inserted_chunks": 0,
            "stored_chunks": 0,
            "empty_documents": 0,
            "files": [],
            "errors": [],
            "persist_directory": str(CHROMA_DIR),
        }

    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    vectorstore = _create_vectorstore()
    batch_size = max(1, settings.ingestion_batch_size)
    total_documents = 0
    total_chunks = 0
    inserted_chunks = 0
    total_empty_documents = 0
    file_results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for path in files:
        print(f"Processando: {path.name}")
        try:
            loaded_documents = _load_file(path)
        except Exception as exc:
            message = str(exc)
            errors.append({"file": path.name, "error": message})
            print(f"  Erro ao carregar arquivo: {message}")
            continue

        documents, empty_documents = _remove_empty_documents(loaded_documents)
        chunks = split_documents(documents) if documents else []

        total_documents += len(documents)
        total_chunks += len(chunks)
        total_empty_documents += empty_documents

        file_result = {
            "file": path.name,
            "documents": len(documents),
            "empty_documents": empty_documents,
            "chunks": len(chunks),
        }
        file_results.append(file_result)

        print(
            "  Paginas/documentos validos: "
            f"{len(documents)} | vazios ignorados: {empty_documents} | chunks: {len(chunks)}"
        )

        if chunks:
            inserted_chunks += _add_documents_in_batches(
                vectorstore=vectorstore,
                chunks=chunks,
                batch_size=batch_size,
            )

    stored_chunks = vectorstore._collection.count()

    return {
        "documents": total_documents,
        "chunks": total_chunks,
        "inserted_chunks": inserted_chunks,
        "stored_chunks": stored_chunks,
        "empty_documents": total_empty_documents,
        "files": file_results,
        "errors": errors,
        "persist_directory": str(CHROMA_DIR),
    }
