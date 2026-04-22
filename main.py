from src.graph import build_graph
from src.ingestion import ingest_documents
from src.utils import format_sources, print_header


def run_ingestion() -> None:
    print_header("Ingestao de documentos")
    result = ingest_documents(reset=True)
    print(f"Documentos carregados: {result['documents']}")
    print(f"Chunks indexados: {result['chunks']}")
    print(f"Chunks inseridos no Chroma: {result.get('inserted_chunks', 0)}")
    print(f"Chunks armazenados no Chroma: {result.get('stored_chunks', 0)}")
    print(f"Paginas/documentos vazios ignorados: {result.get('empty_documents', 0)}")
    if result.get("errors"):
        print("Arquivos com erro:")
        for error in result["errors"]:
            print(f"- {error['file']}: {error['error']}")
    print(f"Banco vetorial: {result['persist_directory']}")


def run_chat() -> None:
    print_header("Chatbot TADS")
    print("Digite sua pergunta sobre o curso de TADS.")
    print("Para sair, digite: sair")
    print()

    app = build_graph()
    history: list[dict[str, str]] = []

    while True:
        question = input("Voce: ").strip()

        if not question:
            continue

        if question.lower() in {"sair", "exit", "quit"}:
            print("Encerrando chatbot.")
            break

        try:
            result = app.invoke({"question": question, "history": history})
            answer = result.get("answer", "Nao foi possivel gerar uma resposta.")
            print()
            print("Resposta:")
            print(answer)
            sources = result.get("sources", [])
            if sources:
                print()
                print(format_sources(sources))
            print()
            history.append({"question": question, "answer": answer})
            history = history[-5:]
        except Exception as exc:
            print()
            print("Ocorreu um erro ao processar a pergunta.")
            print(f"Detalhe: {exc}")
            print()


def main() -> None:
    print("1 - Rodar ingestao dos documentos")
    print("2 - Iniciar chatbot")
    option = input("Escolha uma opcao: ").strip()

    if option == "1":
        run_ingestion()
    elif option == "2":
        run_chat()
    else:
        print("Opcao invalida.")


if __name__ == "__main__":
    main()
