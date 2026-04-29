from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import build_graph
from src.llm import get_llm
from src.retriever import get_embeddings


DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "gabarito_avaliacao.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "ragas_last_run.json"


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError("O arquivo de avaliacao precisa conter uma lista de casos.")

    for index, case in enumerate(cases, start=1):
        if "question" not in case or "ground_truth" not in case:
            raise ValueError(
                f"Caso {index} invalido: informe 'question' e 'ground_truth'."
            )

    return cases


def run_chatbot(
    cases: list[dict[str, Any]],
    limit: int | None = None,
    delay_seconds: float = 0.0,
    continue_on_error: bool = False,
) -> list[dict[str, Any]]:
    app = build_graph()
    selected_cases = cases[:limit] if limit else cases
    rows: list[dict[str, Any]] = []

    for index, case in enumerate(selected_cases, start=1):
        question = case["question"]
        print(f"[{index}/{len(selected_cases)}] Avaliando: {question}")

        try:
            result = app.invoke({"question": question, "history": []})
        except Exception as exc:
            if not continue_on_error:
                raise

            rows.append(
                {
                    "question": question,
                    "answer": "",
                    "contexts": [],
                    "ground_truth": case["ground_truth"],
                    "expected_sources": case.get("expected_sources", []),
                    "sources": [],
                    "user_input": question,
                    "response": "",
                    "retrieved_contexts": [],
                    "reference": case["ground_truth"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  ERRO: {type(exc).__name__}: {exc}")
            if delay_seconds > 0 and index < len(selected_cases):
                time.sleep(delay_seconds)
            continue

        documents = result.get("documents", [])
        contexts = [document.page_content for document in documents]
        sources = result.get("sources", [])

        rows.append(
            {
                "question": question,
                "answer": result.get("answer", ""),
                "contexts": contexts,
                "ground_truth": case["ground_truth"],
                "expected_sources": case.get("expected_sources", []),
                "sources": sources,
                # Nomes usados por versoes mais novas do Ragas.
                "user_input": question,
                "response": result.get("answer", ""),
                "retrieved_contexts": contexts,
                "reference": case["ground_truth"],
            }
        )

        if delay_seconds > 0 and index < len(selected_cases):
            time.sleep(delay_seconds)

    return rows


def source_hit_rate(rows: list[dict[str, Any]]) -> float | None:
    scored_rows = 0
    hits = 0

    for row in rows:
        expected_sources = row.get("expected_sources") or []
        if not expected_sources:
            continue

        scored_rows += 1
        used_sources = "\n".join(row.get("sources") or [])
        if any(expected in used_sources for expected in expected_sources):
            hits += 1

    if scored_rows == 0:
        return None
    return hits / scored_rows


def configure_ragas_metrics() -> list[Any]:
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Dependencias de avaliacao nao encontradas. Rode: "
            "pip install -r requirements-eval.txt"
        ) from exc

    wrapped_llm = LangchainLLMWrapper(get_llm())
    wrapped_embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    for metric in metrics:
        if hasattr(metric, "llm"):
            metric.llm = wrapped_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = wrapped_embeddings

    return metrics


def run_ragas(rows: list[dict[str, Any]]) -> Any:
    try:
        from datasets import Dataset
        from ragas import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "Dependencias de avaliacao nao encontradas. Rode: "
            "pip install -r requirements-eval.txt"
        ) from exc

    dataset = Dataset.from_list(rows)
    metrics = configure_ragas_metrics()
    return evaluate(dataset, metrics=metrics)


def save_results(path: Path, rows: list[dict[str, Any]], ragas_result: Any | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"cases": rows}

    if ragas_result is not None:
        if hasattr(ragas_result, "to_pandas"):
            payload["ragas"] = ragas_result.to_pandas().to_dict(orient="records")
        else:
            payload["ragas"] = str(ragas_result)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia o chatbot TADS com casos manuais e metricas Ragas."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Pausa entre perguntas para reduzir risco de rate limit.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Salva erros por pergunta e continua a avaliacao.",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Executa apenas o chatbot e a checagem simples de fontes.",
    )
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    rows = run_chatbot(
        cases,
        limit=args.limit,
        delay_seconds=args.delay_seconds,
        continue_on_error=args.continue_on_error,
    )

    hit_rate = source_hit_rate(rows)
    if hit_rate is not None:
        print(f"Taxa simples de fontes esperadas: {hit_rate:.2%}")

    ragas_result = None
    if not args.skip_ragas:
        ragas_result = run_ragas(rows)
        print(ragas_result)

    save_results(args.output, rows, ragas_result)
    print(f"Resultado salvo em: {args.output}")


if __name__ == "__main__":
    main()
