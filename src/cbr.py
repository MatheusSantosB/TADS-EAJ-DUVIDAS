import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.config import CASES_FILE, settings


def load_cases(path: Path = CASES_FILE) -> list[dict[str, Any]]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")
        return []

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    return json.loads(content)


def save_cases(cases: list[dict[str, Any]], path: Path = CASES_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9áàâãéêíóôõúç\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def find_similar_case(question: str) -> dict[str, Any] | None:
    best_case: dict[str, Any] | None = None
    best_score = 0.0

    for case in load_cases():
        score = _similarity(question, case.get("pergunta", ""))
        if score > best_score:
            best_score = score
            best_case = case

    if best_case and best_score >= settings.cbr_threshold:
        best_case = dict(best_case)
        best_case["similaridade"] = round(best_score, 3)
        return best_case

    return None


def add_case(
    pergunta: str,
    resposta: str,
    categoria: str,
    fontes: list[str],
) -> None:
    cases = load_cases()

    for case in cases:
        if _normalize(case.get("pergunta", "")) == _normalize(pergunta):
            return

    cases.append(
        {
            "pergunta": pergunta,
            "resposta": resposta,
            "categoria": categoria,
            "fontes": fontes,
        }
    )
    save_cases(cases)
