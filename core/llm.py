"""Вызов модели. При LEGALTECH_MOCK=1 или LLM_MODEL=mock сеть не используется."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from typing import Any

from dotenv import load_dotenv

from core.relevance import risk_label
from core.text_safety import read_block, tokens

load_dotenv()

logger = logging.getLogger(__name__)

VECTOR_SIZE = 1536


def use_mock() -> bool:
    """Мок включается флагом или именем модели и имеет приоритет над живым ключом."""
    flag = os.getenv("LEGALTECH_MOCK", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return True
    return os.getenv("LLM_MODEL", "gpt-4o-mini").strip().lower() in {"mock", "fake"}


def chat_model() -> str:
    return os.getenv("LLM_MODEL", "gpt-4o-mini")


def embedding_model() -> str:
    """Эмбеддинг задаётся явно. Иначе он совпадает с провайдером чат-модели."""
    explicit = os.getenv("EMBEDDING_MODEL", "").strip()
    if explicit:
        return explicit
    model = chat_model()
    if model.startswith("openrouter/"):
        return "openrouter/openai/text-embedding-3-small"
    if model.startswith(("gpt-", "openai/", "ft:")):
        return "text-embedding-3-small"
    raise RuntimeError(
        "Для этой чат-модели эмбеддинг не выбран. "
        "Задайте EMBEDDING_MODEL или запустите python main.py --mock."
    )


def embed(text: str) -> list[float]:
    """Вектор фиксированной длины. В моке это хеш основ слов, не вызов API."""
    if use_mock():
        return hash_embedding(text)
    from litellm import embedding

    try:
        response = embedding(model=embedding_model(), input=text, timeout=60)
    except Exception as exc:
        logger.exception("Запрос эмбеддинга не удался")
        raise RuntimeError("Запрос эмбеддинга не удался") from exc
    vector = response.data[0]["embedding"]
    if len(vector) != VECTOR_SIZE:
        raise RuntimeError(f"Ожидался вектор {VECTOR_SIZE}, получен {len(vector)}")
    return vector


def hash_embedding(text: str) -> list[float]:
    """Детерминированный вектор: одна основа слова — один бакет со знаком."""
    vector = [0.0] * VECTOR_SIZE
    for token in tokens(text):
        digest = hashlib.sha256(token.encode()).digest()
        bucket = int.from_bytes(digest[:2], "big") % VECTOR_SIZE
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def complete(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0,
    timeout: int = 60,
    response_format: Any = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> Any:
    """Один вход для всех этапов. Ошибка провайдера не уносит текст исключения наверх."""
    if use_mock():
        return _mock_complete(messages, response_format=response_format, tools=tools)
    from litellm import completion

    kwargs: dict[str, Any] = {}
    if response_format is not None:
        kwargs["response_format"] = response_format
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    try:
        return completion(
            model=chat_model(),
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            **kwargs,
        )
    except Exception as exc:
        logger.exception("Запрос к модели не удался")
        raise RuntimeError("Запрос к модели не удался") from exc


def message_text(response: Any) -> str:
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Модель вернула пустой ответ")
    return content


class _Fn:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name: str, arguments: str, call_id: str = "call_mock") -> None:
        self.id = call_id
        self.type = "function"
        self.function = _Fn(name, arguments)


class _Message:
    def __init__(self, content: str | None = None, tool_calls: list[_ToolCall] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message: _Message) -> None:
        self.message = message


class _Response:
    def __init__(self, message: _Message) -> None:
        self.choices = [_Choice(message)]


def _joined(messages: list[dict[str, Any]], role: str) -> str:
    return "\n".join(item.get("content") or "" for item in messages if item.get("role") == role)


def _family(scenario: str, contract: str) -> str:
    blob = f"{scenario}\n{contract}".lower()
    if "конфиденц" in blob or "разглаш" in blob:
        return "NDA"
    if "покупател" in blob or "поставщик" in blob or "оборудован" in blob:
        return "supply"
    return "SaaS"


def _best_line(contract: str, scenario: str) -> str:
    """Строка договора, у которой больше всего общих основ со сценарием."""
    wanted = set(tokens(scenario))
    best = ""
    best_score = -1
    for raw in contract.splitlines():
        line = raw.strip()
        if len(line) < 30:
            continue
        score = len(set(tokens(line)) & wanted)
        if score > best_score or (score == best_score and len(line) > len(best)):
            best = line
            best_score = score
    if not best:
        raise RuntimeError("В договоре нет строки, которую можно процитировать")
    return best


def _category(scenario: str) -> str:
    return risk_label(scenario)


def _score(line: str) -> int:
    """Жёсткая дыра — 9, обычный штраф — 7, остальное — 4. Не константа на любой договор."""
    blob = line.lower().replace("ё", "е")
    if any(word in blob for word in ("не ограничен", "без ограничени", "безусловно", "не предусмотрен")):
        return 9
    if any(word in blob for word in ("штраф", "пен", "вправе")):
        return 7
    return 4


def _hits_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for item in messages:
        if item.get("role") != "tool":
            continue
        try:
            payload = json.loads(item.get("content") or "")
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            hits = [row for row in payload if isinstance(row, dict)]
    return hits


def _pick_hit(hits: list[dict[str, Any]], scenario: str) -> str | None:
    wanted = set(tokens(scenario))
    best: str | None = None
    best_score = -1
    for hit in hits:
        clause_id = hit.get("id")
        if not isinstance(clause_id, str) or not clause_id:
            continue
        keywords = hit.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = []
        blob = " ".join(
            [
                str(hit.get("title") or ""),
                str(hit.get("category") or ""),
                " ".join(item for item in keywords if isinstance(item, str)),
            ]
        )
        score = len(set(tokens(blob)) & wanted)
        if score > best_score:
            best = clause_id
            best_score = score
    return best


def _mock_complete(
    messages: list[dict[str, Any]],
    *,
    response_format: Any,
    tools: list[dict[str, Any]] | None,
) -> _Response:
    """Детерминированный ответ по роли запроса. Нужен, чтобы пайплайн шёл без ключа."""
    system = _joined(messages, "system")
    user = _joined(messages, "user")
    name = getattr(response_format, "__name__", "")

    if tools and name == "":
        hits = _hits_from_messages(messages)
        if hits:
            scenario = read_block(user, "scenario")
            chosen = _pick_hit(hits, scenario)
            if chosen:
                arguments = json.dumps({"clause_id": chosen}, ensure_ascii=False)
                return _Response(
                    _Message(content=None, tool_calls=[_ToolCall("select_clause", arguments, "call_select")])
                )
        if any(item.get("role") == "tool" for item in messages):
            return _Response(_Message(content="Подходящей оговорки нет."))
        contract = read_block(user, "contract")
        scenario = read_block(user, "scenario")
        arguments = json.dumps(
            {"query": scenario or user[:500], "applies_to": _family(scenario, contract)},
            ensure_ascii=False,
        )
        return _Response(
            _Message(content=None, tool_calls=[_ToolCall("search_clauses", arguments, "call_search")])
        )

    if name == "ContractModules":
        from core.headings import modules_from_headings

        modules = modules_from_headings(read_block(user, "contract") or user)
        return _Response(_Message(content=modules.model_dump_json()))

    if name == "ValidationResult":
        clause = read_block(user, "clause")
        scenario = read_block(user, "scenario")
        shared = set(tokens(clause)) & set(tokens(scenario))
        relevant = len(shared) >= 3
        reason = (
            "Оговорка закрывает факты сценария."
            if relevant
            else "Оговорка не про этот сценарий."
        )
        payload = json.dumps({"is_relevant": relevant, "reason": reason}, ensure_ascii=False)
        return _Response(_Message(content=payload))

    if isinstance(response_format, dict) and response_format.get("type") == "json_object":
        contract = read_block(user, "contract")
        scenario = read_block(user, "scenario")
        reference = read_block(user, "reference")
        quote = _best_line(contract, scenario)
        fix = reference or "Стороны согласуют последствия отдельно и письменно."
        payload = json.dumps(
            {
                "score": _score(quote),
                "category": _category(scenario),
                "vulnerable_quote": quote,
                "suggested_fix": fix,
            },
            ensure_ascii=False,
        )
        return _Response(_Message(content=payload))

    contract = read_block(user, "contract")
    scenario = read_block(user, "scenario")
    line = _best_line(contract, scenario) if contract else scenario
    who = "Эксперт"
    if "финанс" in system.lower():
        who = "Финансист"
    elif "юрист" in system.lower():
        who = "Юрист"
    elif "операц" in system.lower():
        who = "Операционист"
    return _Response(_Message(content=f"{who}: сценарий проходит через формулировку «{line}»."))
