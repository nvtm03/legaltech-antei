"""Нарезка договора по нумерованным заголовкам. Общая для мока и сверки модулей."""

from __future__ import annotations

import re

from core.text_safety import MISSING
from schemas import ContractModules

_SECTION = re.compile(r"(?m)^(\d+)\.\s+(.+)$")
# Только заголовок раздела, не пункт списка внутри него.
# «1. За срыв сроков — пени.» не начинается с «Срок» и разделом не считается.
_FIELDS = (
    (re.compile(r"^предмет\b", re.IGNORECASE), "subject"),
    (re.compile(r"^порядок\s+расч|^оплат|^расч", re.IGNORECASE), "payment_terms"),
    (re.compile(r"^срок|^приемк|^приёмк", re.IGNORECASE), "deadlines_and_acceptance"),
    (re.compile(r"^ответствен|^неустой", re.IGNORECASE), "liability"),
    (re.compile(r"^разрешен|^спор", re.IGNORECASE), "dispute_resolution"),
)


def modules_from_headings(contract: str) -> ContractModules:
    """Собирает пять модулей из заголовков вида «1. Предмет». Пустой блок не выдумывается."""
    matches = [
        match
        for match in _SECTION.finditer(contract)
        if _field_for(match.group(2).strip()) is not None
    ]
    found = {name: MISSING for _, name in _FIELDS}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(contract)
        body = contract[start:end].strip()
        title = match.group(2).strip()
        field = _field_for(title)
        if field and body:
            found[field] = body
    return ContractModules.model_validate(found)


def _field_for(title: str) -> str | None:
    for pattern, name in _FIELDS:
        if pattern.search(title):
            return name
    return None
