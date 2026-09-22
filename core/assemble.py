"""Сборка договора: одна цитата меняется на текст из базы, противоречащие фразы снимаются."""

from __future__ import annotations

import re

from core.text_safety import tokens


def build_contract(contract: str, quote: str, clause: str, scenario: str) -> str | None:
    """Подставляет оговорку в наиболее уместное вхождение цитаты и убирает фразы, которые ей противоречат."""
    replaced = _replace_best(contract, quote, clause, scenario)
    if replaced is None:
        return None
    cleaned = _scrub(replaced, clause)
    if cleaned == contract:
        return None
    return cleaned


def _replace_best(contract: str, quote: str, clause: str, scenario: str) -> str | None:
    positions: list[int] = []
    start = 0
    while True:
        at = contract.find(quote, start)
        if at < 0:
            break
        positions.append(at)
        start = at + max(len(quote), 1)
    if not positions:
        return None
    wanted = set(tokens(scenario))
    best_at = positions[0]
    best_score = -1
    for at in positions:
        # При равном счёте берём более позднее вхождение: оно ближе к оперативному разделу,
        # а не к преамбуле, где та же фраза часто повторяется.
        value = len(set(tokens(_section_span(contract, at))) & wanted)
        if value >= best_score:
            best_score = value
            best_at = at
    return contract[:best_at] + clause + contract[best_at + len(quote) :]


def _section_span(contract: str, at: int) -> str:
    """Текст между нумерованными заголовками вокруг позиции. Соседние разделы в счёт не входят."""
    starts = [match.start() for match in re.finditer(r"(?m)^\d+\.\s+", contract)]
    left = 0
    right = len(contract)
    for start in starts:
        if start <= at:
            left = start
        elif start > at:
            right = start
            break
    return contract[left:right]


def _scrub(text: str, clause: str) -> str:
    kept_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")
            continue
        parts = [part for part in re.split(r"(?<=[.!?])\s+", stripped) if part]
        kept = [part for part in parts if not _conflicts(part, clause)]
        if kept:
            kept_lines.append(" ".join(kept))
    cleaned = "\n".join(kept_lines)
    if text.endswith("\n"):
        cleaned += "\n"
    return cleaned


def _conflicts(sentence: str, clause: str) -> bool:
    """Фраза спорит с уже вставленной оговоркой из базы."""
    if sentence.strip() and sentence.strip() in clause:
        return False
    source = sentence.lower().replace("ё", "е")
    guard = clause.lower().replace("ё", "е")
    if ("не подписывает" in guard or "считаются принятыми" in guard) and "транш" in guard:
        if "транш" in source and "подписан" in source:
            return True
    if ("пен" in guard or "неусто" in guard) and "оплат" in guard and ("задерж" in guard or "просроч" in guard):
        if "не предусмотрен" in source and any(word in source for word in ("пен", "штраф", "неусто")):
            return True
    if "не более" in guard and any(word in guard for word in ("штраф", "пен")):
        if "без ограничени" in source and any(word in source for word in ("штраф", "пен")):
            return True
    if "доказыван" in guard or "доказательств" in guard:
        if "безусловно" in source or "без требования доказатель" in source:
            return True
    if "итерац" in guard and "огранич" in guard:
        if "не ограничен" in source and "прав" in source:
            return True
    return False
