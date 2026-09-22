"""Общие проверки текста: рамка данных, цитата, верность модулей источнику."""

from __future__ import annotations

import re
import secrets

MISSING = "В договоре не указано."

_TOKEN = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")
_STOP = {
    "для",
    "при",
    "или",
    "как",
    "что",
    "это",
    "эта",
    "эти",
    "этот",
    "без",
    "над",
    "под",
    "все",
    "его",
    "нее",
    "них",
    "том",
    "тем",
    "если",
    "либо",
    "только",
    "также",
    "после",
    "перед",
    "между",
    "через",
    "каждый",
    "каждая",
    "которые",
    "который",
    "которая",
    "договор",
    "договора",
    "сторона",
    "стороны",
    "сторон",
}


# Длинные окончания раньше коротких. «правки» и «правок» сходятся в «прав».
_ENDINGS = (
    "иями",
    "ями",
    "ами",
    "ого",
    "ему",
    "ому",
    "ыми",
    "ими",
    "ах",
    "ях",
    "ов",
    "ев",
    "ей",
    "ий",
    "ый",
    "ой",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
    "ам",
    "ям",
    "ом",
    "ем",
    "ую",
    "юю",
    "ок",
    "ка",
    "ки",
    "ке",
    "ку",
    "а",
    "я",
    "ы",
    "и",
    "о",
    "е",
    "у",
    "ю",
)


def stem(token: str) -> str:
    """Снимает одно окончание: правки и правок попадают в один бакет."""
    word = token.lower().replace("ё", "е")
    if len(word) <= 4:
        return word
    for ending in _ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= 4:
            return word[: -len(ending)]
    return word


def tokens(text: str) -> list[str]:
    """Значимые основы слов. Короткие служебные слова отбрасываются."""
    found: list[str] = []
    for raw in _TOKEN.findall(text):
        word = raw.lower().replace("ё", "е")
        if word in _STOP or len(word) < 4 and word not in {"акт", "суд", "иск"}:
            continue
        found.append(stem(word))
    return found


_BLOCK = re.compile(
    r"(?s)<<<(?P<tag>[a-z0-9_-]+)-(?P<nonce>[0-9a-f]{16})>>>\n(?P<body>.*?)\n<<<END-(?P=tag)-(?P=nonce)>>>"
)


def as_data(tag: str, payload: str) -> str:
    """Кладёт текст между nonce-маркерами. Сам текст не меняется: плейсхолдер <СРОК> остаётся <СРОК>."""
    nonce = secrets.token_hex(8)
    start = f"<<<{tag}-{nonce}>>>"
    end = f"<<<END-{tag}-{nonce}>>>"
    safe = payload.replace(end, end[:-3] + "››>")
    return f"{start}\n{safe}\n{end}"


def read_block(text: str, tag: str) -> str:
    """Достаёт тело блока, который собрал as_data. Чужой маркер с другим nonce не закрывает рамку."""
    for match in _BLOCK.finditer(text):
        if match.group("tag") == tag:
            return match.group("body")
    return ""


def locate_quote(quote: str, source: str) -> str | None:
    """Возвращает фрагмент source, который и есть цитата.

    Сначала ищет точное вхождение. Если модель схлопнула переносы или
    повторила пробелы, находит тот же фрагмент по нормализованным пробелам
    и возвращает срез исходного текста, чтобы замена попала в договор.
    Если модель процитировала ‹ › вместо угловых скобок шаблона, скобки
    возвращаются к исходным < > до поиска.
    """
    found = _locate_once(quote, source)
    if found is not None:
        return found
    if "‹" in quote or "›" in quote:
        return _locate_once(quote.replace("‹", "<").replace("›", ">"), source)
    return None


def _locate_once(quote: str, source: str) -> str | None:
    needle = quote.strip()
    if not needle:
        return None
    if needle in source:
        return needle
    collapsed_quote, _ = _collapse(needle)
    collapsed_source, mapping = _collapse(source)
    if not collapsed_quote:
        return None
    at = collapsed_source.find(collapsed_quote)
    if at < 0:
        return None
    start = mapping[at]
    end = mapping[at + len(collapsed_quote) - 1] + 1
    return source[start:end]


def _collapse(text: str) -> tuple[str, list[int]]:
    """Схлопывает пробелы и помнит, какой символ source дал каждый символ результата."""
    chars: list[str] = []
    mapping: list[int] = []
    pending: int | None = None
    for index, char in enumerate(text):
        if char.isspace():
            if chars and pending is None:
                pending = index
            continue
        if pending is not None:
            chars.append(" ")
            mapping.append(pending)
            pending = None
        chars.append(char)
        mapping.append(index)
    return "".join(chars), mapping


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def grounded_in_source(fragment: str, source: str) -> bool:
    """Каждая фраза модуля — подстрока договора. Пересказ и дописанная норма не проходят."""
    text = fragment.strip()
    if text == MISSING:
        return True
    if locate_quote(text, source) is not None:
        return True
    parts = [part.strip() for part in _SENTENCE.split(text) if part.strip()]
    if len(parts) < 2:
        return False
    return all(locate_quote(part, source) is not None for part in parts)
