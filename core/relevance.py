"""Детерминированная стыковка категории оговорки и сценария. Модель здесь не участвует."""

from __future__ import annotations

# Категории карточек базы и классы риска из RiskReport — разные словари.
# Связка нужна, чтобы вердикт саб-агента нельзя было подменить одной инструкцией в договоре.
CLAUSE_RISK: dict[str, set[str]] = {
    "acceptance": {"Operational", "Financial"},
    "liability": {"Legal", "Financial"},
    "sla": {"Operational"},
    "data": {"Legal"},
    "ip": {"Legal"},
    "payment": {"Financial"},
    "penalty": {"Financial", "Legal"},
    "quality": {"Operational", "Legal"},
    "confidentiality": {"Legal", "Financial"},
    "jurisdiction": {"Legal"},
    "force_majeure": {"Legal", "Operational"},
}


def scenario_risks(scenario: str) -> set[str]:
    """Какие классы риска видны по словам сценария, без ответа модели."""
    blob = scenario.lower().replace("ё", "е")
    found: set[str] = set()
    if any(word in blob for word in ("доказ", "убыт", "подсудност")):
        found.add("Legal")
    if any(word in blob for word in ("правк", "акт", "итерац", "прием")):
        found.add("Operational")
    if any(word in blob for word in ("штраф", "пен", "оплат", "транш", "млн", "неусто")):
        found.add("Financial")
    if not found:
        found.add("Operational")
    return found


def risk_label(scenario: str) -> str:
    """Один класс для отчёта. Доказывание убытков важнее слова «штраф»."""
    risks = scenario_risks(scenario)
    blob = scenario.lower().replace("ё", "е")
    if "Legal" in risks and any(word in blob for word in ("доказ", "убыт")):
        return "Legal"
    if "Operational" in risks and any(word in blob for word in ("правк", "акт", "итерац")):
        return "Operational"
    if "Financial" in risks:
        return "Financial"
    if "Legal" in risks:
        return "Legal"
    return "Operational"


def category_compatible(clause_category: str, scenario: str) -> bool:
    """Карточка SLA не закрывает денежный сценарий, даже если модель сказала «да»."""
    allowed = CLAUSE_RISK.get(clause_category)
    if not allowed:
        return False
    return bool(allowed & scenario_risks(scenario))
