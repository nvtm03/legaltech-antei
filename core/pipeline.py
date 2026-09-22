"""Общий прогон трёх этапов для консоли и API."""

from __future__ import annotations

from pathlib import Path

from core.assemble import build_contract
from core.step1_decompose import decompose_contract
from core.step2_rag import RetrievedClause, ValidationResult, accepts_category, agentic_retrieval, verify_clause
from core.step3_stress import simulate_conflict
from schemas import AnalyzeResponse, FixSource

_MAX_CONTRACT = 50_000
_MAX_SCENARIO = 4_000
_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "clauses_db.json"


class InputError(ValueError):
    """Текст клиента нельзя отдать в пайплайн. Для API это 422, не 502."""


def run(contract_text: str, scenario: str, db_path: str | None = None) -> AnalyzeResponse:
    """Декомпозиция, поиск инструментом, стресс-тест и сборка текста."""
    _check_input(contract_text, scenario)
    path = db_path or str(_DEFAULT_DB)
    modules = decompose_contract(contract_text)
    retrieved = agentic_retrieval(contract_text, scenario, modules, db_path=path)
    validation = _validate(contract_text, scenario, retrieved)
    accepted = _accepted(retrieved, validation, scenario)
    reference = accepted.text if accepted is not None else None
    report = simulate_conflict(contract_text, scenario, reference_clause=reference)
    fix_source = FixSource.NONE
    assembled: str | None = None
    if accepted is not None:
        built = build_contract(contract_text, report.vulnerable_quote, accepted.text, scenario)
        if built is not None:
            assembled = built
            fix_source = FixSource.REFERENCE
    return AnalyzeResponse(
        modules=modules,
        best_clause=reference,
        clause_id=accepted.clause_id if accepted is not None else None,
        clause_relevant=validation.is_relevant and accepted is not None,
        validation_reason=validation.reason,
        risk_report=report,
        assembled_contract=assembled,
        fix_source=fix_source,
    )


def _accepted(
    retrieved: RetrievedClause | None,
    validation: ValidationResult,
    scenario: str,
) -> RetrievedClause | None:
    if retrieved is None or not validation.is_relevant:
        return None
    if not accepts_category(retrieved.category, scenario):
        return None
    return retrieved


def _validate(contract_text: str, scenario: str, retrieved: RetrievedClause | None) -> ValidationResult:
    if retrieved is None:
        return ValidationResult(
            is_relevant=False,
            reason="Поиск не вернул оговорку выше порога сходства.",
        )
    verdict = verify_clause(contract_text, retrieved.text, scenario)
    if verdict.is_relevant and not accepts_category(retrieved.category, scenario):
        return ValidationResult(
            is_relevant=False,
            reason="Категория оговорки не стыкуется со сценарием.",
        )
    return verdict


def _check_input(contract_text: str, scenario: str) -> None:
    if not contract_text.strip():
        raise InputError("Пустой текст договора")
    if len(contract_text) > _MAX_CONTRACT:
        raise InputError("Текст договора длиннее 50000 символов")
    if not scenario.strip():
        raise InputError("Пустой сценарий")
    if len(scenario) > _MAX_SCENARIO:
        raise InputError("Сценарий длиннее 4000 символов")
