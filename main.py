"""Консольный запуск: разбор, поиск эталона, симуляция конфликта, сборка текста."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.llm import chat_model, use_mock
from core.pipeline import run
from core.scenarios import DEFAULT_SCENARIOS
from schemas import AnalyzeResponse, FixSource

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
console = Console()


def main() -> None:
    try:
        _run()
    except Exception as exc:
        console.print(Panel(escape(str(exc)), title="Ошибка", border_style="red"))
        raise SystemExit(1) from None


def _run() -> None:
    parser = argparse.ArgumentParser(
        description="Проверка договора: разбор, поиск эталона, симуляция конфликта."
    )
    parser.add_argument(
        "contracts",
        nargs="*",
        help="Пути к текстам. Без аргументов берутся data/contract_*.txt",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Один конфликтный сценарий для всех договоров. Без флага берётся сценарий кейса.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Прогон без API-ключа: хеш-поиск и детерминированные ответы.",
    )
    args = parser.parse_args()
    if args.mock:
        os.environ["LEGALTECH_MOCK"] = "1"
    paths = [Path(item) for item in args.contracts] or sorted(DATA.glob("contract_*.txt"))
    if not paths:
        raise RuntimeError("В data/ нет файлов contract_*.txt")

    console.print(
        Panel(
            "Декомпозиция  →  поиск с проверкой  →  стресс-тест  →  сборка",
            title="[bold]LegalTech AI[/]",
            subtitle="модульный конструктор договоров",
            border_style="bright_cyan",
            padding=(1, 2),
        )
    )
    db_path = str(DATA / "clauses_db.json")
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"Файл не найден: {path}")
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(f"Не удалось прочитать {path}") from exc
        scenario = _scenario_for(path, args.scenario)
        _render_case(path, text, scenario, db_path)


def _render_case(path: Path, text: str, scenario: str, db_path: str) -> None:
    """Прогоняет один договор и рисует панели этапов."""
    console.print(f"\n[bold]{escape(path.name)}[/]")
    with console.status("[bold green]Агент думает..."):
        result = run(text, scenario, db_path=db_path)
    console.print(_decomposition_panel(result))
    console.print(_retrieval_panel(result))
    console.print(_stress_panel(scenario, result))
    _announce_assembly(result)

    REPORTS.mkdir(exist_ok=True)
    destination = REPORTS / f"{path.stem}.md"
    try:
        destination.write_text(_markdown_report(path, scenario, result), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Не удалось сохранить отчет: {destination}") from exc
    console.print(f"[bold green]Отчет сохранен в {destination.as_posix()}[/]")


def _announce_assembly(result: AnalyzeResponse) -> None:
    if result.fix_source is FixSource.REFERENCE and result.assembled_contract:
        console.print("[bold green]Сборка успешна![/] В текст подставлена оговорка из базы.")
        return
    console.print("[dim]Сборка не выполнена: в базу не легла подтверждённая оговорка.[/]")


def _scenario_for(path: Path, override: str | None) -> str:
    """Берёт сценарий кейса из ТЗ, если пользователь не задал общий флаг."""
    if override:
        return override
    try:
        return DEFAULT_SCENARIOS[path.name]
    except KeyError as exc:
        raise RuntimeError(f"Для {path.name} нет сценария. Передайте --scenario.") from exc


def _markdown_report(path: Path, scenario: str, result: AnalyzeResponse) -> str:
    """Текстовый отчёт трёх этапов. Отклонённая оговорка в файл не попадает."""
    contract = result.modules
    report = result.risk_report
    modules = (
        ("Предмет", contract.subject),
        ("Оплата", contract.payment_terms),
        ("Сроки и приёмка", contract.deadlines_and_acceptance),
        ("Ответственность", contract.liability),
        ("Споры", contract.dispute_resolution),
    )
    decomposition = "\n".join(f"- **{label}:** {value}" for label, value in modules)
    if result.clause_relevant and result.best_clause:
        retrieval = (
            f"- **Релевантна:** да\n"
            f"- **Карточка:** {result.clause_id or '—'}\n"
            f"- **Оговорка:** {result.best_clause}\n"
            f"- **Почему:** {result.validation_reason}\n"
        )
    else:
        retrieval = (
            f"- **Релевантна:** нет\n"
            f"- **Оговорка:** не вставлена\n"
            f"- **Почему:** {result.validation_reason}\n"
        )
    assembled = result.assembled_contract or "Сборка не выполнена."
    mode = "Режим: mock" if use_mock() else f"Режим: модель {chat_model()}"
    return (
        f"# {path.name}\n\n"
        f"{mode}\n\n"
        f"## Декомпозиция\n\n"
        f"{decomposition}\n\n"
        f"## Результат RAG\n\n"
        f"{retrieval}\n"
        f"## Стресс-тест\n\n"
        f"- **Сценарий:** {scenario}\n"
        f"- **Категория:** {report.category.value}\n"
        f"- **Оценка:** {report.score}/10\n"
        f"- **Уязвимость:** {report.vulnerable_quote}\n"
        f"- **Защита:** {report.suggested_fix}\n"
        f"- **Источник защиты:** {result.fix_source.value}\n\n"
        f"## Собранный договор\n\n"
        f"{assembled}\n"
    )


def _decomposition_panel(result: AnalyzeResponse) -> Panel:
    contract = result.modules
    table = Table.grid(padding=(0, 2), expand=True)
    table.add_column(style="bold cyan", width=18, no_wrap=True)
    table.add_column()
    rows = (
        ("Предмет", contract.subject),
        ("Оплата", contract.payment_terms),
        ("Сроки и приёмка", contract.deadlines_and_acceptance),
        ("Ответственность", contract.liability),
        ("Споры", contract.dispute_resolution),
    )
    for label, value in rows:
        table.add_row(label, escape(value))
    return Panel(table, title="Этап 1 · Декомпозиция", border_style="cyan", padding=(1, 2))


def _retrieval_panel(result: AnalyzeResponse) -> Panel:
    accepted = result.clause_relevant and bool(result.best_clause)
    verdict = Text("РЕЛЕВАНТНА", style="bold green") if accepted else Text("ОТКЛОНЕНА", style="bold red")
    body = Table.grid(padding=(0, 2), expand=True)
    body.add_column(style="bold", width=18, no_wrap=True)
    body.add_column()
    body.add_row("Вердикт", verdict)
    body.add_row("Почему", escape(result.validation_reason))
    if accepted and result.best_clause:
        body.add_row("Оговорка", escape(result.best_clause))
    else:
        body.add_row("Оговорка", Text("В текст не вставляется.", style="dim"))
    border = "green" if accepted else "red"
    return Panel(body, title="Этап 2 · Поиск и проверка", border_style=border, padding=(1, 2))


def _stress_panel(scenario: str, result: AnalyzeResponse) -> Panel:
    report = result.risk_report
    danger = report.score >= 8
    tone = "red" if danger else "yellow"
    label = "DANGER" if danger else "WARNING"
    meter = Text()
    meter.append("█" * report.score, style=tone)
    meter.append("░" * (10 - report.score), style="dim")
    meter.append(f"  {report.score}/10", style=f"bold {tone}")

    body = Table.grid(padding=(0, 2), expand=True)
    body.add_column(style="bold", width=18, no_wrap=True)
    body.add_column()
    body.add_row("Сценарий", escape(scenario))
    body.add_row("Категория", report.category.value)
    body.add_row("Оценка", meter)
    body.add_row("Уязвимость", escape(report.vulnerable_quote))
    body.add_row("Защита", escape(report.suggested_fix))
    body.add_row("Источник", result.fix_source.value)
    return Panel(
        body,
        title=f"Этап 3 · Стресс-тест · {label}",
        border_style=tone,
        padding=(1, 2),
    )


if __name__ == "__main__":
    main()
