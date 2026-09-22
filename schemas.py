from enum import Enum

from pydantic import BaseModel, Field, field_validator


class FixSource(str, Enum):
    """Откуда взята формулировка, которой закрыли пролом."""

    REFERENCE = "reference"
    GENERATED = "generated"
    NONE = "none"


class RiskCategory(str, Enum):
    """Класс риска, к которому относится найденный пролом."""

    FINANCIAL = "Financial"
    LEGAL = "Legal"
    OPERATIONAL = "Operational"


class ContractModules(BaseModel):
    """Пять модулей договора, на которые режется исходный текст."""
    subject: str = Field(min_length=1, max_length=50_000, description="Предмет договора. О чем договорились стороны.")
    payment_terms: str = Field(min_length=1, max_length=50_000, description="Порядок расчетов. Суммы, авансы, сроки оплаты.")
    deadlines_and_acceptance: str = Field(
        min_length=1,
        max_length=50_000,
        description="Сроки и порядок приемки работ/услуг/товаров.",
    )
    liability: str = Field(
        min_length=1,
        max_length=50_000,
        description="Ответственность сторон, штрафы, неустойки, пени.",
    )
    dispute_resolution: str = Field(
        min_length=1,
        max_length=50_000,
        description="Порядок разрешения споров, суды, досудебный порядок.",
    )


class RiskReport(BaseModel):
    """Оценка того, насколько заданный конфликт проламывает договор."""

    score: int = Field(
        ge=1,
        le=10,
        description="Сила пролома: 1 — защита держит сценарий, 10 — защита отсутствует.",
    )
    category: RiskCategory = Field(
        description="Класс риска: Financial, Legal или Operational.",
    )
    vulnerable_quote: str = Field(
        min_length=1,
        max_length=1_500,
        description="Дословная подстрока текста договора, без пересказа. Одна формулировка, не весь договор.",
    )
    suggested_fix: str = Field(
        min_length=1,
        max_length=4_000,
        description="Формулировка, которая закрывает найденный пролом.",
    )


class AnalyzeRequest(BaseModel):
    """Вход анализа: текст договора и конфликт, который нужно прогнать."""

    contract_text: str = Field(min_length=1, max_length=50_000, description="Полный текст договора.")
    scenario: str = Field(min_length=1, max_length=4_000, description="Конфликтный сценарий для стресс-теста.")

    @field_validator("contract_text", "scenario")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Пустое значение")
        return value


class AnalyzeResponse(BaseModel):
    """Сводка трёх этапов: модули, проверенная оговорка и отчёт о риске."""

    modules: ContractModules
    best_clause: str | None = Field(
        default=None,
        description="Эталонная оговорка, которую саб-агент разрешил вставить. null, если поиск пуст или оговорка отклонена.",
    )
    clause_id: str | None = Field(
        default=None,
        description="id карточки из базы. null, если оговорка не принята.",
    )
    clause_relevant: bool = Field(description="Саб-агент подтвердил оговорку. False, если поиск ничего не вернул или вердикт отрицательный.")
    validation_reason: str = Field(default="", description="Почему оговорка принята или отклонена.")
    risk_report: RiskReport
    assembled_contract: str | None = Field(
        default=None,
        description="Текст договора, в котором уязвимая цитата заменена на защиту.",
    )
    fix_source: FixSource = Field(
        default=FixSource.NONE,
        description="reference — в текст подставлена оговорка из базы. none — сборка не делалась. Черновик модели в договор не вставляется.",
    )