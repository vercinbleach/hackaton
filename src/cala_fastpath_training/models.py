from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class FilterValue(StrictModel):
    kind: str = Field(min_length=1)
    mention: str = Field(min_length=1)
    value: str | int | float


class EntityMention(StrictModel):
    mention: str = Field(min_length=1)


class Plan(StrictModel):
    operation: str = Field(min_length=1)
    root: str | None = None
    filters: list[FilterValue] = Field(default_factory=list)
    return_fields: list[str] = Field(default_factory=list, alias="return")
    entity: EntityMention | None = None
    order_by: str | None = None
    limit: int | None = Field(default=None, ge=1, le=100)
    limit_mention: str | None = None
    reason: str | None = None


class TrainingExample(StrictModel):
    id: str = Field(min_length=1)
    group: str = Field(min_length=1)
    language: str = Field(min_length=2, max_length=16)
    text: str = Field(min_length=1)
    plan: Plan

    @model_validator(mode="after")
    def validate_verbatim_spans(self) -> TrainingExample:
        folded_text = self.text.casefold()
        for filter_value in self.plan.filters:
            if filter_value.mention.casefold() not in folded_text:
                raise ValueError(
                    f"filter mention {filter_value.mention!r} is not a verbatim text span"
                )
        if self.plan.entity and self.plan.entity.mention.casefold() not in folded_text:
            raise ValueError("entity mention is not a verbatim text span")
        if self.plan.limit is not None:
            if not self.plan.limit_mention:
                raise ValueError("limit requires limit_mention")
            if self.plan.limit_mention.casefold() not in folded_text:
                raise ValueError("limit_mention is not a verbatim text span")
        return self


class NerEntity(StrictModel):
    text: str = Field(min_length=1)
    label: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> NerEntity:
        if self.end != self.start + len(self.text):
            raise ValueError("NER entity end must be the exclusive end of text")
        return self


class PioneerRow(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(min_length=1)
    labels: list[str] = Field(min_length=1)
    entities: list[NerEntity] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_labels(self) -> PioneerRow:
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Pioneer labels must be unique")
        for entity in self.entities:
            if entity.end > len(self.text):
                raise ValueError("NER entity is outside the input text")
            if self.text[entity.start : entity.end] != entity.text:
                raise ValueError("NER entity offsets do not match its text")
        return self


class SplitResult(StrictModel):
    train: list[TrainingExample]
    validation: list[TrainingExample]
    test: list[TrainingExample]


class UploadReservation(BaseModel):
    model_config = ConfigDict(extra="allow")

    presigned_url: str
    dataset_id: str
    dataset_name: str
    version_number: str


class UploadedDataset(StrictModel):
    dataset_id: str
    dataset_name: str
    version_number: str


class EvaluationReference(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str


class EvaluationStart(BaseModel):
    model_config = ConfigDict(extra="allow")

    evaluations: list[EvaluationReference]


class CalaQueryResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    results: list[dict[str, Any]] = Field(default_factory=list)
    entities: list[dict[str, Any]] | None = None


class BenchmarkCase(StrictModel):
    id: str = Field(min_length=1)
    query: str = Field(min_length=1)


class GenerationRecord(StrictModel):
    case_id: str
    query: str
    system: str
    model: str
    reasoning_effort: str | None = None
    plan: Plan | None = None
    cala_query: str | None = None
    latency_ms: float = Field(ge=0)
    usage: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None
    decision: Literal["accepted", "abstained"] | None = None
    abstention_reason: str | None = None
    threshold: float | None = Field(default=None, ge=0, le=1)


JsonObject = dict[str, Any]
