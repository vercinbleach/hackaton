from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .catalog import Catalog
from .compiler import PlanCompilationError, validate_plan_for_fastpath
from .models import NerEntity, PioneerRow, SplitResult, TrainingExample


class DatasetValidationError(ValueError):
    pass


def read_jsonl(path: Path) -> list[TrainingExample]:
    rows: list[TrainingExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(TrainingExample.model_validate_json(line))
        except ValidationError as exc:
            raise DatasetValidationError(f"{path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[TrainingExample | PioneerRow]) -> None:
    """Write training examples or pioneer rows to a JSONL file securely.

    This function protects against symlink attacks by:
    1. Checking if the destination path is a symlink
    2. Writing to a temporary file first
    3. Atomically replacing the destination with os.replace()

    Args:
        path: Destination file path
        rows: Iterable of training examples or pioneer rows to write

    Raises:
        ValueError: If the destination path is a symlink
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Check if the destination is a symlink before writing
    if path.exists() and path.is_symlink():
        raise ValueError(f"Refusing to write to symlink: {path}")

    # Write to a temporary file in the same directory to ensure atomic replacement
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            payload = row.model_dump(by_alias=True, exclude_none=True)
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    try:
        # Verify the destination is still not a symlink (TOCTOU mitigation)
        if path.exists() and path.is_symlink():
            raise ValueError(f"Refusing to write to symlink: {path}")
        # Atomically replace the destination file
        os.replace(temporary, path)
    except Exception:
        # Clean up temporary file on error
        temporary.unlink(missing_ok=True)
        raise


def validate_examples(rows: list[TrainingExample], catalog: Catalog) -> None:
    if not rows:
        raise DatasetValidationError("dataset is empty")
    ids: set[str] = set()
    allowed_operations = set(catalog.operations)
    allowed_roots = set(catalog.roots)
    allowed_returns = set(catalog.returns)
    allowed_orders = set(catalog.order_by)
    allowed_reasons = set(catalog.reasons)

    for index, row in enumerate(rows, 1):
        prefix = f"row {index}"
        if row.id in ids:
            raise DatasetValidationError(f"{prefix}: duplicate id {row.id!r}")
        ids.add(row.id)
        plan = row.plan
        if plan.operation not in allowed_operations:
            raise DatasetValidationError(f"{prefix}: unknown operation {plan.operation!r}")
        if plan.operation == "unsupported":
            if plan.reason not in allowed_reasons:
                raise DatasetValidationError(f"{prefix}: unsupported plan needs a known reason")
            if any(
                (
                    plan.root is not None,
                    bool(plan.filters),
                    bool(plan.return_fields),
                    plan.entity is not None,
                    plan.order_by is not None,
                    plan.limit is not None,
                )
            ):
                raise DatasetValidationError(f"{prefix}: unsupported plan has executable fields")
        elif plan.operation == "retrieve_entity":
            if plan.root not in allowed_roots:
                raise DatasetValidationError(f"{prefix}: unknown root {plan.root!r}")
            if plan.entity is None or not plan.return_fields:
                raise DatasetValidationError(
                    f"{prefix}: retrieve_entity needs an entity and result projection"
                )
            if plan.filters or plan.reason or plan.order_by or plan.limit:
                raise DatasetValidationError(
                    f"{prefix}: retrieve_entity has incompatible executable fields"
                )
        elif plan.root not in allowed_roots:
            raise DatasetValidationError(f"{prefix}: unknown root {plan.root!r}")
        elif plan.operation == "knowledge_query":
            try:
                validate_plan_for_fastpath(plan, catalog)
            except PlanCompilationError as exc:
                raise DatasetValidationError(f"{prefix}: {exc}") from exc
        for field in plan.return_fields:
            if field not in allowed_returns:
                raise DatasetValidationError(f"{prefix}: unknown return field {field!r}")
        if plan.order_by is not None and plan.order_by not in allowed_orders:
            raise DatasetValidationError(f"{prefix}: unknown order_by {plan.order_by!r}")
        for filter_value in plan.filters:
            if filter_value.kind not in catalog.filters:
                raise DatasetValidationError(f"{prefix}: unknown filter kind {filter_value.kind!r}")


def to_pioneer_row(row: TrainingExample, catalog: Catalog) -> PioneerRow:
    plan = row.plan
    selected_labels = {f"operation:{plan.operation}"}
    if plan.root:
        selected_labels.add(f"root:{plan.root}")
    selected_labels.update(f"return:{field}" for field in plan.return_fields if field != "name")
    selected_labels.update(f"filter:{item.kind}" for item in plan.filters)
    if plan.order_by:
        selected_labels.add(f"order_by:{plan.order_by}")
    if plan.reason:
        selected_labels.add(f"reason:{plan.reason}")
    labels = [label for label in catalog.classification_labels if label in selected_labels]
    if set(labels) != selected_labels:
        unknown = sorted(selected_labels - set(labels))
        raise DatasetValidationError(f"plan cannot be flattened into catalog labels: {unknown!r}")

    entities: list[NerEntity] = []

    def annotation(mention: str, label: str) -> NerEntity:
        start = row.text.casefold().index(mention.casefold())
        text = row.text[start : start + len(mention)]
        return NerEntity(text=text, label=label, start=start, end=start + len(text))

    entities.extend(
        annotation(item.mention, catalog.filters[item.kind].structure) for item in plan.filters
    )
    if plan.entity:
        entities.append(annotation(plan.entity.mention, "target_entity"))
    if plan.limit is not None:
        if not plan.limit_mention:
            raise DatasetValidationError("limit requires a verbatim limit_mention")
        entities.append(annotation(plan.limit_mention, "result_limit"))

    return PioneerRow(
        text=row.text,
        labels=labels,
        entities=entities,
    )


def _pioneer_to_gliner_row(row: PioneerRow, catalog: Catalog) -> dict[str, Any]:
    classifications = [
        {
            "task": task,
            "labels": config["labels"],
            "true_label": row.labels,
            "multi_label": config["multi_label"],
        }
        for task, config in catalog.classification_tasks.items()
    ]
    entities: dict[str, list[str]] = {}
    for entity in row.entities:
        entities.setdefault(entity.label, []).append(entity.text)
    return {
        "input": row.text,
        "output": {
            "classifications": classifications,
            "entities": entities,
            "entity_descriptions": {
                label: catalog.ner_entities[label] for label in entities
            },
        },
    }


def validate_pioneer_jsonl(path: Path, catalog: Catalog) -> dict[str, Any]:
    """Validate Pioneer upload rows and their conversion to native GLiNER2 records."""
    from gliner2.training.data import InputExample, TrainingDataset

    rows: list[PioneerRow] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(PioneerRow.model_validate_json(line))
        except ValidationError as exc:
            raise DatasetValidationError(f"{path}:{line_number}: {exc}") from exc
    if not rows:
        raise DatasetValidationError("Pioneer dataset is empty")

    allowed_labels = set(catalog.classification_labels) - {"return:name"}
    allowed_entities = set(catalog.ner_labels)
    converted = []
    for index, row in enumerate(rows, 1):
        unknown_labels = set(row.labels) - allowed_labels
        if unknown_labels:
            raise DatasetValidationError(
                f"row {index}: unknown Pioneer labels {sorted(unknown_labels)!r}"
            )
        for entity in row.entities:
            if entity.label not in allowed_entities:
                raise DatasetValidationError(
                    f"row {index}: unknown entity label {entity.label!r}"
                )
        converted.append(InputExample.from_dict(_pioneer_to_gliner_row(row, catalog)))
    return TrainingDataset(converted).validate()


def grouped_split(
    rows: list[TrainingExample],
    *,
    seed: int,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> SplitResult:
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("ratios must leave non-empty train, validation, and test ranges")
    groups: dict[str, list[TrainingExample]] = {}
    for row in rows:
        groups.setdefault(row.group, []).append(row)
    if len(groups) < 3:
        raise DatasetValidationError("at least three groups are required for grouped splitting")

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    group_count = len(keys)
    train_group_count = max(1, round(group_count * train_ratio))
    validation_group_count = max(1, round(group_count * validation_ratio))
    if train_group_count + validation_group_count >= group_count:
        train_group_count = group_count - validation_group_count - 1
    train_keys = keys[:train_group_count]
    validation_keys = keys[train_group_count : train_group_count + validation_group_count]
    test_keys = keys[train_group_count + validation_group_count :]
    result = SplitResult(
        train=[row for key in train_keys for row in groups[key]],
        validation=[row for key in validation_keys for row in groups[key]],
        test=[row for key in test_keys for row in groups[key]],
    )
    if any(not value for value in (result.train, result.validation, result.test)):
        raise DatasetValidationError("group sizes produced an empty split")
    return result


def dataset_summary(rows: list[TrainingExample]) -> dict[str, Any]:
    operations = Counter(row.plan.operation for row in rows)
    languages = Counter(row.language for row in rows)
    filters = Counter(item.kind for row in rows for item in row.plan.filters)
    canonical = "".join(
        json.dumps(
            row.model_dump(by_alias=True, exclude_none=True),
            ensure_ascii=False,
            sort_keys=True,
        )
        for row in rows
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "examples": len(rows),
        "groups": len({row.group for row in rows}),
        "operations": dict(sorted(operations.items())),
        "languages": dict(sorted(languages.items())),
        "filters": dict(sorted(filters.items())),
        "sha256": digest,
    }
