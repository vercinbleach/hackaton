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
from .models import PioneerRow, SplitResult, TrainingExample


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
        elif plan.root not in allowed_roots:
            raise DatasetValidationError(f"{prefix}: unknown root {plan.root!r}")
        for field in plan.return_fields:
            if field not in allowed_returns:
                raise DatasetValidationError(f"{prefix}: unknown return field {field!r}")
        if plan.order_by is not None and plan.order_by not in allowed_orders:
            raise DatasetValidationError(f"{prefix}: unknown order_by {plan.order_by!r}")
        for filter_value in plan.filters:
            if filter_value.kind not in catalog.filters:
                raise DatasetValidationError(f"{prefix}: unknown filter kind {filter_value.kind!r}")


def to_pioneer_row(row: TrainingExample) -> PioneerRow:
    plan = row.plan
    labels = [f"operation:{plan.operation}"]
    if plan.operation == "unsupported":
        labels.append(f"reason:{plan.reason}")
    else:
        labels.append(f"root:{plan.root}")
        labels.extend(f"return:{field}" for field in plan.return_fields)
        if plan.order_by:
            labels.append(f"order_by:{plan.order_by}")

    structures = [{item.kind: {"value": item.mention}} for item in plan.filters]
    if plan.entity:
        structures.append({"entity_name": {"value": plan.entity.mention}})
    if plan.limit is not None:
        structures.append({"limit_value": {"value": plan.limit_mention or ""}})

    return PioneerRow(
        text=row.text,
        labels=sorted(set(labels)),
        json_structures=structures or None,
    )


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
