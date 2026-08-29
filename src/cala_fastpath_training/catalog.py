from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class FilterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    path: tuple[str, ...]
    operator: str
    structure: str
    field: str
    description: str


class StructureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str
    description: str


class Catalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operations: tuple[str, ...]
    roots: tuple[str, ...]
    returns: tuple[str, ...]
    order_by: tuple[str, ...]
    filters: dict[str, FilterSpec]
    reasons: tuple[str, ...]
    special_structures: dict[str, StructureSpec]

    @property
    def classification_labels(self) -> list[str]:
        """Return the flat labels shared by Pioneer training and inference."""
        labels = [f"operation:{item}" for item in self.operations]
        labels += [f"root:{item}" for item in self.roots]
        labels += [f"filter:{item}" for item in self.filters]
        labels += [f"return:{item}" for item in self.returns if item != "name"]
        labels += [f"order_by:{item}" for item in self.order_by]
        labels += [f"reason:{item}" for item in self.reasons]
        return labels

    @property
    def classification_tasks(self) -> dict[str, dict[str, Any]]:
        return {
            "plan_labels": {
                "labels": self.classification_labels,
                "multi_label": True,
            }
        }

    @property
    def structures(self) -> dict[str, list[dict[str, str]]]:
        structures = {
            spec.structure: [
                {
                    "name": spec.field,
                    "dtype": "str",
                    "description": spec.description,
                }
            ]
            for spec in self.filters.values()
        }
        structures.update(
            {
                name: [
                    {
                        "name": spec.field,
                        "dtype": "str",
                        "description": spec.description,
                    }
                ]
                for name, spec in self.special_structures.items()
            }
        )
        return structures

    def inference_schema(self) -> dict[str, Any]:
        return {
            "classifications": [
                {
                    "task": task,
                    **config,
                }
                for task, config in self.classification_tasks.items()
            ],
            "entities": self.ner_entities,
        }

    @property
    def ner_labels(self) -> list[str]:
        """Semantic entity labels used by GLiNER2 for generated slot data."""
        return [spec.structure for spec in self.filters.values()] + list(self.special_structures)

    @property
    def ner_entities(self) -> dict[str, str]:
        """Entity labels and descriptions used at inference time."""
        entities = {spec.structure: spec.description for spec in self.filters.values()}
        entities.update(
            {name: spec.description for name, spec in self.special_structures.items()}
        )
        return entities


def load_catalog(path: Path) -> Catalog:
    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "operations": raw["operations"],
        "roots": raw["roots"],
        "returns": raw["returns"],
        "order_by": raw.get("order_by", []),
        "reasons": raw.get("unsupported_reasons", []),
        "filters": {name: {"name": name, **value} for name, value in raw["filters"].items()},
        "special_structures": raw.get("special_structures", {}),
    }
    return Catalog.model_validate(payload)
