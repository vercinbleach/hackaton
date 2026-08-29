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
    description: str


class Catalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operations: tuple[str, ...]
    roots: tuple[str, ...]
    returns: tuple[str, ...]
    order_by: tuple[str, ...]
    filters: dict[str, FilterSpec]
    reasons: tuple[str, ...]

    @property
    def classification_labels(self) -> list[str]:
        labels = [f"operation:{item}" for item in self.operations]
        labels += [f"root:{item}" for item in self.roots]
        labels += [f"return:{item}" for item in self.returns]
        labels += [f"order_by:{item}" for item in self.order_by]
        labels += [f"reason:{item}" for item in self.reasons]
        return labels

    def inference_schema(self) -> dict[str, Any]:
        structures: dict[str, Any] = {
            name: {
                "fields": [
                    {
                        "name": "value",
                        "dtype": "str",
                        "description": spec.description,
                    }
                ]
            }
            for name, spec in self.filters.items()
        }
        structures["entity_name"] = {
            "fields": [
                {
                    "name": "value",
                    "dtype": "str",
                    "description": "Exact name of the entity requested by the user",
                }
            ]
        }
        structures["limit_value"] = {
            "fields": [
                {
                    "name": "value",
                    "dtype": "str",
                    "description": "Maximum result count stated by the user",
                }
            ]
        }
        return {
            "classifications": [
                {
                    "task": "plan_tags",
                    "labels": self.classification_labels,
                    "multi_label": True,
                }
            ],
            "structures": structures,
        }


def load_catalog(path: Path) -> Catalog:
    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "operations": raw["operations"],
        "roots": raw["roots"],
        "returns": raw["returns"],
        "order_by": raw.get("order_by", []),
        "reasons": raw.get("unsupported_reasons", []),
        "filters": {name: {"name": name, **value} for name, value in raw["filters"].items()},
    }
    return Catalog.model_validate(payload)
