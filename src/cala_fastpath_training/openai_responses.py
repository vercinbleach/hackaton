from __future__ import annotations

import os

import httpx
from pydantic import ValidationError

from .catalog import Catalog
from .models import JsonObject, Plan


class OpenAIResponsesError(RuntimeError):
    pass


def plan_output_schema(catalog: Catalog) -> JsonObject:
    nullable_root = {"anyOf": [{"type": "string", "enum": list(catalog.roots)}, {"type": "null"}]}
    nullable_order = {
        "anyOf": [{"type": "string", "enum": list(catalog.order_by)}, {"type": "null"}]
    }
    nullable_reason = {
        "anyOf": [{"type": "string", "enum": list(catalog.reasons)}, {"type": "null"}]
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "operation": {"type": "string", "enum": list(catalog.operations)},
            "root": nullable_root,
            "filters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "kind": {"type": "string", "enum": list(catalog.filters)},
                        "mention": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["kind", "mention", "value"],
                },
            },
            "return": {"type": "array", "items": {"type": "string", "enum": list(catalog.returns)}},
            "entity": {
                "anyOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"mention": {"type": "string"}},
                        "required": ["mention"],
                    },
                    {"type": "null"},
                ]
            },
            "order_by": nullable_order,
            "limit": {
                "anyOf": [{"type": "integer", "minimum": 1, "maximum": 100}, {"type": "null"}]
            },
            "limit_mention": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "reason": nullable_reason,
        },
        "required": [
            "operation",
            "root",
            "filters",
            "return",
            "entity",
            "order_by",
            "limit",
            "limit_mention",
            "reason",
        ],
    }


class OpenAIResponsesClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 180,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> OpenAIResponsesClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @classmethod
    def from_environment(cls) -> OpenAIResponsesClient:
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    def generate_plan(
        self,
        query: str,
        *,
        model: str,
        reasoning_effort: str,
        instructions: str,
        catalog: Catalog,
    ) -> tuple[Plan, JsonObject, JsonObject | None]:
        payload = {
            "model": model,
            "reasoning": {"effort": reasoning_effort},
            "instructions": instructions,
            "input": query,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "cala_query_plan",
                    "strict": True,
                    "schema": plan_output_schema(catalog),
                }
            },
        }
        try:
            response = self._client.post("/responses", json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            raise OpenAIResponsesError(
                f"OpenAI Responses returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OpenAIResponsesError(f"OpenAI Responses failed: {exc}") from exc
        except ValueError as exc:
            raise OpenAIResponsesError("OpenAI Responses returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise OpenAIResponsesError("OpenAI Responses returned a non-object")

        output_text = body.get("output_text")
        if not isinstance(output_text, str):
            output_text = self._find_output_text(body)
        if not output_text:
            raise OpenAIResponsesError("OpenAI Responses did not return output text")
        try:
            plan = Plan.model_validate_json(output_text)
        except ValidationError as exc:
            raise OpenAIResponsesError(f"OpenAI returned an invalid plan: {exc}") from exc
        usage = body.get("usage")
        return plan, body, usage if isinstance(usage, dict) else None

    @staticmethod
    def _find_output_text(body: JsonObject) -> str | None:
        for item in body.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        return text
        return None


def compact_openai_raw(body: JsonObject) -> JsonObject:
    return {
        key: body[key]
        for key in ("id", "model", "status", "output_text", "usage", "error", "incomplete_details")
        if key in body
    }
