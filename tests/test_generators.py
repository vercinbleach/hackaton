from __future__ import annotations

import json
from pathlib import Path

import httpx

from cala_fastpath_training.catalog import load_catalog
from cala_fastpath_training.generators import GLiNERBaseGenerator, load_skill
from cala_fastpath_training.openai_responses import OpenAIResponsesClient, plan_output_schema

ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_catalog(ROOT / "training" / "config" / "catalog.json")


def test_openai_responses_uses_strict_plan_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        assert request.headers["Authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "gpt-5.6-luna"
        assert body["reasoning"] == {"effort": "high"}
        assert body["text"]["format"] == {
            "type": "json_schema",
            "name": "cala_query_plan",
            "strict": True,
            "schema": plan_output_schema(CATALOG),
        }
        output = {
            "operation": "knowledge_query",
            "root": "companies",
            "filters": [{"kind": "previous_job_eq", "mention": "Google", "value": "Google"}],
            "return": ["name", "founder"],
            "entity": None,
            "order_by": None,
            "limit": None,
            "limit_mention": None,
            "reason": None,
        }
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "model": "test-model",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": json.dumps(output)}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            },
        )

    with OpenAIResponsesClient("test-key", transport=httpx.MockTransport(handler)) as client:
        plan, _, usage = client.generate_plan(
            "compañías fundadas por ex empleados de Google",
            model="gpt-5.6-luna",
            reasoning_effort="high",
            instructions="test",
            catalog=CATALOG,
        )

    assert plan.return_fields == ["name", "founder"]
    assert usage == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}


def test_gliner_output_converts_to_shared_plan() -> None:
    generator = GLiNERBaseGenerator(model="unused", catalog=CATALOG)
    plan = generator._to_plan(
        {
            "plan_tags": [
                {"label": "root:companies", "confidence": 0.9},
                {"label": "return:founder", "confidence": 0.8},
            ]
        },
        {"previous_job_eq": [{"value": {"text": "Google", "confidence": 0.99}}]},
    )

    assert plan.operation == "knowledge_query"
    assert plan.root == "companies"
    assert plan.return_fields == ["founder"]
    assert plan.filters[0].value == "Google"


def test_project_skill_contains_projection_rule() -> None:
    content = load_skill(ROOT / "benchmark" / "skills" / "cala-query" / "SKILL.md")
    assert "only explicitly requested output properties" in content
