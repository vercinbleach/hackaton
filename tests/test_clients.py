from __future__ import annotations

import json
from pathlib import Path

import httpx

from cala_fastpath_training.cala import CalaClient
from cala_fastpath_training.pioneer import PioneerClient


def test_cala_knowledge_query_uses_structured_endpoint_and_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/knowledge/query"
        assert request.headers["X-API-KEY"] == "test-key"
        assert json.loads(request.content) == {
            "input": "companies.location=Spain.return(name)",
            "return_entities": False,
        }
        return httpx.Response(200, json={"results": [{"name": "Acme"}], "entities": None})

    with CalaClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
        result = client.knowledge_query(
            "companies.location=Spain.return(name)",
            return_entities=False,
        )

    assert result.results == [{"name": "Acme"}]


def test_pioneer_upload_runs_reserve_put_process(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/felix/datasets/upload/url":
            assert request.headers["X-API-Key"] == "test-key"
            return httpx.Response(
                200,
                json={
                    "presigned_url": "https://storage.example/upload",
                    "dataset_id": "dataset-1",
                    "dataset_name": "fastpath-train",
                    "version_number": "1",
                },
            )
        if request.url.host == "storage.example":
            assert request.content == b'{"text":"example"}\n'
            return httpx.Response(200)
        assert json.loads(request.content) == {"dataset_id": "dataset-1"}
        return httpx.Response(200, json={"status": "processing"})

    dataset = tmp_path / "train.jsonl"
    dataset.write_bytes(b'{"text":"example"}\n')
    with PioneerClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
        uploaded = client.upload_dataset(
            dataset,
            dataset_name="fastpath-train",
            purpose="training",
        )

    assert uploaded.dataset_id == "dataset-1"
    assert calls == [
        ("POST", "/felix/datasets/upload/url"),
        ("PUT", "/upload"),
        ("POST", "/felix/datasets/upload/process"),
    ]
