from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from cala_fastpath_training.cala import CalaClient
from cala_fastpath_training.models import UploadedDataset
from cala_fastpath_training.pioneer import PioneerClient, PioneerError


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


def test_pioneer_upload_excludes_api_key_from_storage_request(tmp_path: Path) -> None:
    """Regression test: X-API-Key must not be sent to presigned URL storage host."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/felix/datasets/upload/url":
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
            # The storage request must NOT contain the API key
            assert "X-API-Key" not in request.headers
            assert "x-api-key" not in request.headers
            return httpx.Response(200)
        return httpx.Response(200, json={})

    dataset = tmp_path / "train.jsonl"
    dataset.write_bytes(b'{"text":"example"}\n')
    with PioneerClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
        client.upload_dataset(
            dataset,
            dataset_name="fastpath-train",
            purpose="training",
        )


def test_pioneer_upload_uses_content_type_signed_in_url(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/upload/url"):
            return httpx.Response(200, json={
                "presigned_url": "https://storage.example/upload?content-type=application%2Foctet-stream",
                "dataset_id": "dataset-1", "dataset_name": "fastpath-train", "version_number": "1",
            })
        if request.url.host == "storage.example":
            assert request.headers["Content-Type"] == "application/octet-stream"
            assert "X-API-Key" not in request.headers
            return httpx.Response(200)
        return httpx.Response(200, json={})

    dataset = tmp_path / "train.jsonl"
    dataset.write_bytes(b'{}\n')
    with PioneerClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
        client.upload_dataset(dataset, dataset_name="fastpath-train", purpose="training")


def test_pioneer_upload_rejects_reservation_name_mismatch(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "presigned_url": "https://storage.example/upload", "dataset_id": "dataset-1",
            "dataset_name": "other", "version_number": "1",
        })

    dataset = tmp_path / "train.jsonl"
    dataset.write_bytes(b'{}\n')
    with (
        PioneerClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client,
        pytest.raises(PioneerError, match="unexpected name"),
    ):
        client.upload_dataset(dataset, dataset_name="fastpath-train", purpose="training")


def test_pioneer_upload_rejects_http_urls(tmp_path: Path) -> None:
    """Regression test: HTTP presigned URLs must be rejected before upload."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/felix/datasets/upload/url":
            return httpx.Response(
                200,
                json={
                    "presigned_url": "http://storage.example/upload",
                    "dataset_id": "dataset-1",
                    "dataset_name": "fastpath-train",
                    "version_number": "1",
                },
            )
        # Should never reach here
        raise AssertionError("Upload should have been rejected")

    dataset = tmp_path / "train.jsonl"
    dataset.write_bytes(b'{"text":"example"}\n')
    with (
        PioneerClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client,
        pytest.raises(PioneerError, match="presigned URL must use HTTPS"),
    ):
        client.upload_dataset(
            dataset,
            dataset_name="fastpath-train",
            purpose="training",
        )


def test_pioneer_upload_rejects_urls_without_hostname(tmp_path: Path) -> None:
    """Regression test: presigned URLs without hostname must be rejected."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/felix/datasets/upload/url":
            return httpx.Response(
                200,
                json={
                    "presigned_url": "https:///upload",
                    "dataset_id": "dataset-1",
                    "dataset_name": "fastpath-train",
                    "version_number": "1",
                },
            )
        # Should never reach here
        raise AssertionError("Upload should have been rejected")

    dataset = tmp_path / "train.jsonl"
    dataset.write_bytes(b'{"text":"example"}\n')
    with (
        PioneerClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client,
        pytest.raises(PioneerError, match="presigned URL must have a valid hostname"),
    ):
        client.upload_dataset(
            dataset,
            dataset_name="fastpath-train",
            purpose="training",
        )


def test_pioneer_client_rejects_http_base_url() -> None:
    with pytest.raises(ValueError, match="base_url must use HTTPS"):
        PioneerClient(api_key="test-key", base_url="http://api.pioneer.ai")


def test_pioneer_client_rejects_empty_base_url() -> None:
    with pytest.raises(ValueError, match="base_url cannot be empty"):
        PioneerClient(api_key="test-key", base_url="")


def test_pioneer_client_rejects_base_url_without_hostname() -> None:
    with pytest.raises(ValueError, match="base_url must have a valid hostname"):
        PioneerClient(api_key="test-key", base_url="https:///path")


def test_pioneer_client_accepts_valid_https_base_url() -> None:
    client = PioneerClient(api_key="test-key", base_url="https://custom.pioneer.ai")
    client.close()


def test_pioneer_client_strips_trailing_slash_from_base_url() -> None:
    client = PioneerClient(api_key="test-key", base_url="https://api.pioneer.ai/")
    assert client._client.base_url == "https://api.pioneer.ai"
    client.close()


@pytest.mark.parametrize(
    "base_url, message",
    [
        ("https://user@api.pioneer.ai", "user information"),
        ("https://api.pioneer.ai?target=other", "query or fragment"),
        ("https://api.pioneer.ai/#fragment", "query or fragment"),
    ],
)
def test_pioneer_client_rejects_ambiguous_base_urls(base_url: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PioneerClient(api_key="test-key", base_url=base_url)


def test_wait_for_dataset_polls_the_uploaded_version() -> None:
    uploaded = UploadedDataset(
        dataset_id="dataset-1", dataset_name="fastpath-train", version_number="7"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/felix/datasets/fastpath-train/7"
        return httpx.Response(
            200,
            json={"id": "dataset-1", "version_number": "7", "status": "ready"},
        )

    with PioneerClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
        assert client.wait_for_dataset(uploaded, interval=0, timeout=1)["status"] == "ready"


def test_pioneer_starts_classification_generation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/generate"
        assert json.loads(request.content) == {
            "task_type": "classification",
            "dataset_name": "cala-plan-labels",
            "labels": ["operation:knowledge_query", "operation:unsupported"],
            "num_examples": 100,
            "domain_description": "Bilingual Cala queries",
        }
        return httpx.Response(200, json={"job_id": "generation-1", "status": "queued"})

    with PioneerClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
        result = client.start_generation(
            task_type="classification",
            dataset_name="cala-plan-labels",
            labels=["operation:knowledge_query", "operation:unsupported"],
            num_examples=100,
            domain_description="Bilingual Cala queries",
        )

    assert result["job_id"] == "generation-1"


def test_pioneer_starts_ner_generation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {
            "task_type": "ner",
            "dataset_name": "cala-plan-entities",
            "labels": ["target_entity", "result_limit"],
            "num_examples": 50,
        }
        return httpx.Response(200, json={"id": "generation-2", "status": "queued"})

    with PioneerClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
        result = client.start_generation(
            task_type="ner",
            dataset_name="cala-plan-entities",
            labels=["target_entity", "result_limit"],
            num_examples=50,
        )

    assert result["id"] == "generation-2"


@pytest.mark.parametrize("terminal", ["ready", "complete"])
def test_wait_for_generation_accepts_documented_success_states(terminal: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.raw_path == b"/generate/jobs/job%2F1"
        return httpx.Response(200, json={"job_id": "job/1", "status": terminal})

    with PioneerClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
        assert client.wait_for_generation("job/1", interval=0, timeout=1)["status"] == terminal


def test_wait_for_generation_surfaces_provider_error() -> None:
    with (
        PioneerClient(
            api_key="test-key",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"job_id": "bad", "status": "failed", "error": "no valid samples"},
                )
            ),
        ) as client,
        pytest.raises(PioneerError, match="no valid samples"),
    ):
        client.wait_for_generation("bad", interval=0, timeout=1)


def test_pioneer_training_accepts_multiple_datasets() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/felix/training-jobs"
        body = json.loads(request.content)
        assert body["datasets"] == [
            {"name": "cala-plan-labels"},
            {"name": "cala-plan-entities"},
        ]
        return httpx.Response(200, json={"id": "training-1"})

    with PioneerClient(api_key="test-key", transport=httpx.MockTransport(handler)) as client:
        result = client.start_training(
            model_name="cala-fastpath",
            dataset_names=["cala-plan-labels", "cala-plan-entities"],
            base_model="fastino/gliner2-multi-v1",
            epochs=5,
            learning_rate=5e-5,
        )

    assert result["id"] == "training-1"


@pytest.mark.parametrize(
    "response, message",
    [
        ({"id": "other", "version_number": "7", "status": "ready"}, "unexpected ID"),
        ({"id": "dataset-1", "version_number": "8", "status": "ready"}, "unexpected version"),
    ],
)
def test_wait_for_dataset_rejects_identity_mismatch(response: dict[str, str], message: str) -> None:
    uploaded = UploadedDataset(
        dataset_id="dataset-1", dataset_name="fastpath-train", version_number="7"
    )

    with (
        PioneerClient(
            api_key="test-key",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)),
        ) as client,
        pytest.raises(PioneerError, match=message),
    ):
        client.wait_for_dataset(uploaded, interval=0, timeout=1)
